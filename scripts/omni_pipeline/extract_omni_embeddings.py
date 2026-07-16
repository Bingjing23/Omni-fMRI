#!/usr/bin/env python3
"""Extract one 768-dimensional Omni-fMRI CLS embedding per UKB subject.

This wrapper reuses the repository's existing `extract_feat.py` model,
checkpoint, tensor conversion, and token extraction functions. It adds
manifest-driven subject alignment, optional NIfTI preprocessing, segment-level
aggregation, TSV export, and failure/QC logs for HPC batch use.

Inputs:
  - Manifest TSV with columns `eid` and `image_path`.
  - Omni-fMRI checkpoint and config.

Outputs:
  - embeddings TSV: eid + emb_001 ... emb_768.
  - failure log TSV.
  - missing-subject log TSV.
  - QC summary TSV.

Example:
  python scripts/omni_pipeline/extract_omni_embeddings.py \
    --manifest manifests/ukb_omni_manifest.tsv \
    --checkpoint pretrain_checkpoint/checkpoint.pth \
    --output-tsv outputs/omni_embeddings/embeddings.tsv \
    --work-dir outputs/omni_embeddings/work \
    --input-kind npz \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract manifest-aligned Omni-fMRI CLS embeddings.")
    parser.add_argument("--manifest", required=True, help="TSV with eid and image_path columns.")
    parser.add_argument("--output-tsv", required=True, help="Output embeddings TSV.")
    parser.add_argument("--work-dir", required=True, help="Working directory for temporary NPZ and logs.")
    parser.add_argument("--checkpoint", default="pretrain_checkpoint/checkpoint.pth", help="Omni checkpoint.")
    parser.add_argument("--config", default="configs/pretrain.yaml", help="Fallback config if checkpoint lacks one.")
    parser.add_argument("--input-kind", choices=["npz", "nifti"], default="npz", help="Manifest image path type.")
    parser.add_argument("--npz-key", default=None, help="Array key inside NPZ. Default: first key.")
    parser.add_argument("--layout", choices=["auto", "dhwt", "cdhw"], default="auto", help="NPZ array layout.")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame for DHWT arrays longer than in_chans.")
    parser.add_argument("--pad-short", action="store_true", help="Zero-pad short DHWT arrays.")
    parser.add_argument("--segment-length", type=int, default=40, help="NIfTI preprocessing segment length.")
    parser.add_argument("--target-shape", nargs=3, type=int, default=[96, 96, 96], help="NIfTI target shape.")
    parser.add_argument(
        "--segment-aggregation",
        choices=["mean", "first"],
        default="mean",
        help="How to combine multiple segment CLS tokens per subject. Default: mean.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for real extraction. Default: cuda:0")
    parser.add_argument("--eid-column", default="eid")
    parser.add_argument("--path-column", default="image_path")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N row limit for dry runs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest and paths without loading model.")
    return parser.parse_args()


def load_config(args: argparse.Namespace, checkpoint: object | None) -> dict:
    from src.utils.cli_app import YamlBackedCliApp
    from src.utils.config_overrides import apply_pretrain_overrides

    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if config is None:
        app = YamlBackedCliApp()
        config = app.load_yaml_config(Path(args.config))

    class EmptyOverrides:
        cfg_options = None
        add_args = None

    apply_pretrain_overrides(config, EmptyOverrides())
    return config


def emb_columns(width: int) -> list[str]:
    return [f"emb_{index:03d}" for index in range(1, width + 1)]


def prepare_outputs(output_tsv: Path, work_dir: Path, force: bool) -> dict[str, Path]:
    if output_tsv.exists() and not force:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {output_tsv}")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    return {
        "failures": output_tsv.with_suffix(".failures.tsv"),
        "missing": output_tsv.with_suffix(".missing_subjects.tsv"),
        "qc": output_tsv.with_suffix(".qc_summary.tsv"),
        "segment_qc": output_tsv.with_suffix(".segment_qc.tsv"),
    }


def read_manifest(path: Path, eid_column: str, path_column: str, limit: int | None) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t", dtype=str)
    missing = [column for column in (eid_column, path_column) if column not in manifest.columns]
    if missing:
        raise KeyError(f"Missing manifest columns: {missing}")
    manifest = manifest[[eid_column, path_column]].rename(columns={eid_column: "eid", path_column: "image_path"})
    manifest["eid"] = manifest["eid"].fillna("").astype(str).str.strip()
    manifest["image_path"] = manifest["image_path"].fillna("").astype(str).str.strip()
    manifest = manifest[manifest["eid"] != ""].copy()
    if limit is not None:
        manifest = manifest.head(limit)
    return manifest


def npz_files_for_subject(path: Path, input_kind: str, work_dir: Path, args: argparse.Namespace) -> list[Path]:
    if input_kind == "npz":
        if path.is_dir():
            files = sorted(path.glob("*.npz"))
            if not files:
                raise FileNotFoundError(f"No NPZ files in directory: {path}")
            return files
        if not path.is_file():
            raise FileNotFoundError(f"NPZ path does not exist: {path}")
        if path.suffix != ".npz":
            raise ValueError(f"Expected .npz path, got: {path}")
        return [path]

    if not path.is_file():
        raise FileNotFoundError(f"NIfTI path does not exist: {path}")
    from data_preparation.preprocessing import process_single_subject

    subject_work = work_dir / "preprocessed" / path.stem.replace(".nii", "")
    subject_work.mkdir(parents=True, exist_ok=True)
    ok = process_single_subject(
        path,
        subject_work,
        target_shape=tuple(args.target_shape),
        segment_length=args.segment_length,
    )
    if not ok:
        raise RuntimeError(f"NIfTI preprocessing failed: {path}")
    files = sorted(subject_work.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"NIfTI preprocessing produced no NPZ files: {path}")
    return files


def extract_one_npz(npz_path: Path, backbone, spatial_size: tuple[int, int, int], in_chans: int, args) -> np.ndarray:
    import torch
    from extract_feat import array_to_model_tensor, extract_tokens, load_npz_array

    array = load_npz_array(npz_path, args.npz_key)
    sample = array_to_model_tensor(
        array,
        in_chans=in_chans,
        spatial_size=spatial_size,
        layout=args.layout,
        start_frame=args.start_frame,
        pad_short=args.pad_short,
    ).to(device=torch.device(args.device), non_blocking=True)
    cls_token, _, _ = extract_tokens(backbone, sample)
    return np.asarray(cls_token, dtype=np.float32)


def aggregate_segments(tokens: list[np.ndarray], method: str) -> np.ndarray:
    if not tokens:
        raise ValueError("No segment tokens to aggregate.")
    if method == "first":
        return tokens[0]
    stacked = np.vstack(tokens)
    return stacked.mean(axis=0).astype(np.float32)


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_tsv = Path(args.output_tsv)
    work_dir = Path(args.work_dir)
    aux_paths = prepare_outputs(output_tsv, work_dir, args.force)
    manifest = read_manifest(Path(args.manifest), args.eid_column, args.path_column, args.limit)

    missing_rows = [
        {"eid": row.eid, "image_path": row.image_path, "reason": "image_path missing or does not exist"}
        for row in manifest.itertuples(index=False)
        if not row.image_path or not Path(row.image_path).exists()
    ]

    if args.dry_run:
        print(f"DRY_RUN manifest_rows={len(manifest)} missing_paths={len(missing_rows)}")
        write_rows(aux_paths["missing"], ["eid", "image_path", "reason"], missing_rows)
        return 0

    import torch
    from extract_feat import create_backbone, load_checkpoint

    checkpoint = load_checkpoint(Path(args.checkpoint).expanduser().resolve())
    config = load_config(args, checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    backbone = create_backbone(checkpoint, config, device)
    in_chans = int(backbone.patch_embed.proj.in_channels)
    spatial_size = tuple(int(x) for x in backbone.img_size)

    embedding_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    width: int | None = None

    for row in manifest.itertuples(index=False):
        eid = str(row.eid)
        image_path = Path(str(row.image_path)).expanduser()
        if not image_path.exists():
            failure_rows.append({"eid": eid, "image_path": str(image_path), "error": "path does not exist"})
            continue
        try:
            npz_files = npz_files_for_subject(image_path, args.input_kind, work_dir, args)
            tokens = [
                extract_one_npz(npz_path, backbone, spatial_size, in_chans, args)
                for npz_path in npz_files
            ]
            embedding = aggregate_segments(tokens, args.segment_aggregation)
            if width is None:
                width = int(embedding.shape[0])
            if int(embedding.shape[0]) != width:
                raise ValueError(f"Inconsistent embedding width: {embedding.shape[0]} != {width}")
            out = {"eid": eid}
            out.update({name: float(value) for name, value in zip(emb_columns(width), embedding)})
            embedding_rows.append(out)
            segment_rows.append(
                {
                    "eid": eid,
                    "image_path": str(image_path),
                    "segments": len(tokens),
                    "embedding_width": width,
                    "aggregation": args.segment_aggregation,
                }
            )
        except Exception as exc:
            failure_rows.append({"eid": eid, "image_path": str(image_path), "error": repr(exc)})

    if width is None:
        width = 768
    write_rows(output_tsv, ["eid"] + emb_columns(width), embedding_rows)
    write_rows(aux_paths["failures"], ["eid", "image_path", "error"], failure_rows)
    write_rows(aux_paths["missing"], ["eid", "image_path", "reason"], missing_rows)
    write_rows(aux_paths["segment_qc"], ["eid", "image_path", "segments", "embedding_width", "aggregation"], segment_rows)

    values = np.asarray([[row[col] for col in emb_columns(width)] for row in embedding_rows], dtype=float)
    finite = int(np.isfinite(values).sum()) if values.size else 0
    total = int(values.size)
    qc_rows = [
        {"metric": "manifest_rows", "value": len(manifest)},
        {"metric": "embedded_subjects", "value": len(embedding_rows)},
        {"metric": "failed_subjects", "value": len(failure_rows)},
        {"metric": "missing_paths", "value": len(missing_rows)},
        {"metric": "embedding_width", "value": width},
        {"metric": "finite_embedding_values", "value": finite},
        {"metric": "total_embedding_values", "value": total},
        {"metric": "finite_fraction", "value": finite / total if total else math.nan},
    ]
    write_rows(aux_paths["qc"], ["metric", "value"], qc_rows)
    print(f"Wrote embeddings: {output_tsv}")
    print(f"Embedded subjects: {len(embedding_rows)}; failures: {len(failure_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
