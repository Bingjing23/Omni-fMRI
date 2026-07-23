#!/usr/bin/env python3
"""Extract one 768-dimensional Omni-fMRI CLS embedding per UKB subject.

This wrapper reuses the repository's existing `extract_feat.py` model,
checkpoint, tensor conversion, and token extraction functions. It adds
manifest-driven subject alignment, optional NIfTI preprocessing, segment-level
aggregation, TSV export, and failure/QC logs for HPC batch use.

Inputs:
  - Manifest TSV with columns `eid`, `subject_id`, and `image_path`.
  - Omni-fMRI checkpoint and config.

Outputs:
  - embeddings TSV: eid + subject_id + sample_id + image_path + emb_001 ... emb_768.
  - failure log TSV.
  - missing-subject log TSV.
  - progress TSV for tail-able per-subject status.
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
import time
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
    parser.add_argument(
        "--segment-batch-size",
        type=int,
        default=4,
        help="Number of NPZ segments to forward per GPU call. Default: 4.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for real extraction. Default: cuda:0")
    parser.add_argument("--eid-column", default="eid")
    parser.add_argument("--path-column", default="image_path")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N row limit for dry runs.")
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Optional 1-based shard index for PBS array extraction.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help="Total number of manifest shards. Requires --shard-index.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs and ignore resume state.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already present in the output TSV when resuming. Default: true.",
    )
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


def prepare_outputs(output_tsv: Path, work_dir: Path, force: bool, resume: bool) -> dict[str, Path]:
    if output_tsv.exists() and not force and not resume:
        raise FileExistsError(f"Output exists; pass --force or keep --resume enabled: {output_tsv}")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    return {
        "failures": output_tsv.with_suffix(".failures.tsv"),
        "missing": output_tsv.with_suffix(".missing_subjects.tsv"),
        "qc": output_tsv.with_suffix(".qc_summary.tsv"),
        "segment_qc": output_tsv.with_suffix(".segment_qc.tsv"),
        "progress": output_tsv.with_suffix(".progress.tsv"),
    }


def read_manifest(
    path: Path,
    eid_column: str,
    path_column: str,
    limit: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> pd.DataFrame:
    if (shard_index is None) != (num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together.")
    if shard_index is not None:
        if num_shards is None or num_shards < 1:
            raise ValueError("--num-shards must be >= 1.")
        if shard_index < 1 or shard_index > num_shards:
            raise ValueError("--shard-index must be between 1 and --num-shards.")

    manifest = pd.read_csv(path, sep="\t", dtype=str)
    missing = [column for column in (eid_column, path_column) if column not in manifest.columns]
    if missing:
        raise KeyError(f"Missing manifest columns: {missing}")
    keep_columns = [eid_column, path_column]
    optional_columns = ["subject_id", "sample_id", "batch", "input_kind"]
    keep_columns.extend(column for column in optional_columns if column in manifest.columns)
    manifest = manifest[keep_columns].rename(columns={eid_column: "eid", path_column: "image_path"})
    manifest["eid"] = manifest["eid"].fillna("").astype(str).str.strip()
    manifest["image_path"] = manifest["image_path"].fillna("").astype(str).str.strip()
    if "subject_id" not in manifest.columns:
        manifest["subject_id"] = manifest["eid"]
    if "sample_id" not in manifest.columns:
        manifest["sample_id"] = manifest["image_path"].map(lambda value: Path(value).name)
    if "batch" not in manifest.columns:
        manifest["batch"] = ""
    if "input_kind" not in manifest.columns:
        manifest["input_kind"] = ""
    manifest["subject_id"] = manifest["subject_id"].fillna("").astype(str).str.strip()
    manifest["sample_id"] = manifest["sample_id"].fillna("").astype(str).str.strip()
    manifest["batch"] = manifest["batch"].fillna("").astype(str).str.strip()
    manifest["input_kind"] = manifest["input_kind"].fillna("").astype(str).str.strip()
    manifest = manifest[manifest["eid"] != ""].copy()
    if shard_index is not None and num_shards is not None:
        positions = np.arange(len(manifest))
        manifest = manifest.iloc[(positions % num_shards) == (shard_index - 1)].copy()
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


def extract_npz_batch(
    npz_paths: list[Path],
    backbone,
    spatial_size: tuple[int, int, int],
    in_chans: int,
    args,
) -> list[np.ndarray]:
    import torch
    from extract_feat import array_to_model_tensor, extract_cls_tokens, load_npz_array

    samples = []
    for npz_path in npz_paths:
        array = load_npz_array(npz_path, args.npz_key)
        samples.append(
            array_to_model_tensor(
                array,
                in_chans=in_chans,
                spatial_size=spatial_size,
                layout=args.layout,
                start_frame=args.start_frame,
                pad_short=args.pad_short,
            )
        )
    sample_batch = torch.cat(samples, dim=0).to(device=torch.device(args.device), non_blocking=True)
    cls_tokens = extract_cls_tokens(backbone, sample_batch)
    return [np.asarray(token, dtype=np.float32) for token in cls_tokens]


def extract_subject_npzs(
    npz_files: list[Path],
    backbone,
    spatial_size: tuple[int, int, int],
    in_chans: int,
    args,
) -> list[np.ndarray]:
    if args.segment_batch_size < 1:
        raise ValueError("--segment-batch-size must be >= 1.")
    tokens: list[np.ndarray] = []
    for start in range(0, len(npz_files), args.segment_batch_size):
        tokens.extend(
            extract_npz_batch(
                npz_files[start : start + args.segment_batch_size],
                backbone,
                spatial_size,
                in_chans,
                args,
            )
        )
    return tokens


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


def completed_key(eid: object, subject_id: object, sample_id: object, image_path: object) -> tuple[str, str, str, str]:
    return (str(eid), str(subject_id), str(sample_id), str(image_path))


def read_existing_embeddings(path: Path) -> tuple[set[tuple[str, str, str, str]], int | None, int, int, int]:
    """Return completed row keys and finite-value counts from an existing TSV."""
    if not path.exists() or path.stat().st_size == 0:
        return set(), None, 0, 0, 0

    completed: set[tuple[str, str, str, str]] = set()
    finite_values = 0
    total_values = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return completed, None, 0, 0, 0
        embedding_columns = [name for name in reader.fieldnames if name.startswith("emb_")]
        width = len(embedding_columns)
        for row in reader:
            completed.add(
                completed_key(
                    row.get("eid", ""),
                    row.get("subject_id", ""),
                    row.get("sample_id", ""),
                    row.get("image_path", ""),
                )
            )
            for column in embedding_columns:
                total_values += 1
                try:
                    if math.isfinite(float(row[column])):
                        finite_values += 1
                except (TypeError, ValueError):
                    pass
    return completed, width, len(completed), finite_values, total_values


def open_stream_writer(path: Path, fieldnames: list[str], force: bool, resume: bool):
    append = path.exists() and path.stat().st_size > 0 and resume and not force
    mode = "a" if append else "w"
    handle = path.open(mode, encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    if not append:
        writer.writeheader()
        handle.flush()
    return handle, writer


def write_stream_row(handle, writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow(row)
    handle.flush()


def main() -> int:
    run_start = time.perf_counter()
    args = parse_args()
    output_tsv = Path(args.output_tsv)
    work_dir = Path(args.work_dir)
    aux_paths = prepare_outputs(output_tsv, work_dir, args.force, args.resume)
    manifest = read_manifest(
        Path(args.manifest),
        args.eid_column,
        args.path_column,
        args.limit,
        args.shard_index,
        args.num_shards,
    )

    if args.dry_run:
        missing_rows = [
            {
                "eid": row.eid,
                "subject_id": row.subject_id,
                "sample_id": row.sample_id,
                "image_path": row.image_path,
                "reason": "image_path missing or does not exist",
            }
            for row in manifest.itertuples(index=False)
            if not row.image_path or not Path(row.image_path).exists()
        ]
        print(f"DRY_RUN manifest_rows={len(manifest)} missing_paths={len(missing_rows)}")
        write_rows(aux_paths["missing"], ["eid", "subject_id", "sample_id", "image_path", "reason"], missing_rows)
        return 0

    completed, existing_width, completed_at_start, finite_values, total_embedding_values = read_existing_embeddings(
        output_tsv
    )

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
    expected_width = int(config.get("model", {}).get("embed_dim", 768))
    width = existing_width or expected_width
    if existing_width is not None and existing_width != expected_width:
        raise ValueError(f"Existing output width {existing_width} does not match checkpoint width {expected_width}.")

    metadata_columns = ["eid", "subject_id", "sample_id", "batch", "image_path"]
    embedding_fields = metadata_columns + emb_columns(width)
    failure_fields = ["eid", "subject_id", "sample_id", "image_path", "error"]
    missing_fields = ["eid", "subject_id", "sample_id", "image_path", "reason"]
    segment_fields = [
        "eid",
        "subject_id",
        "sample_id",
        "batch",
        "image_path",
        "segments",
        "embedding_width",
        "aggregation",
        "segment_batch_size",
        "preprocess_seconds",
        "inference_seconds",
        "subject_seconds",
        "seconds_per_segment",
    ]
    progress_fields = [
        "event_index",
        "timestamp",
        "elapsed_seconds",
        "manifest_rows",
        "rows_seen",
        "completed_at_start",
        "skipped_completed",
        "embedded_this_run",
        "failed_this_run",
        "missing_this_run",
        "completed_total",
        "total_segments_this_run",
        "total_preprocess_seconds",
        "total_inference_seconds",
        "last_eid",
        "last_subject_id",
        "last_sample_id",
        "last_status",
        "last_error",
    ]

    embedding_handle, embedding_writer = open_stream_writer(output_tsv, embedding_fields, args.force, args.resume)
    failure_handle, failure_writer = open_stream_writer(aux_paths["failures"], failure_fields, args.force, args.resume)
    missing_handle, missing_writer = open_stream_writer(aux_paths["missing"], missing_fields, args.force, args.resume)
    segment_handle, segment_writer = open_stream_writer(aux_paths["segment_qc"], segment_fields, args.force, args.resume)
    progress_handle, progress_writer = open_stream_writer(aux_paths["progress"], progress_fields, args.force, args.resume)

    skipped_completed = 0
    embedded_this_run = 0
    failed_this_run = 0
    missing_this_run = 0
    rows_seen = 0
    progress_events = 0
    total_segments = 0
    total_preprocess_seconds = 0.0
    total_inference_seconds = 0.0

    def write_progress(
        eid: str,
        subject_id: str,
        sample_id: str,
        status: str,
        error: str = "",
    ) -> None:
        nonlocal progress_events
        progress_events += 1
        write_stream_row(
            progress_handle,
            progress_writer,
            {
                "event_index": progress_events,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "elapsed_seconds": f"{time.perf_counter() - run_start:.6f}",
                "manifest_rows": len(manifest),
                "rows_seen": rows_seen,
                "completed_at_start": completed_at_start,
                "skipped_completed": skipped_completed,
                "embedded_this_run": embedded_this_run,
                "failed_this_run": failed_this_run,
                "missing_this_run": missing_this_run,
                "completed_total": completed_at_start + embedded_this_run,
                "total_segments_this_run": total_segments,
                "total_preprocess_seconds": f"{total_preprocess_seconds:.6f}",
                "total_inference_seconds": f"{total_inference_seconds:.6f}",
                "last_eid": eid,
                "last_subject_id": subject_id,
                "last_sample_id": sample_id,
                "last_status": status,
                "last_error": error,
            },
        )

    try:
        for row in manifest.itertuples(index=False):
            rows_seen += 1
            subject_start = time.perf_counter()
            eid = str(row.eid)
            subject_id = str(row.subject_id)
            sample_id = str(row.sample_id)
            batch = str(row.batch)
            image_path = Path(str(row.image_path)).expanduser()
            image_path_text = str(image_path)
            key = completed_key(eid, subject_id, sample_id, image_path_text)

            if key in completed:
                skipped_completed += 1
                write_progress(eid, subject_id, sample_id, "skipped_completed")
                continue

            if not image_path_text or not image_path.exists():
                missing_this_run += 1
                write_stream_row(
                    missing_handle,
                    missing_writer,
                    {
                        "eid": eid,
                        "subject_id": subject_id,
                        "sample_id": sample_id,
                        "image_path": image_path_text,
                        "reason": "image_path missing or does not exist",
                    },
                )
                write_progress(eid, subject_id, sample_id, "missing_path", "image_path missing or does not exist")
                continue

            try:
                preprocess_start = time.perf_counter()
                npz_files = npz_files_for_subject(image_path, args.input_kind, work_dir, args)
                preprocess_seconds = time.perf_counter() - preprocess_start
                inference_start = time.perf_counter()
                tokens = extract_subject_npzs(npz_files, backbone, spatial_size, in_chans, args)
                inference_seconds = time.perf_counter() - inference_start
                embedding = aggregate_segments(tokens, args.segment_aggregation)
                subject_seconds = time.perf_counter() - subject_start
                total_preprocess_seconds += preprocess_seconds
                total_inference_seconds += inference_seconds
                total_segments += len(tokens)
                if int(embedding.shape[0]) != width:
                    raise ValueError(f"Inconsistent embedding width: {embedding.shape[0]} != {width}")

                embedding_values = {name: float(value) for name, value in zip(emb_columns(width), embedding)}
                finite_values += int(np.isfinite(embedding).sum())
                total_embedding_values += int(embedding.size)
                output_row = {
                    "eid": eid,
                    "subject_id": subject_id,
                    "sample_id": sample_id,
                    "batch": batch,
                    "image_path": image_path_text,
                }
                output_row.update(embedding_values)
                write_stream_row(embedding_handle, embedding_writer, output_row)
                write_stream_row(
                    segment_handle,
                    segment_writer,
                    {
                        "eid": eid,
                        "subject_id": subject_id,
                        "sample_id": sample_id,
                        "batch": batch,
                        "image_path": image_path_text,
                        "segments": len(tokens),
                        "embedding_width": width,
                        "aggregation": args.segment_aggregation,
                        "segment_batch_size": args.segment_batch_size,
                        "preprocess_seconds": f"{preprocess_seconds:.6f}",
                        "inference_seconds": f"{inference_seconds:.6f}",
                        "subject_seconds": f"{subject_seconds:.6f}",
                        "seconds_per_segment": f"{inference_seconds / len(tokens):.6f}" if tokens else "nan",
                    },
                )
                embedded_this_run += 1
                completed.add(key)
                write_progress(eid, subject_id, sample_id, "embedded")
            except Exception as exc:
                failed_this_run += 1
                error = repr(exc)
                write_stream_row(
                    failure_handle,
                    failure_writer,
                    {
                        "eid": eid,
                        "subject_id": subject_id,
                        "sample_id": sample_id,
                        "image_path": image_path_text,
                        "error": error,
                    },
                )
                write_progress(eid, subject_id, sample_id, "failed", error)
    finally:
        for handle in [embedding_handle, failure_handle, missing_handle, segment_handle, progress_handle]:
            handle.close()

    total_run_seconds = time.perf_counter() - run_start
    embedded_subjects = completed_at_start + embedded_this_run
    qc_rows = [
        {"metric": "manifest_rows", "value": len(manifest)},
        {"metric": "shard_index", "value": args.shard_index or ""},
        {"metric": "num_shards", "value": args.num_shards or ""},
        {"metric": "completed_at_start", "value": completed_at_start},
        {"metric": "skipped_completed", "value": skipped_completed},
        {"metric": "embedded_subjects", "value": embedded_subjects},
        {"metric": "embedded_this_run", "value": embedded_this_run},
        {"metric": "failed_subjects", "value": failed_this_run},
        {"metric": "missing_paths", "value": missing_this_run},
        {"metric": "embedding_width", "value": width},
        {"metric": "finite_embedding_values", "value": finite_values},
        {"metric": "total_embedding_values", "value": total_embedding_values},
        {"metric": "finite_fraction", "value": finite_values / total_embedding_values if total_embedding_values else math.nan},
        {"metric": "total_segments", "value": total_segments},
        {"metric": "total_run_seconds", "value": f"{total_run_seconds:.6f}"},
        {"metric": "total_preprocess_seconds", "value": f"{total_preprocess_seconds:.6f}"},
        {"metric": "total_inference_seconds", "value": f"{total_inference_seconds:.6f}"},
        {"metric": "segment_batch_size", "value": args.segment_batch_size},
        {
            "metric": "mean_seconds_per_embedded_subject",
            "value": f"{total_run_seconds / embedded_this_run:.6f}" if embedded_this_run else "nan",
        },
        {
            "metric": "mean_inference_seconds_per_segment",
            "value": f"{total_inference_seconds / total_segments:.6f}" if total_segments else "nan",
        },
    ]
    write_rows(aux_paths["qc"], ["metric", "value"], qc_rows)
    print(f"Wrote embeddings: {output_tsv}")
    print(f"Embedded this run: {embedded_this_run}; skipped completed: {skipped_completed}; failures: {failed_this_run}")
    print(f"Progress log: {aux_paths['progress']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
