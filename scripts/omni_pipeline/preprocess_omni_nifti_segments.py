#!/usr/bin/env python3
"""Preprocess Omni-fMRI NIfTI inputs into per-case NPZ segment directories.

This CPU-only stage separates slow NIfTI I/O, normalization, segmentation, and
NPZ writing from GPU model inference. The output manifest points each case to a
directory of NPZ segments and can be consumed by `extract_omni_embeddings.py`
with `--input-kind npz`.
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
    parser = argparse.ArgumentParser(description="Preprocess Omni NIfTI cases into NPZ segment directories.")
    parser.add_argument("--manifest", required=True, help="Input manifest with NIfTI paths.")
    parser.add_argument("--output-root", required=True, help="Root directory for per-case NPZ segment directories.")
    parser.add_argument("--output-manifest", required=True, help="Output NPZ-dir manifest TSV.")
    parser.add_argument("--qc-output", default=None, help="Optional preprocessing QC summary TSV.")
    parser.add_argument("--failures-output", default=None, help="Optional failures TSV.")
    parser.add_argument("--eid-column", default="eid")
    parser.add_argument("--case-id-column", default="case_id")
    parser.add_argument("--tag-column", default="tag")
    parser.add_argument("--path-column", default="nifti_path")
    parser.add_argument("--segment-length", type=int, default=40)
    parser.add_argument("--target-shape", nargs=3, type=int, default=[96, 96, 96])
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N row limit after sharding.")
    parser.add_argument("--shard-index", type=int, default=None, help="Optional 1-based PBS shard index.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total shard count. Requires --shard-index.")
    parser.add_argument("--force", action="store_true", help="Overwrite output TSVs.")
    return parser.parse_args()


def output_paths(output_manifest: Path, qc_output: str | None, failures_output: str | None) -> dict[str, Path]:
    return {
        "manifest": output_manifest,
        "qc": Path(qc_output) if qc_output else output_manifest.with_suffix(".preprocess_qc.tsv"),
        "failures": Path(failures_output) if failures_output else output_manifest.with_suffix(".preprocess_failures.tsv"),
    }


def ensure_writable(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def read_manifest(args: argparse.Namespace) -> pd.DataFrame:
    if (args.shard_index is None) != (args.num_shards is None):
        raise ValueError("--shard-index and --num-shards must be provided together.")
    if args.shard_index is not None:
        if args.num_shards < 1:
            raise ValueError("--num-shards must be >= 1.")
        if args.shard_index < 1 or args.shard_index > args.num_shards:
            raise ValueError("--shard-index must be between 1 and --num-shards.")

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    required = [args.eid_column, args.case_id_column, args.path_column]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise KeyError(f"Missing manifest columns: {missing}")

    keep = [args.eid_column, args.case_id_column, args.path_column]
    if args.tag_column in manifest.columns:
        keep.append(args.tag_column)
    manifest = manifest[keep].rename(
        columns={
            args.eid_column: "eid",
            args.case_id_column: "case_id",
            args.path_column: "nifti_path",
            args.tag_column: "tag",
        }
    )
    if "tag" not in manifest.columns:
        manifest["tag"] = ""
    for column in ["eid", "case_id", "tag", "nifti_path"]:
        manifest[column] = manifest[column].fillna("").astype(str).str.strip()
    manifest = manifest[(manifest["eid"] != "") & (manifest["case_id"] != "") & (manifest["nifti_path"] != "")].copy()

    if args.shard_index is not None and args.num_shards is not None:
        positions = np.arange(len(manifest))
        manifest = manifest.iloc[(positions % args.num_shards) == (args.shard_index - 1)].copy()
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    return manifest


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    run_start = time.perf_counter()
    args = parse_args()
    from data_preparation.preprocessing import process_single_subject

    paths = output_paths(Path(args.output_manifest), args.qc_output, args.failures_output)
    ensure_writable(list(paths.values()), args.force)

    manifest = read_manifest(args)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    total_segments = 0
    total_preprocess_seconds = 0.0

    for row in manifest.itertuples(index=False):
        nifti_path = Path(row.nifti_path)
        case_dir = output_root / str(row.tag) / str(row.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        case_start = time.perf_counter()
        try:
            if not nifti_path.is_file():
                raise FileNotFoundError(f"NIfTI path does not exist: {nifti_path}")
            ok = process_single_subject(
                nifti_path,
                case_dir,
                target_shape=tuple(args.target_shape),
                segment_length=args.segment_length,
            )
            if not ok:
                raise RuntimeError(f"Omni preprocessing returned false: {nifti_path}")
            npz_files = sorted(case_dir.glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(f"No NPZ segments produced: {case_dir}")
            seconds = time.perf_counter() - case_start
            total_preprocess_seconds += seconds
            total_segments += len(npz_files)
            manifest_rows.append(
                {
                    "eid": row.eid,
                    "subject_id": row.case_id,
                    "sample_id": row.case_id,
                    "case_id": row.case_id,
                    "tag": row.tag,
                    "batch": row.tag,
                    "nifti_path": str(nifti_path),
                    "image_path": str(case_dir),
                    "input_kind": "npz",
                    "segments": len(npz_files),
                    "preprocess_seconds": f"{seconds:.6f}",
                    "seconds_per_segment": f"{seconds / len(npz_files):.6f}",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "eid": row.eid,
                    "case_id": row.case_id,
                    "tag": row.tag,
                    "nifti_path": str(nifti_path),
                    "output_dir": str(case_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    manifest_fields = [
        "eid",
        "subject_id",
        "sample_id",
        "case_id",
        "tag",
        "batch",
        "nifti_path",
        "image_path",
        "input_kind",
        "segments",
        "preprocess_seconds",
        "seconds_per_segment",
    ]
    failure_fields = ["eid", "case_id", "tag", "nifti_path", "output_dir", "error"]
    write_rows(paths["manifest"], manifest_fields, manifest_rows)
    write_rows(paths["failures"], failure_fields, failure_rows)

    total_run_seconds = time.perf_counter() - run_start
    qc_rows = [
        {"metric": "manifest_rows", "value": len(manifest)},
        {"metric": "preprocessed_cases", "value": len(manifest_rows)},
        {"metric": "failed_cases", "value": len(failure_rows)},
        {"metric": "total_segments", "value": total_segments},
        {"metric": "shard_index", "value": args.shard_index or ""},
        {"metric": "num_shards", "value": args.num_shards or ""},
        {"metric": "total_run_seconds", "value": f"{total_run_seconds:.6f}"},
        {"metric": "total_preprocess_seconds", "value": f"{total_preprocess_seconds:.6f}"},
        {
            "metric": "mean_seconds_per_case",
            "value": f"{total_preprocess_seconds / len(manifest_rows):.6f}" if manifest_rows else "nan",
        },
        {
            "metric": "mean_seconds_per_segment",
            "value": f"{total_preprocess_seconds / total_segments:.6f}" if total_segments else "nan",
        },
    ]
    write_rows(paths["qc"], ["metric", "value"], qc_rows)

    print(f"Wrote NPZ manifest: {paths['manifest']}")
    print(f"Wrote QC: {paths['qc']}")
    print(f"Wrote failures: {paths['failures']}")
    print(f"Preprocessed cases: {len(manifest_rows)}; failures: {len(failure_rows)}")
    return 1 if failure_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
