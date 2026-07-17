#!/usr/bin/env python3
"""Build and audit the header-ready UKB 20227 NIfTI manifest for Omni-fMRI.

Purpose:
  Scan only explicitly selected UKB 20227 rs-fMRI MNI 4D NIfTI batch
  directories, validate that each NIfTI still has the expected TR and dtype,
  and write a case-level manifest for Omni preprocessing/inference.

Inputs:
  UKB root directory containing mni_4d_20227_casebatch_* directories.

Outputs:
  TSV manifest with at least:
    eid, case_id, tag, nifti_path
  plus Omni-compatible aliases:
    subject_id, sample_id, image_path, batch, input_kind
  plus header audit fields:
    file_exists, header_status, pixdim4, dim4, dtype, raw_dtype, shape, note

Example:
  python scripts/omni_pipeline/prepare_header_ready_ukb_20227_manifest.py \
    --ukb-root /working/lab_puyag/bingjinZ/UKBB \
    --output /working/lab_puyag/bingjinZ/UKBB/omni_fmri/manifests/manifest_header_ready_all_cases.tsv \
    --summary-output /working/lab_puyag/bingjinZ/UKBB/omni_fmri/manifests/manifest_header_ready_all_cases.summary.tsv
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Iterable


try:
    import nibabel as nib
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    nib = None
    np = None
    NIFTI_IMPORT_ERROR = exc
else:
    NIFTI_IMPORT_ERROR = None


DEFAULT_BATCHES = [
    "mni_4d_20227_casebatch_0001_rest9800",
    "mni_4d_20227_casebatch_0002",
    "mni_4d_20227_casebatch_0003",
    "mni_4d_20227_casebatch_0009_missing_afterbench100",
]

BLOCKED_BATCHES = {
    "mni_4d_20227_casebatch_0004",
    "mni_4d_20227_casebatch_0005",
    "mni_4d_20227_casebatch_0006",
    "mni_4d_20227_casebatch_0007",
    "mni_4d_20227_casebatch_0008",
}

OUTPUT_COLUMNS = [
    "eid",
    "case_id",
    "tag",
    "nifti_path",
    "subject_id",
    "sample_id",
    "image_path",
    "batch",
    "input_kind",
    "file_exists",
    "header_status",
    "pixdim4",
    "dim4",
    "dtype",
    "raw_dtype",
    "shape",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare audited header-ready UKB 20227 Omni NIfTI manifest.")
    parser.add_argument("--ukb-root", required=True, help="Root containing UKB mni_4d_20227_casebatch_* directories.")
    parser.add_argument(
        "--batches",
        default=",".join(DEFAULT_BATCHES),
        help="Comma-separated batch directory names. Defaults to known header-ready batches only.",
    )
    parser.add_argument("--glob", default="*.nii.gz", help="NIfTI glob within each batch directory.")
    parser.add_argument(
        "--case-id-regex",
        default=r"(?P<case_id>[0-9]+_20227_[0-9]+_[0-9]+)",
        help="Regex extracting case_id from path.",
    )
    parser.add_argument("--eid-regex", default=r"(?P<eid>[0-9]{7})", help="Fallback regex extracting eid.")
    parser.add_argument("--expected-tr", type=float, default=0.735)
    parser.add_argument("--tr-tolerance", type=float, default=1e-4)
    parser.add_argument("--expected-dtype", default="float32")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=40,
        help="Minimum 4D timepoints required for Omni preprocessing. Default: 40.",
    )
    parser.add_argument("--output", required=True, help="Output manifest TSV.")
    parser.add_argument("--summary-output", default=None, help="Optional summary TSV.")
    parser.add_argument("--failed-output", default=None, help="Optional failed-header rows TSV.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def normalize_eid(value: str) -> str:
    return value.lstrip("0") or "0" if value.isdigit() else value


def extract_ids(path: Path, case_pattern: re.Pattern[str], eid_pattern: re.Pattern[str]) -> tuple[str, str]:
    text = str(path)
    case_match = case_pattern.search(text)
    if case_match:
        case_id = case_match.groupdict().get("case_id") if case_match.groupdict() else case_match.group(1)
        eid = normalize_eid(case_id.split("_", 1)[0])
        return eid, case_id
    eid_match = eid_pattern.search(text)
    if not eid_match:
        return "", strip_nifti_suffix(path)
    eid = eid_match.groupdict().get("eid") if eid_match.groupdict() else eid_match.group(1)
    eid = normalize_eid(eid)
    return eid, strip_nifti_suffix(path)


def audit_header(
    path: Path,
    expected_tr: float,
    tr_tolerance: float,
    expected_dtype: str,
    min_frames: int,
) -> dict[str, str]:
    if NIFTI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "NIfTI header audit requires nibabel and numpy in the active Python environment. "
            f"Import failed: {NIFTI_IMPORT_ERROR!r}"
        )

    if not path.exists():
        return {
            "file_exists": "0",
            "header_status": "missing_path",
            "pixdim4": "",
            "dim4": "",
            "dtype": "",
            "raw_dtype": "",
            "shape": "",
            "note": "nifti_path does not exist",
        }

    try:
        assert nib is not None
        assert np is not None
        img = nib.load(str(path))
        zooms = img.header.get_zooms()
        tr = float(zooms[3]) if len(zooms) > 3 else math.nan
        dim4 = int(img.shape[3]) if len(img.shape) > 3 else 1
        raw_dtype = img.header.get_data_dtype()
        dtype = np.dtype(raw_dtype).name
        shape = "x".join(str(value) for value in img.shape)
    except Exception as exc:  # pragma: no cover - depends on real NIfTI files.
        return {
            "file_exists": "1",
            "header_status": "header_error",
            "pixdim4": "",
            "dim4": "",
            "dtype": "",
            "raw_dtype": "",
            "shape": "",
            "note": f"{type(exc).__name__}: {exc}",
        }

    problems: list[str] = []
    if not math.isfinite(tr) or abs(tr - expected_tr) > tr_tolerance:
        problems.append(f"TR {tr:.12g} != expected {expected_tr:.12g}")
    if dim4 < min_frames:
        problems.append(f"dim4 {dim4} < min_frames {min_frames}")
    if dtype != expected_dtype:
        problems.append(f"dtype {dtype} != expected {expected_dtype} (raw={raw_dtype!s})")

    return {
        "file_exists": "1",
        "header_status": "ready" if not problems else "header_mismatch",
        "pixdim4": f"{tr:.12g}" if math.isfinite(tr) else "NA",
        "dim4": str(dim4),
        "dtype": dtype,
        "raw_dtype": str(raw_dtype),
        "shape": shape,
        "note": "; ".join(problems),
    }


def iter_nifti_paths(ukb_root: Path, batches: Iterable[str], pattern: str) -> Iterable[tuple[str, Path]]:
    for batch in batches:
        if batch in BLOCKED_BATCHES:
            raise ValueError(f"Refusing to scan blocked header-repair batch: {batch}")
        batch_dir = ukb_root / batch
        if not batch_dir.is_dir():
            raise NotADirectoryError(f"Batch directory not found: {batch_dir}")
        tag = batch.replace("mni_4d_20227_", "")
        for path in sorted(batch_dir.glob(pattern)):
            if path.is_file():
                yield tag, path.resolve()


def ensure_writable(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    summary_output = Path(args.summary_output) if args.summary_output else output.with_suffix(".summary.tsv")
    failed_output = Path(args.failed_output) if args.failed_output else output.with_suffix(".failed_header.tsv")
    ensure_writable([output, summary_output, failed_output], args.force)

    ukb_root = Path(args.ukb_root)
    batches = split_csv(args.batches)
    case_pattern = re.compile(args.case_id_regex)
    eid_pattern = re.compile(args.eid_regex)

    rows: list[dict[str, object]] = []
    for tag, path in iter_nifti_paths(ukb_root, batches, args.glob):
        eid, case_id = extract_ids(path, case_pattern, eid_pattern)
        audit = audit_header(
            path,
            args.expected_tr,
            args.tr_tolerance,
            args.expected_dtype,
            args.min_frames,
        )
        rows.append(
            {
                "eid": eid,
                "case_id": case_id,
                "tag": tag,
                "nifti_path": str(path),
                "subject_id": case_id,
                "sample_id": case_id,
                "image_path": str(path),
                "batch": tag,
                "input_kind": "nifti",
                **audit,
            }
        )

    rows.sort(key=lambda row: (str(row["tag"]), str(row["case_id"]), str(row["nifti_path"])))
    failed_rows = [row for row in rows if row["header_status"] != "ready"]
    write_tsv(output, OUTPUT_COLUMNS, rows)
    write_tsv(failed_output, OUTPUT_COLUMNS, failed_rows)

    by_tag: dict[str, dict[str, object]] = {}
    for row in rows:
        tag = str(row["tag"])
        by_tag.setdefault(tag, {"tag": tag, "rows": 0, "ready": 0, "failed": 0})
        by_tag[tag]["rows"] = int(by_tag[tag]["rows"]) + 1
        if row["header_status"] == "ready":
            by_tag[tag]["ready"] = int(by_tag[tag]["ready"]) + 1
        else:
            by_tag[tag]["failed"] = int(by_tag[tag]["failed"]) + 1

    summary_rows = sorted(by_tag.values(), key=lambda row: str(row["tag"]))
    summary_rows.append(
        {
            "tag": "__TOTAL__",
            "rows": len(rows),
            "ready": len(rows) - len(failed_rows),
            "failed": len(failed_rows),
        }
    )
    write_tsv(summary_output, ["tag", "rows", "ready", "failed"], summary_rows)

    print(f"Wrote manifest: {output}")
    print(f"Wrote summary: {summary_output}")
    print(f"Wrote failed-header rows: {failed_output}")
    print(f"Rows: {len(rows)}; ready: {len(rows) - len(failed_rows)}; failed: {len(failed_rows)}")
    return 1 if failed_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
