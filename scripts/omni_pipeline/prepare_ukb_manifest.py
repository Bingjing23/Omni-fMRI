#!/usr/bin/env python3
"""Prepare a UKB subject-to-imaging manifest for Omni-fMRI extraction.

Inputs:
  - Optional UKB subject list table containing pure `eid` values.
  - Either an existing imaging table with path columns, or a directory scan.

Outputs:
  - TSV manifest with columns: eid, image_path, source, status, note.
  - Optional missing-subject log for requested EIDs without an image path.

Example:
  python scripts/omni_pipeline/prepare_ukb_manifest.py \
    --subject-list /path/ukb_subjects.tsv \
    --scan-root /path/ukb_fmri_npz \
    --scan-glob '**/*.npz' \
    --output manifests/ukb_omni_manifest.tsv \
    --missing-output manifests/ukb_omni_missing.tsv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a UKB Omni-fMRI imaging manifest.")
    parser.add_argument("--subject-list", default=None, help="Optional TSV/CSV with an eid column.")
    parser.add_argument("--subject-eid-column", default="eid", help="EID column in --subject-list.")
    parser.add_argument("--input-table", default=None, help="Optional TSV/CSV with imaging paths.")
    parser.add_argument("--input-eid-column", default="eid", help="EID column in --input-table.")
    parser.add_argument("--input-path-column", default="image_path", help="Path column in --input-table.")
    parser.add_argument("--scan-root", default=None, help="Directory to scan when --input-table is absent.")
    parser.add_argument("--scan-glob", default="**/*.npz", help="Glob under --scan-root. Default: **/*.npz")
    parser.add_argument(
        "--eid-regex",
        default=r"(?P<eid>\d{7})",
        help="Regex used to extract pure UKB eid from scanned paths. Default: (?P<eid>\\d{7})",
    )
    parser.add_argument("--output", required=True, help="Output manifest TSV.")
    parser.add_argument("--missing-output", default=None, help="Optional missing-subject TSV.")
    parser.add_argument("--allow-duplicates", action="store_true", help="Keep duplicate EID rows.")
    return parser.parse_args()


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str)
    return pd.read_csv(path, sep="\t", dtype=str)


def normalize_eid(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit():
        return text
    return text.lstrip("0") or "0"


def load_requested_subjects(path: str | None, eid_column: str) -> pd.DataFrame | None:
    if not path:
        return None
    subjects = read_table(path)
    if eid_column not in subjects.columns:
        raise KeyError(f"Missing subject EID column '{eid_column}' in {path}")
    out = subjects[[eid_column]].rename(columns={eid_column: "eid"}).copy()
    out["eid"] = out["eid"].map(normalize_eid)
    out = out[out["eid"] != ""].drop_duplicates("eid")
    return out


def manifest_from_input_table(path: str, eid_column: str, path_column: str) -> pd.DataFrame:
    table = read_table(path)
    missing = [column for column in (eid_column, path_column) if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    out = table[[eid_column, path_column]].rename(
        columns={eid_column: "eid", path_column: "image_path"}
    )
    out["eid"] = out["eid"].map(normalize_eid)
    out["image_path"] = out["image_path"].fillna("").astype(str).str.strip()
    out["source"] = "input_table"
    return out


def manifest_from_scan(scan_root: str, scan_glob: str, eid_regex: str) -> pd.DataFrame:
    root = Path(scan_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Scan root does not exist or is not a directory: {root}")
    pattern = re.compile(eid_regex)
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(scan_glob)):
        if not path.is_file():
            continue
        match = pattern.search(str(path))
        if not match:
            continue
        eid = match.groupdict().get("eid") if match.groupdict() else match.group(1)
        rows.append(
            {
                "eid": normalize_eid(eid),
                "image_path": str(path.resolve()),
                "source": "scan",
            }
        )
    return pd.DataFrame(rows, columns=["eid", "image_path", "source"])


def finalize_manifest(
    manifest: pd.DataFrame,
    requested: pd.DataFrame | None,
    allow_duplicates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = manifest.copy()
    manifest["eid"] = manifest["eid"].map(normalize_eid)
    manifest["image_path"] = manifest["image_path"].fillna("").astype(str).str.strip()
    manifest = manifest[(manifest["eid"] != "") & (manifest["image_path"] != "")]
    manifest["path_exists"] = manifest["image_path"].map(lambda value: Path(value).exists())
    manifest["status"] = manifest["path_exists"].map(lambda exists: "ready" if exists else "missing_path")
    manifest["note"] = manifest["path_exists"].map(lambda exists: "" if exists else "image_path does not exist")

    if not allow_duplicates:
        manifest = manifest.sort_values(["eid", "path_exists", "image_path"], ascending=[True, False, True])
        manifest = manifest.drop_duplicates("eid", keep="first")

    missing = pd.DataFrame(columns=["eid", "status", "note"])
    if requested is not None:
        manifest = requested.merge(manifest, on="eid", how="left")
        missing_mask = manifest["image_path"].isna() | (manifest["image_path"].astype(str) == "")
        missing = manifest.loc[missing_mask, ["eid"]].copy()
        missing["status"] = "missing_subject"
        missing["note"] = "requested eid has no manifest image_path"
        manifest.loc[missing_mask, "image_path"] = ""
        manifest.loc[missing_mask, "source"] = "subject_list"
        manifest.loc[missing_mask, "path_exists"] = False
        manifest.loc[missing_mask, "status"] = "missing_subject"
        manifest.loc[missing_mask, "note"] = "requested eid has no manifest image_path"

    columns = ["eid", "image_path", "source", "status", "note"]
    return manifest[columns].sort_values("eid"), missing


def main() -> int:
    args = parse_args()
    if bool(args.input_table) == bool(args.scan_root):
        raise ValueError("Provide exactly one of --input-table or --scan-root.")

    requested = load_requested_subjects(args.subject_list, args.subject_eid_column)
    if args.input_table:
        manifest = manifest_from_input_table(
            args.input_table,
            args.input_eid_column,
            args.input_path_column,
        )
    else:
        manifest = manifest_from_scan(args.scan_root, args.scan_glob, args.eid_regex)

    final_manifest, missing = finalize_manifest(manifest, requested, args.allow_duplicates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final_manifest.to_csv(output, sep="\t", index=False)

    if args.missing_output:
        missing_output = Path(args.missing_output)
        missing_output.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(missing_output, sep="\t", index=False)

    ready = int((final_manifest["status"] == "ready").sum())
    print(f"Wrote manifest: {output}")
    print(f"Rows: {len(final_manifest)}; ready: {ready}; missing: {len(final_manifest) - ready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
