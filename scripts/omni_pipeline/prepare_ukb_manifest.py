#!/usr/bin/env python3
"""Prepare a UKB subject-to-imaging manifest for Omni-fMRI extraction.

Inputs:
  - Optional UKB subject list table containing pure `eid` values.
  - Either an existing imaging table with path columns, or a directory scan.

Outputs:
  - TSV manifest with columns:
    eid, subject_id, sample_id, image_path, batch, input_kind, source, status, note.
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
    parser.add_argument("--input-subject-id-column", default=None, help="Optional subject_id column in --input-table.")
    parser.add_argument("--input-path-column", default="image_path", help="Path column in --input-table.")
    parser.add_argument("--input-kind", choices=["nifti", "npz"], default="nifti", help="Input path type.")
    parser.add_argument("--scan-root", default=None, help="Directory to scan when --input-table is absent.")
    parser.add_argument("--scan-glob", default="**/*.npz", help="Glob under --scan-root. Default: **/*.npz")
    parser.add_argument(
        "--eid-regex",
        default=r"(?P<eid>\d{7})",
        help="Regex used to extract pure UKB eid from scanned paths. Default: (?P<eid>\\d{7})",
    )
    parser.add_argument(
        "--subject-id-regex",
        default=r"(?P<subject_id>\d+_20227_\d+_\d+)",
        help=(
            "Regex used to extract NeuroSTORM-compatible subject_id from paths. "
            "Default: (?P<subject_id>\\d+_20227_\\d+_\\d+)."
        ),
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


def normalize_subject_id(value: object, fallback_eid: str) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return text or fallback_eid


def strip_known_suffixes(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".npz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def infer_batch(path: Path) -> str:
    for part in path.parts:
        if part.startswith("mni_4d_20227_casebatch_"):
            return part.replace("mni_4d_20227_", "")
    return ""


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


def manifest_from_input_table(
    path: str,
    eid_column: str,
    subject_id_column: str | None,
    path_column: str,
    input_kind: str,
) -> pd.DataFrame:
    table = read_table(path)
    missing = [column for column in (eid_column, path_column) if column not in table.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")
    selected = [eid_column, path_column]
    if subject_id_column:
        if subject_id_column not in table.columns:
            raise KeyError(f"Missing subject_id column in {path}: {subject_id_column}")
        selected.append(subject_id_column)
    out = table[selected].rename(
        columns={eid_column: "eid", path_column: "image_path"}
    )
    out["eid"] = out["eid"].map(normalize_eid)
    out["image_path"] = out["image_path"].fillna("").astype(str).str.strip()
    if subject_id_column:
        out = out.rename(columns={subject_id_column: "subject_id"})
        out["subject_id"] = [
            normalize_subject_id(subject_id, eid)
            for subject_id, eid in zip(out["subject_id"], out["eid"])
        ]
    else:
        out["subject_id"] = out["eid"]
    out["sample_id"] = out["image_path"].map(lambda value: strip_known_suffixes(Path(value)))
    out["batch"] = out["image_path"].map(lambda value: infer_batch(Path(value)))
    out["input_kind"] = input_kind
    out["source"] = "input_table"
    return out


def manifest_from_scan(
    scan_root: str,
    scan_glob: str,
    eid_regex: str,
    subject_id_regex: str,
    input_kind: str,
) -> pd.DataFrame:
    root = Path(scan_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Scan root does not exist or is not a directory: {root}")
    eid_pattern = re.compile(eid_regex)
    subject_pattern = re.compile(subject_id_regex) if subject_id_regex else None
    rows: list[dict[str, str]] = []
    for path in sorted(root.glob(scan_glob)):
        if not path.is_file():
            continue
        path_text = str(path)
        match = eid_pattern.search(path_text)
        if not match:
            continue
        eid = match.groupdict().get("eid") if match.groupdict() else match.group(1)
        normalized_eid = normalize_eid(eid)
        subject_id = normalized_eid
        if subject_pattern is not None:
            subject_match = subject_pattern.search(path_text)
            if subject_match:
                subject_id = (
                    subject_match.groupdict().get("subject_id")
                    if subject_match.groupdict()
                    else subject_match.group(1)
                )
        rows.append(
            {
                "eid": normalized_eid,
                "subject_id": subject_id,
                "sample_id": strip_known_suffixes(path),
                "image_path": str(path.resolve()),
                "batch": infer_batch(path),
                "input_kind": input_kind,
                "source": "scan",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["eid", "subject_id", "sample_id", "image_path", "batch", "input_kind", "source"],
    )


def finalize_manifest(
    manifest: pd.DataFrame,
    requested: pd.DataFrame | None,
    allow_duplicates: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = manifest.copy()
    manifest["eid"] = manifest["eid"].map(normalize_eid)
    manifest["image_path"] = manifest["image_path"].fillna("").astype(str).str.strip()
    if "subject_id" not in manifest.columns:
        manifest["subject_id"] = manifest["eid"]
    if "sample_id" not in manifest.columns:
        manifest["sample_id"] = manifest["image_path"].map(lambda value: strip_known_suffixes(Path(value)))
    if "batch" not in manifest.columns:
        manifest["batch"] = manifest["image_path"].map(lambda value: infer_batch(Path(value)))
    if "input_kind" not in manifest.columns:
        manifest["input_kind"] = ""
    manifest = manifest[(manifest["eid"] != "") & (manifest["image_path"] != "")]
    manifest["path_exists"] = manifest["image_path"].map(lambda value: Path(value).exists())
    manifest["status"] = manifest["path_exists"].map(lambda exists: "ready" if exists else "missing_path")
    manifest["note"] = manifest["path_exists"].map(lambda exists: "" if exists else "image_path does not exist")

    if not allow_duplicates:
        manifest = manifest.sort_values(
            ["eid", "path_exists", "subject_id", "image_path"],
            ascending=[True, False, True, True],
        )
        manifest = manifest.drop_duplicates("eid", keep="first")

    missing = pd.DataFrame(columns=["eid", "status", "note"])
    if requested is not None:
        manifest = requested.merge(manifest, on="eid", how="left")
        missing_mask = manifest["image_path"].isna() | (manifest["image_path"].astype(str) == "")
        missing = manifest.loc[missing_mask, ["eid"]].copy()
        missing["status"] = "missing_subject"
        missing["note"] = "requested eid has no manifest image_path"
        manifest.loc[missing_mask, "image_path"] = ""
        manifest.loc[missing_mask, "subject_id"] = manifest.loc[missing_mask, "eid"]
        manifest.loc[missing_mask, "sample_id"] = ""
        manifest.loc[missing_mask, "batch"] = ""
        manifest.loc[missing_mask, "input_kind"] = ""
        manifest.loc[missing_mask, "source"] = "subject_list"
        manifest.loc[missing_mask, "path_exists"] = False
        manifest.loc[missing_mask, "status"] = "missing_subject"
        manifest.loc[missing_mask, "note"] = "requested eid has no manifest image_path"

    columns = ["eid", "subject_id", "sample_id", "image_path", "batch", "input_kind", "source", "status", "note"]
    return manifest[columns].sort_values(["eid", "subject_id", "image_path"]), missing


def main() -> int:
    args = parse_args()
    if bool(args.input_table) == bool(args.scan_root):
        raise ValueError("Provide exactly one of --input-table or --scan-root.")

    requested = load_requested_subjects(args.subject_list, args.subject_eid_column)
    if args.input_table:
        manifest = manifest_from_input_table(
            args.input_table,
            args.input_eid_column,
            args.input_subject_id_column,
            args.input_path_column,
            args.input_kind,
        )
    else:
        manifest = manifest_from_scan(
            args.scan_root,
            args.scan_glob,
            args.eid_regex,
            args.subject_id_regex,
            args.input_kind,
        )

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
