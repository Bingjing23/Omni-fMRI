#!/usr/bin/env python3
"""Filter case-level Omni-fMRI embeddings by a NeuroSTORM subject_id list.

Inputs:
  - Case-level Omni embedding TSV with columns `subject_id`, `eid`, and emb_*.
  - One-column subject_id keep list, e.g. NeuroSTORM
    `neurostorm_mae_5ds_7batch_one_instance_per_eid.subjects.txt`.

Outputs:
  - Filtered embedding TSV preserving the keep-list order when possible.
  - Summary TSV with requested, matched, unmatched, and duplicate counts.
  - Optional unmatched subject_id list.

This script is intended to make Omni GWAS inputs use the exact same first-case
selection as the NeuroSTORM MAE-5ds GWAS-ready embedding table.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Omni embeddings by subject_id keep list.")
    parser.add_argument("--embeddings-tsv", required=True, help="Case-level Omni embedding TSV.")
    parser.add_argument("--keep-subject-list", required=True, help="One-column subject_id list.")
    parser.add_argument("--output-tsv", required=True, help="Filtered output TSV.")
    parser.add_argument("--summary-tsv", required=True, help="Filter summary TSV.")
    parser.add_argument("--unmatched-output", default=None, help="Optional unmatched subject_id TSV.")
    parser.add_argument("--subject-id-column", default="subject_id")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def read_keep_list(path: Path) -> list[str]:
    values: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            value = raw_line.strip()
            if not value:
                continue
            if line_number == 1 and value == "subject_id":
                continue
            values.append(value)
    if not values:
        raise ValueError(f"No subject IDs found in keep list: {path}")
    return values


def ensure_writable(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    output = Path(args.output_tsv)
    summary = Path(args.summary_tsv)
    unmatched_output = Path(args.unmatched_output) if args.unmatched_output else None
    ensure_writable([path for path in [output, summary, unmatched_output] if path is not None], args.force)

    keep_subjects = read_keep_list(Path(args.keep_subject_list))
    keep_order = pd.DataFrame(
        {
            args.subject_id_column: keep_subjects,
            "_keep_order": range(len(keep_subjects)),
        }
    )
    embeddings = pd.read_csv(args.embeddings_tsv, sep="\t", dtype=str)
    if args.subject_id_column not in embeddings.columns:
        raise KeyError(f"Missing subject_id column in embeddings TSV: {args.subject_id_column}")
    embeddings[args.subject_id_column] = embeddings[args.subject_id_column].fillna("").astype(str).str.strip()

    duplicate_embedding_subjects = int(embeddings.duplicated(args.subject_id_column).sum())
    matched = keep_order.merge(embeddings, on=args.subject_id_column, how="left", indicator=True)
    unmatched = matched.loc[matched["_merge"] == "left_only", [args.subject_id_column]].copy()
    filtered = matched.loc[matched["_merge"] == "both"].drop(columns=["_merge"]).sort_values("_keep_order")
    filtered = filtered.drop(columns=["_keep_order"])
    filtered.to_csv(output, sep="\t", index=False)

    rows = [
        {"metric": "keep_subjects_requested", "value": len(keep_subjects)},
        {"metric": "embedding_rows_input", "value": len(embeddings)},
        {"metric": "duplicate_subject_id_rows_in_embeddings", "value": duplicate_embedding_subjects},
        {"metric": "matched_subjects", "value": len(filtered)},
        {"metric": "unmatched_subjects", "value": len(unmatched)},
        {"metric": "output_tsv", "value": str(output)},
    ]
    pd.DataFrame(rows).to_csv(summary, sep="\t", index=False)
    if unmatched_output is not None:
        unmatched.to_csv(unmatched_output, sep="\t", index=False)

    print(f"Wrote filtered embeddings: {output}")
    print(f"Matched subjects: {len(filtered)}; unmatched subjects: {len(unmatched)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
