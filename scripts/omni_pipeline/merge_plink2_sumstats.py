#!/usr/bin/env python3
"""Merge chromosome-level PLINK2 GWAS outputs into per-embedding sumstats.

Inputs:
  - PLINK2 output tree produced by submit_plink2_gwas.pbs.

Outputs:
  - One gzip-compressed genome-wide TSV per embedding.
  - Merge QC summary with missing chromosomes and row counts.

Example:
  python scripts/omni_pipeline/merge_plink2_sumstats.py \
    --plink-dir /path/to/outputs/omni/plink2 \
    --output-dir /path/to/outputs/omni/sumstats_merged \
    --embedding-count 768
"""

from __future__ import annotations

import argparse
import csv
import gzip
import glob
from pathlib import Path
from typing import Iterator


REQUIRED_COLUMNS = ["#CHROM", "POS", "ID", "A1", "P", "BETA", "SE", "OBS_CT"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge PLINK2 chromosome GWAS outputs.")
    parser.add_argument("--plink-dir", required=True, help="Root directory containing emb_*/ chromosome outputs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged *.sumstats.tsv.gz files.")
    parser.add_argument("--embedding-count", type=int, default=768)
    parser.add_argument("--trait-prefix", default="emb")
    parser.add_argument(
        "--input-template",
        default="{plink_dir}/{trait_id}/{trait_id}.chr{chr}*.glm.linear*",
        help="Glob template with {plink_dir}, {trait_id}, and {chr}.",
    )
    parser.add_argument("--test-col", default="TEST")
    parser.add_argument("--test-value", default="ADD", help="Keep this TEST value when TEST exists. Empty keeps all.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing merged files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned merges without writing sumstats.")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def find_one(pattern: str) -> Path | None:
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    if len(hits) > 1:
        exact = [path for path in hits if path.endswith(".glm.linear") or path.endswith(".glm.linear.gz")]
        if len(exact) == 1:
            return Path(exact[0])
    return Path(hits[0])


def iter_rows(path: Path, test_col: str, test_value: str) -> tuple[list[str], Iterator[dict[str, str]]]:
    handle = open_text(path)
    reader = csv.DictReader(handle, delimiter="\t")
    if reader.fieldnames is None:
        handle.close()
        raise ValueError(f"Missing header in {path}")
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        handle.close()
        raise KeyError(f"Missing required columns in {path}: {missing}")

    def generator() -> Iterator[dict[str, str]]:
        try:
            for row in reader:
                if test_value and test_col in row and row.get(test_col) != test_value:
                    continue
                yield row
        finally:
            handle.close()

    return list(reader.fieldnames), generator()


def merge_trait(args: argparse.Namespace, trait_id: str, outdir: Path) -> dict[str, object]:
    out_path = outdir / f"{trait_id}.sumstats.tsv.gz"
    if out_path.exists() and not args.force and not args.dry_run:
        raise FileExistsError(f"Output exists; pass --force to overwrite: {out_path}")

    chr_paths: dict[int, Path] = {}
    missing_chr: list[int] = []
    for chrom in range(1, 23):
        pattern = args.input_template.format(
            plink_dir=str(Path(args.plink_dir)),
            trait_id=trait_id,
            chr=chrom,
        )
        path = find_one(pattern)
        if path is None:
            missing_chr.append(chrom)
        else:
            chr_paths[chrom] = path

    if args.dry_run:
        return {
            "embedding_id": trait_id,
            "output": str(out_path),
            "chromosomes_found": len(chr_paths),
            "chromosomes_missing": ",".join(map(str, missing_chr)),
            "rows_written": 0,
            "status": "dry_run",
        }

    fieldnames: list[str] | None = None
    rows_written = 0
    with gzip.open(out_path, "wt", encoding="utf-8", newline="") as out_handle:
        writer: csv.DictWriter | None = None
        for chrom in range(1, 23):
            path = chr_paths.get(chrom)
            if path is None:
                continue
            current_fields, rows = iter_rows(path, args.test_col, args.test_value)
            if fieldnames is None:
                fieldnames = current_fields
                writer = csv.DictWriter(out_handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
                writer.writeheader()
            elif current_fields != fieldnames:
                raise ValueError(f"Column mismatch for {trait_id} chr{chrom}: {path}")
            assert writer is not None
            for row in rows:
                writer.writerow(row)
                rows_written += 1

    status = "complete" if not missing_chr else "missing_chromosomes"
    return {
        "embedding_id": trait_id,
        "output": str(out_path),
        "chromosomes_found": len(chr_paths),
        "chromosomes_missing": ",".join(map(str, missing_chr)),
        "rows_written": rows_written,
        "status": status,
    }


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(1, args.embedding_count + 1):
        trait_id = f"{args.trait_prefix}_{index:03d}"
        rows.append(merge_trait(args, trait_id, outdir))

    summary = outdir / "merge_plink2_sumstats_summary.tsv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["embedding_id", "output", "chromosomes_found", "chromosomes_missing", "rows_written", "status"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote merge summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
