#!/usr/bin/env python3
"""Merge homogeneous TSV shards while keeping one header.

This is intended for Omni embedding shard outputs, failure logs, and QC tables
created by PBS arrays. It validates that all shard headers match before writing
the merged file.
"""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge same-schema TSV shard files.")
    parser.add_argument("--shards-glob", required=True, help="Glob for input shard TSV files.")
    parser.add_argument("--output", required=True, help="Merged output TSV.")
    parser.add_argument("--summary", default=None, help="Optional merge summary TSV.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def ensure_writable(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(f"Output exists; pass --force to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.shards_glob))]
    if not paths:
        raise FileNotFoundError(f"No shards matched --shards-glob {args.shards_glob!r}")

    output = Path(args.output)
    summary = Path(args.summary) if args.summary else output.with_suffix(".merge_summary.tsv")
    ensure_writable([output, summary], args.force)

    expected_header: list[str] | None = None
    summary_rows: list[dict[str, object]] = []
    total_rows = 0
    with output.open("w", encoding="utf-8", newline="") as out_handle:
        writer: csv.writer | None = None
        for path in paths:
            with path.open("r", encoding="utf-8", newline="") as in_handle:
                reader = csv.reader(in_handle, delimiter="\t")
                try:
                    header = next(reader)
                except StopIteration:
                    summary_rows.append({"shard": str(path), "rows": 0, "status": "empty"})
                    continue
                if expected_header is None:
                    expected_header = header
                    writer = csv.writer(out_handle, delimiter="\t", lineterminator="\n")
                    writer.writerow(header)
                elif header != expected_header:
                    raise ValueError(f"Header mismatch in shard: {path}")
                assert writer is not None
                rows = 0
                for row in reader:
                    writer.writerow(row)
                    rows += 1
                total_rows += rows
                summary_rows.append({"shard": str(path), "rows": rows, "status": "merged"})

    summary_rows.append({"shard": "__TOTAL__", "rows": total_rows, "status": "merged"})
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shard", "rows", "status"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {output}")
    print(f"Wrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
