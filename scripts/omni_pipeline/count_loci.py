#!/usr/bin/env python3
"""Count approximate independent loci across Omni-fMRI embedding GWAS results.

The script reads PLINK2-like summary statistics for many embedding traits,
counts variants passing P < 5e-8 and P < 5e-8 / number_of_embeddings, merges
nearby significant variants into approximate regions, and writes per-embedding
and model-level region summaries.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


AUTOSOMES = {str(index) for index in range(1, 23)}
MISSING = {"", "NA", "NaN", "nan", ".", "None", "null"}


@dataclass(frozen=True)
class Hit:
    embedding_id: str
    chrom: str
    pos: int
    variant_id: str
    p: float
    beta: str
    se: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count Omni-fMRI GWAS loci.")
    parser.add_argument("--sumstats-glob", required=True, help="Glob for per-embedding GWAS sumstats.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--embedding-regex", default=r"(emb_\d{3})", help="Regex extracting embedding_id from path.")
    parser.add_argument("--p-threshold", type=float, default=5e-8)
    parser.add_argument("--strict-threshold", type=float, default=None, help="Default: p-threshold / embedding-count.")
    parser.add_argument("--embedding-count", type=int, default=768)
    parser.add_argument("--window-bp", type=int, default=1_000_000, help="Merge hits within this distance.")
    parser.add_argument("--chrom-col", default="#CHROM")
    parser.add_argument("--pos-col", default="POS")
    parser.add_argument("--p-col", default="P")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--beta-col", default="BETA")
    parser.add_argument("--se-col", default="SE")
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def norm(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text in MISSING else text


def parse_float(value: object) -> float | None:
    text = norm(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_chrom(value: object) -> str:
    text = norm(value)
    if text.lower().startswith("chr"):
        text = text[3:]
    return text


def extract_embedding_id(path: Path, regex: re.Pattern[str]) -> str:
    match = regex.search(str(path))
    if not match:
        raise ValueError(f"Could not extract embedding_id from path: {path}")
    return match.group(1)


def iter_hits(path: Path, embedding_id: str, threshold: float, args: argparse.Namespace) -> Iterator[Hit]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or args.p_col not in reader.fieldnames:
            raise ValueError(f"Missing P column {args.p_col!r} in {path}")
        for row in reader:
            chrom = parse_chrom(row.get(args.chrom_col))
            if chrom not in AUTOSOMES:
                continue
            p = parse_float(row.get(args.p_col))
            if p is None or p >= threshold:
                continue
            try:
                pos = int(float(norm(row.get(args.pos_col))))
            except ValueError:
                continue
            yield Hit(
                embedding_id=embedding_id,
                chrom=chrom,
                pos=pos,
                variant_id=norm(row.get(args.id_col)) or f"chr{chrom}:{pos}",
                p=p,
                beta=norm(row.get(args.beta_col)),
                se=norm(row.get(args.se_col)),
            )


def merge_regions(hits: Iterable[Hit], window_bp: int) -> list[dict[str, object]]:
    sorted_hits = sorted(hits, key=lambda hit: (int(hit.chrom), hit.pos, hit.p))
    regions: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for hit in sorted_hits:
        if (
            current is None
            or current["chrom"] != hit.chrom
            or hit.pos > int(current["end"]) + window_bp
        ):
            current = {
                "chrom": hit.chrom,
                "start": hit.pos,
                "end": hit.pos,
                "lead_variant": hit.variant_id,
                "lead_p": hit.p,
                "lead_embedding": hit.embedding_id,
                "hit_count": 1,
                "embeddings": {hit.embedding_id},
            }
            regions.append(current)
        else:
            current["end"] = max(int(current["end"]), hit.pos)
            current["hit_count"] = int(current["hit_count"]) + 1
            current["embeddings"].add(hit.embedding_id)  # type: ignore[union-attr]
            if hit.p < float(current["lead_p"]):
                current["lead_variant"] = hit.variant_id
                current["lead_p"] = hit.p
                current["lead_embedding"] = hit.embedding_id
    for region in regions:
        embeddings = sorted(region["embeddings"])  # type: ignore[arg-type]
        region["embedding_count"] = len(embeddings)
        region["embeddings"] = ",".join(embeddings)
    return regions


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_threshold(paths: list[Path], threshold: float, regex: re.Pattern[str], args: argparse.Namespace, label: str):
    per_embedding: list[dict[str, object]] = []
    all_hits: list[Hit] = []
    for path in paths:
        embedding_id = extract_embedding_id(path, regex)
        hits = list(iter_hits(path, embedding_id, threshold, args))
        all_hits.extend(hits)
        regions = merge_regions(hits, args.window_bp)
        top = min(hits, key=lambda hit: hit.p) if hits else None
        per_embedding.append(
            {
                "embedding_id": embedding_id,
                f"{label}_significant_variants": len(hits),
                f"{label}_loci_count": len(regions),
                f"{label}_min_p": "" if top is None else f"{top.p:.6g}",
                f"{label}_top_locus": "" if top is None else f"chr{top.chrom}:{top.pos}:{top.variant_id}",
            }
        )
    unique_regions = merge_regions(all_hits, args.window_bp)
    return per_embedding, unique_regions


def main() -> int:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.sumstats_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched --sumstats-glob {args.sumstats_glob!r}")
    regex = re.compile(args.embedding_regex)
    strict = args.strict_threshold if args.strict_threshold is not None else args.p_threshold / args.embedding_count
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    p5_rows, p5_regions = summarize_threshold(paths, args.p_threshold, regex, args, "p5e8")
    strict_rows, strict_regions = summarize_threshold(paths, strict, regex, args, "strict")
    strict_by_embedding = {row["embedding_id"]: row for row in strict_rows}
    merged_rows: list[dict[str, object]] = []
    for row in p5_rows:
        combined = dict(row)
        combined.update(strict_by_embedding.get(row["embedding_id"], {}))
        merged_rows.append(combined)
    merged_rows.sort(
        key=lambda row: (
            -int(row.get("p5e8_loci_count", 0)),
            -int(row.get("strict_loci_count", 0)),
            row["embedding_id"],
        )
    )

    write_tsv(
        outdir / "per_embedding_loci.tsv",
        [
            "embedding_id",
            "p5e8_significant_variants",
            "p5e8_loci_count",
            "p5e8_min_p",
            "p5e8_top_locus",
            "strict_significant_variants",
            "strict_loci_count",
            "strict_min_p",
            "strict_top_locus",
        ],
        merged_rows,
    )
    region_fields = [
        "chrom",
        "start",
        "end",
        "lead_variant",
        "lead_p",
        "lead_embedding",
        "hit_count",
        "embedding_count",
        "embeddings",
    ]
    write_tsv(outdir / "unique_regions_p5e8.tsv", region_fields, p5_regions)
    write_tsv(outdir / "unique_regions_strict.tsv", region_fields, strict_regions)
    write_tsv(
        outdir / "loci_summary.tsv",
        ["metric", "value"],
        [
            {"metric": "sumstats_files", "value": len(paths)},
            {"metric": "p_threshold", "value": args.p_threshold},
            {"metric": "strict_threshold", "value": strict},
            {"metric": "p5e8_unique_regions", "value": len(p5_regions)},
            {"metric": "strict_unique_regions", "value": len(strict_regions)},
        ],
    )
    print(f"Wrote locus summaries to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
