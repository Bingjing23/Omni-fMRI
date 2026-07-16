#!/usr/bin/env python3
"""Build an integrated Omni-fMRI embedding priority table."""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "embedding_id",
    "h2",
    "h2_se",
    "loci_count",
    "strict_loci_count",
    "top_locus",
    "ENIGMA_top_trait",
    "ENIGMA_top_rg",
    "structural_top_idp",
    "structural_top_abs_r",
    "novelty_status",
    "cluster/module if available",
    "priority_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Omni priority table from staged outputs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--h2-summary", default=None, help="TSV/CSV with embedding_id,h2,h2_se.")
    parser.add_argument("--loci-summary", default=None, help="per_embedding_loci.tsv.")
    parser.add_argument("--enigma-rg", default=None, help="TSV with embedding_id, trait, rg, p/q columns.")
    parser.add_argument("--structural-glob", default=None, help="Glob for structural association TSVs.")
    parser.add_argument("--novelty", default=None, help="Optional TSV with embedding_id, novelty_status.")
    parser.add_argument("--clusters", default=None, help="Optional TSV with embedding_id and cluster/module.")
    return parser.parse_args()


def read_auto(path: str | None) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    if Path(path).suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def add_h2(rows: dict[str, dict[str, object]], path: str | None) -> None:
    table = read_auto(path)
    if table.empty or "embedding_id" not in table.columns:
        return
    for row in table.itertuples(index=False):
        embedding_id = str(getattr(row, "embedding_id"))
        rows.setdefault(embedding_id, {"embedding_id": embedding_id})
        rows[embedding_id]["h2"] = getattr(row, "h2", "")
        rows[embedding_id]["h2_se"] = getattr(row, "h2_se", "")


def add_loci(rows: dict[str, dict[str, object]], path: str | None) -> None:
    table = read_auto(path)
    if table.empty or "embedding_id" not in table.columns:
        return
    for row in table.itertuples(index=False):
        embedding_id = str(getattr(row, "embedding_id"))
        rows.setdefault(embedding_id, {"embedding_id": embedding_id})
        rows[embedding_id]["loci_count"] = getattr(row, "p5e8_loci_count", "")
        rows[embedding_id]["strict_loci_count"] = getattr(row, "strict_loci_count", "")
        rows[embedding_id]["top_locus"] = getattr(row, "p5e8_top_locus", "")


def add_enigma(rows: dict[str, dict[str, object]], path: str | None) -> None:
    table = read_auto(path)
    if table.empty or "embedding_id" not in table.columns:
        return
    trait_col = first_present(list(table.columns), ["trait", "enigma_trait", "external_trait"])
    rg_col = first_present(list(table.columns), ["rg", "ENIGMA_top_rg"])
    q_col = first_present(list(table.columns), ["q_bh", "fdr", "q", "p"])
    if trait_col is None or rg_col is None:
        return
    table["_abs_rg"] = numeric(table[rg_col]).abs()
    if q_col:
        table["_q"] = numeric(table[q_col])
        table = table.sort_values(["embedding_id", "_q", "_abs_rg"], ascending=[True, True, False])
    else:
        table = table.sort_values(["embedding_id", "_abs_rg"], ascending=[True, False])
    top = table.drop_duplicates("embedding_id", keep="first")
    for row in top.itertuples(index=False):
        embedding_id = str(getattr(row, "embedding_id"))
        rows.setdefault(embedding_id, {"embedding_id": embedding_id})
        rows[embedding_id]["ENIGMA_top_trait"] = getattr(row, trait_col)
        rows[embedding_id]["ENIGMA_top_rg"] = getattr(row, rg_col)


def add_structural(rows: dict[str, dict[str, object]], pattern: str | None) -> None:
    if not pattern:
        return
    files = sorted(glob.glob(pattern))
    if not files:
        return
    tables = [pd.read_csv(path, sep="\t") for path in files]
    table = pd.concat(tables, ignore_index=True)
    embedding_col = first_present(list(table.columns), ["embedding", "embedding_id"])
    idp_col = first_present(list(table.columns), ["idp", "structural_top_idp", "title"])
    r_col = first_present(list(table.columns), ["partial_r", "residual_r", "r"])
    if embedding_col is None or idp_col is None or r_col is None:
        return
    table["_abs_r"] = numeric(table[r_col]).abs()
    top = table.sort_values([embedding_col, "_abs_r"], ascending=[True, False]).drop_duplicates(embedding_col)
    for row in top.itertuples(index=False):
        embedding_id = str(getattr(row, embedding_col))
        rows.setdefault(embedding_id, {"embedding_id": embedding_id})
        rows[embedding_id]["structural_top_idp"] = getattr(row, idp_col)
        rows[embedding_id]["structural_top_abs_r"] = getattr(row, "_abs_r")


def add_simple_lookup(rows: dict[str, dict[str, object]], path: str | None, output_col: str, candidates: list[str]) -> None:
    table = read_auto(path)
    if table.empty or "embedding_id" not in table.columns:
        return
    source_col = first_present(list(table.columns), candidates)
    if source_col is None:
        return
    for row in table.itertuples(index=False):
        embedding_id = str(getattr(row, "embedding_id"))
        rows.setdefault(embedding_id, {"embedding_id": embedding_id})
        rows[embedding_id][output_col] = getattr(row, source_col)


def build_reason(row: dict[str, object]) -> str:
    reasons: list[str] = []
    h2 = pd.to_numeric(pd.Series([row.get("h2")]), errors="coerce").iloc[0]
    loci = pd.to_numeric(pd.Series([row.get("loci_count")]), errors="coerce").iloc[0]
    strict = pd.to_numeric(pd.Series([row.get("strict_loci_count")]), errors="coerce").iloc[0]
    if math.isfinite(h2):
        reasons.append("high/available h2")
    if math.isfinite(loci) and loci > 0:
        reasons.append(f"{int(loci)} P<5e-8 loci")
    if math.isfinite(strict) and strict > 0:
        reasons.append(f"{int(strict)} strict loci")
    if row.get("ENIGMA_top_trait"):
        reasons.append("ENIGMA rg support")
    if row.get("structural_top_idp"):
        reasons.append("structural IDP support")
    if row.get("novelty_status"):
        reasons.append(str(row["novelty_status"]))
    return "; ".join(reasons) if reasons else "pending evidence"


def main() -> int:
    args = parse_args()
    rows: dict[str, dict[str, object]] = {}
    add_h2(rows, args.h2_summary)
    add_loci(rows, args.loci_summary)
    add_enigma(rows, args.enigma_rg)
    add_structural(rows, args.structural_glob)
    add_simple_lookup(rows, args.novelty, "novelty_status", ["novelty_status", "status"])
    add_simple_lookup(rows, args.clusters, "cluster/module if available", ["cluster", "module", "cluster_id"])

    output_rows = []
    for embedding_id in sorted(rows):
        row = {column: rows[embedding_id].get(column, "") for column in OUTPUT_COLUMNS}
        row["priority_reason"] = build_reason(row)
        output_rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS).to_csv(output, sep="\t", index=False)
    print(f"Wrote {output}")
    print(f"Rows: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
