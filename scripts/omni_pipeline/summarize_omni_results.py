#!/usr/bin/env python3
"""Summarize staged Omni-fMRI analysis outputs into tracker-friendly tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Omni-fMRI analysis status and numeric results.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-qc", default=None, help="*.qc_summary.tsv from extract_omni_embeddings.py.")
    parser.add_argument("--gwas-summary", default=None, help="sample_inclusion_summary.tsv from prepare_gwas_inputs.py.")
    parser.add_argument("--ldsc-summary", default=None, help="LDSC h2 summary TSV/CSV with embedding_id,h2,h2_se,intercept.")
    parser.add_argument("--loci-summary", default=None, help="loci_summary.tsv from count_loci.py.")
    parser.add_argument("--per-embedding-loci", default=None, help="per_embedding_loci.tsv from count_loci.py.")
    parser.add_argument("--enigma-summary", default=None, help="Optional ENIGMA rg summary TSV.")
    parser.add_argument("--structural-summary", default=None, help="Optional structural mapping summary TSV.")
    return parser.parse_args()


def read_metric_table(path: str | None) -> dict[str, str]:
    if not path or not Path(path).is_file():
        return {}
    df = pd.read_csv(path, sep="\t", dtype=str)
    if {"metric", "value"}.issubset(df.columns):
        return dict(zip(df["metric"], df["value"]))
    return {}


def read_table_auto(path: str | None) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fmt(value: object) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "pending"
        return f"{value:.6g}"
    text = str(value)
    return text if text else "pending"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["big_parts", "detailed_part", "neurostorm_old", "neurostorm_updated", "omni_fmri"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    embedding_qc = read_metric_table(args.embedding_qc)
    gwas = read_metric_table(args.gwas_summary)
    loci = read_metric_table(args.loci_summary)
    ldsc = read_table_auto(args.ldsc_summary)
    per_loci = read_table_auto(args.per_embedding_loci)
    enigma = read_table_auto(args.enigma_summary)
    structural = read_table_auto(args.structural_summary)

    ldsc_status = "pending"
    if not ldsc.empty and "h2" in ldsc.columns:
        h2 = numeric(ldsc["h2"])
        ldsc_status = (
            f"Completed/parsed {h2.notna().sum()} embeddings; "
            f"mean h2={fmt(float(h2.mean(skipna=True)))}; "
            f"range={fmt(float(h2.min(skipna=True)))}-{fmt(float(h2.max(skipna=True)))}"
        )

    top_h2 = "pending"
    if not ldsc.empty and {"embedding_id", "h2"}.issubset(ldsc.columns):
        tmp = ldsc.assign(_h2=numeric(ldsc["h2"])).sort_values("_h2", ascending=False).head(5)
        top_h2 = "; ".join(f"{row.embedding_id} {fmt(row._h2)}" for row in tmp.itertuples())

    loci_status = "pending"
    if loci:
        loci_status = (
            f"P<5e-8 regions={loci.get('p5e8_unique_regions', 'pending')}; "
            f"strict regions={loci.get('strict_unique_regions', 'pending')}"
        )

    top_loci = "pending"
    if not per_loci.empty and {"embedding_id", "p5e8_loci_count", "strict_loci_count"}.issubset(per_loci.columns):
        tmp = per_loci.assign(
            _loci=numeric(per_loci["p5e8_loci_count"]),
            _strict=numeric(per_loci["strict_loci_count"]),
        ).sort_values(["_loci", "_strict"], ascending=False).head(5)
        top_loci = "; ".join(
            f"{row.embedding_id} {int(row._loci)}/{int(row._strict)} strict"
            for row in tmp.itertuples()
            if math.isfinite(row._loci)
        )

    enigma_status = "pending"
    if not enigma.empty:
        enigma_status = f"Parsed rows={len(enigma)}; run model-level FDR and ICV/global sensitivity."

    structural_status = "pending"
    if not structural.empty:
        structural_status = f"Parsed rows={len(structural)}; summarize conservative anatomical subset and ICV sensitivity."

    tracker_rows = [
        {
            "big_parts": "Embedding And GWAS Inputs",
            "detailed_part": "Embedding extraction",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": (
                f"{'Completed' if embedding_qc.get('embedded_subjects') else 'pending'}; "
                f"embedded_subjects={embedding_qc.get('embedded_subjects', 'pending')}; "
                f"embedding_width={embedding_qc.get('embedding_width', 'pending')}"
            ),
        },
        {
            "big_parts": "Embedding And GWAS Inputs",
            "detailed_part": "GWAS input preparation",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": (
                f"{'Completed' if gwas.get('merged_rows_complete_covariates') else 'pending'}; "
                f"N={gwas.get('merged_rows_complete_covariates', 'pending')}; "
                "FID=IID=pure_UKB_eid; RankINT per embedding"
            ),
        },
        {
            "big_parts": "SNP Heritability / LDSC",
            "detailed_part": "LDSC h2 completion",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": ldsc_status,
        },
        {
            "big_parts": "SNP Heritability / LDSC",
            "detailed_part": "Top h2 embeddings",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": top_h2,
        },
        {
            "big_parts": "Loci And Discovery Signal",
            "detailed_part": "P < 5e-8 and strict loci",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": loci_status,
        },
        {
            "big_parts": "Loci And Discovery Signal",
            "detailed_part": "Top locus-count embeddings",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": top_loci,
        },
        {
            "big_parts": "ENIGMA Genetic Correlation Validation",
            "detailed_part": "Omni x ENIGMA 768 x 77 rg grid",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": enigma_status,
        },
        {
            "big_parts": "Structural Brain-IDP Mapping",
            "detailed_part": "All-768 structural mapping",
            "neurostorm_old": "see NeuroSTORM tracker",
            "neurostorm_updated": "see NeuroSTORM tracker",
            "omni_fmri": structural_status,
        },
        {
            "big_parts": "Benchmark and comparison",
            "detailed_part": "Omni vs NeuroSTORM old / updated MAE-5ds",
            "neurostorm_old": "model-level distributions and regions only",
            "neurostorm_updated": "model-level distributions and regions only",
            "omni_fmri": "planned; do not compare emb_001 across models",
        },
    ]
    tracker_path = outdir / "omni_tracker_compatible_summary.tsv"
    write_tsv(tracker_path, tracker_rows)

    report = outdir / "omni_summary_report.md"
    report.write_text(
        "# Omni-fMRI Analysis Summary\n\n"
        f"- Embedding extraction: {tracker_rows[0]['omni_fmri']}\n"
        f"- GWAS inputs: {tracker_rows[1]['omni_fmri']}\n"
        f"- LDSC h2: {ldsc_status}\n"
        f"- Loci: {loci_status}\n"
        f"- ENIGMA rg: {enigma_status}\n"
        f"- Structural mapping: {structural_status}\n\n"
        "PLINK2 results are screening outputs only. Use SAIGE or another relatedness-aware route "
        "before final discovery claims.\n",
        encoding="utf-8",
    )
    print(f"Wrote {tracker_path}")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
