#!/usr/bin/env python3
"""Prepare RankINT Omni-fMRI embeddings and covariates for GWAS.

Inputs:
  - Omni embeddings TSV with columns eid, emb_001 ... emb_768.
  - UKB covariate TSV with pure eid and selected covariate columns.

Outputs:
  - embeddings.tsv: FID, IID, eid, RankINT emb_001 ... emb_768.
  - covariates.tsv: FID, IID, eid, selected covariates.
  - sample_inclusion_summary.tsv.
  - pheno_manifest.tsv describing all embedding phenotypes.

FID and IID are always set to the pure UKB eid. Do not pass image-instance IDs.
"""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COVARIATES = [
    "age",
    "sex",
    "PC1",
    "PC2",
    "PC3",
    "PC4",
    "PC5",
    "PC6",
    "PC7",
    "PC8",
    "PC9",
    "PC10",
    "scanner_age_time_since_first_mri",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Omni-fMRI GWAS input tables.")
    parser.add_argument("--embeddings-tsv", required=True, help="eid + emb_001..emb_768 TSV.")
    parser.add_argument("--covariates-tsv", required=True, help="Covariate TSV with pure eid.")
    parser.add_argument("--outdir", required=True, help="Output directory.")
    parser.add_argument("--eid-column", default="eid")
    parser.add_argument("--covariates", default=",".join(DEFAULT_COVARIATES), help="Comma-separated covariates.")
    parser.add_argument("--embedding-prefix", default="emb_", help="Embedding column prefix. Default: emb_")
    parser.add_argument("--min-nonmissing", type=int, default=1000, help="Minimum nonmissing subjects per embedding.")
    parser.add_argument("--force", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_eid(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.lstrip("0") or "0"
    return text


def rank_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    mask = numeric.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    n = int(mask.sum())
    if n == 0:
        return out
    ranks = numeric[mask].rank(method="average")
    quantiles = (ranks - 0.5) / n
    normal = NormalDist()
    out.loc[mask] = [normal.inv_cdf(float(value)) for value in quantiles]
    return out


def ensure_outputs(outdir: Path, force: bool) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "embeddings": outdir / "embeddings.tsv",
        "covariates": outdir / "covariates.tsv",
        "summary": outdir / "sample_inclusion_summary.tsv",
        "manifest": outdir / "pheno_manifest.tsv",
    }
    if not force:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs exist; pass --force to overwrite: {existing}")
    return outputs


def main() -> int:
    args = parse_args()
    outputs = ensure_outputs(Path(args.outdir), args.force)
    covariate_cols = split_csv(args.covariates)

    embeddings = pd.read_csv(args.embeddings_tsv, sep="\t", dtype={args.eid_column: str})
    covariates = pd.read_csv(args.covariates_tsv, sep="\t", dtype={args.eid_column: str}, low_memory=False)
    for frame_name, frame in (("embeddings", embeddings), ("covariates", covariates)):
        if args.eid_column not in frame.columns:
            raise KeyError(f"Missing {args.eid_column} column in {frame_name}")
        frame[args.eid_column] = frame[args.eid_column].map(normalize_eid)

    embedding_cols = sorted(column for column in embeddings.columns if column.startswith(args.embedding_prefix))
    if not embedding_cols:
        raise ValueError(f"No embedding columns found with prefix {args.embedding_prefix!r}")
    missing_covariates = [column for column in covariate_cols if column not in covariates.columns]
    if missing_covariates:
        raise KeyError(f"Missing covariate columns: {missing_covariates}")

    before_embeddings = len(embeddings)
    before_covariates = len(covariates)
    embeddings = embeddings.drop_duplicates(args.eid_column, keep="first")
    covariates = covariates.drop_duplicates(args.eid_column, keep="first")
    merged = embeddings[[args.eid_column] + embedding_cols].merge(
        covariates[[args.eid_column] + covariate_cols],
        on=args.eid_column,
        how="inner",
    )

    complete_cov = merged[covariate_cols].notna().all(axis=1)
    merged = merged.loc[complete_cov].copy()
    for column in embedding_cols:
        merged[column] = rank_int(merged[column])

    keep_embedding_cols = [
        column
        for column in embedding_cols
        if int(pd.to_numeric(merged[column], errors="coerce").notna().sum()) >= args.min_nonmissing
    ]
    if len(keep_embedding_cols) != len(embedding_cols):
        dropped = sorted(set(embedding_cols) - set(keep_embedding_cols))
        print(f"Dropping embeddings below --min-nonmissing: {dropped}")

    merged.insert(0, "IID", merged[args.eid_column])
    merged.insert(0, "FID", merged[args.eid_column])
    merged = merged.rename(columns={args.eid_column: "eid"})

    embeddings_out = merged[["FID", "IID", "eid"] + keep_embedding_cols].copy()
    covariates_out = merged[["FID", "IID", "eid"] + covariate_cols].copy()
    embeddings_out.to_csv(outputs["embeddings"], sep="\t", index=False, na_rep="NA")
    covariates_out.to_csv(outputs["covariates"], sep="\t", index=False, na_rep="NA")

    manifest = pd.DataFrame(
        {
            "embedding_id": keep_embedding_cols,
            "pheno_name": keep_embedding_cols,
            "n_nonmissing": [
                int(pd.to_numeric(merged[column], errors="coerce").notna().sum())
                for column in keep_embedding_cols
            ],
            "strict_threshold": [5e-8 / len(keep_embedding_cols)] * len(keep_embedding_cols),
        }
    )
    manifest.to_csv(outputs["manifest"], sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "embedding_rows_input", "value": before_embeddings},
            {"metric": "covariate_rows_input", "value": before_covariates},
            {"metric": "embedding_rows_unique_eid", "value": len(embeddings)},
            {"metric": "covariate_rows_unique_eid", "value": len(covariates)},
            {"metric": "merged_rows_complete_covariates", "value": len(merged)},
            {"metric": "embedding_columns_input", "value": len(embedding_cols)},
            {"metric": "embedding_columns_output", "value": len(keep_embedding_cols)},
            {"metric": "fid_iid_rule", "value": "FID=IID=pure_UKB_eid"},
            {"metric": "rankint", "value": "applied_per_embedding_after_covariate_complete_case_filter"},
            {"metric": "plink2_role", "value": "screening_only_not_final_discovery_claim"},
        ]
    )
    summary.to_csv(outputs["summary"], sep="\t", index=False)

    print(f"Wrote {outputs['embeddings']}")
    print(f"Wrote {outputs['covariates']}")
    print(f"Merged complete-case subjects: {len(merged)}; embeddings: {len(keep_embedding_cols)}")
    if any(embeddings_out["FID"].astype(str) != embeddings_out["IID"].astype(str)):
        raise RuntimeError("FID/IID mismatch detected.")
    if not math.isclose(5e-8 / len(keep_embedding_cols), float(manifest["strict_threshold"].iloc[0])):
        raise RuntimeError("Internal strict-threshold check failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
