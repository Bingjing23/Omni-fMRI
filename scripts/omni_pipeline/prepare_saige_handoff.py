#!/usr/bin/env python3
"""Package Omni-fMRI GWAS inputs for SAIGE handoff.

This script does not run SAIGE and does not assume collaborator-specific null
model or sparse-GRM paths. It validates the Omni GWAS input tables and writes a
portable handoff directory with phenotypes, covariates, trait manifest, and a
README describing the remaining SAIGE-specific placeholders.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAIGE handoff files for Omni embeddings.")
    parser.add_argument("--gwas-input-dir", required=True, help="Directory with embeddings.tsv and covariates.tsv.")
    parser.add_argument("--output-dir", required=True, help="SAIGE handoff output directory.")
    parser.add_argument("--embedding-prefix", default="emb_")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def ensure_outputs(outdir: Path, force: bool) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "combined": outdir / "saige_phenotypes_covariates.tsv",
        "manifest": outdir / "saige_trait_manifest.tsv",
        "summary": outdir / "saige_handoff_summary.tsv",
        "readme": outdir / "README_saige_handoff.md",
    }
    if not force:
        existing = [str(path) for path in outputs.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Outputs exist; pass --force to overwrite: {existing}")
    return outputs


def main() -> int:
    args = parse_args()
    import pandas as pd

    gwas_dir = Path(args.gwas_input_dir)
    outputs = ensure_outputs(Path(args.output_dir), args.force)
    embeddings = pd.read_csv(gwas_dir / "embeddings.tsv", sep="\t", dtype={"FID": str, "IID": str, "eid": str})
    covariates = pd.read_csv(gwas_dir / "covariates.tsv", sep="\t", dtype={"FID": str, "IID": str, "eid": str})

    required = {"FID", "IID", "eid"}
    for name, table in (("embeddings", embeddings), ("covariates", covariates)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise KeyError(f"Missing columns in {name}: {missing}")
        if not (table["FID"].astype(str).equals(table["IID"].astype(str)) and table["FID"].astype(str).equals(table["eid"].astype(str))):
            raise ValueError(f"{name} must satisfy FID=IID=eid with pure UKB eids.")

    embedding_cols = sorted(column for column in embeddings.columns if column.startswith(args.embedding_prefix))
    if not embedding_cols:
        raise ValueError(f"No embedding columns found with prefix {args.embedding_prefix!r}")
    covariate_cols = [column for column in covariates.columns if column not in {"FID", "IID", "eid"}]

    combined = embeddings[["FID", "IID", "eid"] + embedding_cols].merge(
        covariates[["FID", "IID", "eid"] + covariate_cols],
        on=["FID", "IID", "eid"],
        how="inner",
        validate="one_to_one",
    )
    combined.to_csv(outputs["combined"], sep="\t", index=False, na_rep="NA")

    manifest = pd.DataFrame(
        {
            "embedding_id": embedding_cols,
            "phenotype_col": embedding_cols,
            "phenotype_type": "quantitative_rankint",
            "sample_file": outputs["combined"].name,
            "covariates": ",".join(covariate_cols),
            "fid_iid_rule": "FID=IID=pure_UKB_eid",
            "saige_status": "ready_for_collaborator_null_model_and_step2_paths",
        }
    )
    manifest.to_csv(outputs["manifest"], sep="\t", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "subjects", "value": len(combined)},
            {"metric": "embedding_traits", "value": len(embedding_cols)},
            {"metric": "covariates", "value": len(covariate_cols)},
            {"metric": "combined_file", "value": str(outputs["combined"])},
            {"metric": "trait_manifest", "value": str(outputs["manifest"])},
            {"metric": "remaining_todo", "value": "confirm SAIGE null model, sparse GRM, genotype layout, and exact command schema"},
        ]
    )
    summary.to_csv(outputs["summary"], sep="\t", index=False)

    outputs["readme"].write_text(
        "# Omni-fMRI SAIGE Handoff\n\n"
        "This directory packages RankINT Omni-fMRI phenotypes and covariates for a relatedness-aware SAIGE route.\n\n"
        "Files:\n\n"
        "- `saige_phenotypes_covariates.tsv`: one row per pure UKB `eid`; `FID=IID=eid`; embedding phenotypes plus covariates.\n"
        "- `saige_trait_manifest.tsv`: one row per `emb_001` ... `emb_768` phenotype.\n"
        "- `saige_handoff_summary.tsv`: validation counts and remaining TODOs.\n\n"
        "Use PLINK2 output only as screening. Final discovery claims should use this or another relatedness-aware route.\n\n"
        "TODO before execution: fill collaborator-specific SAIGE null model, sparse GRM, genotype file layout, sample inclusion policy, and Step1/Step2 commands.\n",
        encoding="utf-8",
    )

    print(f"Wrote SAIGE handoff directory: {outputs['combined'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
