# Omni-fMRI UKB Analysis Workflow

This workflow describes a reusable Omni-fMRI analysis line from UKB rs-fMRI
images to GWAS, LDSC, validation, and NeuroSTORM comparison. It is designed to
match the existing NeuroSTORM old / updated MAE-5ds analysis logic wherever
possible.

## Confirmed Omni-fMRI Code Map

| Area | File | Confirmed role |
| --- | --- | --- |
| Human docs | `README.md` | Local project overview, install, checkpoint, extraction, training |
| Codex handoff | `README.codex.md` | Local branch, entry point, and operational notes |
| Inference / embedding extraction | `extract_feat.py` | Correct frozen backbone extraction entry point |
| Preprocessing | `data_preparation/preprocessing.py` | NIfTI to normalized `96 x 96 x 96 x 40` NPZ segments |
| Pre-training config | `configs/pretrain.yaml` | Default model shape, `in_chans=40`, `embed_dim=768` |
| Fine-tuning config | `configs/finetune.yaml` | Downstream task defaults and checkpoint path |
| Pre-training data | `src/data/pretrain_dataset.py` | Directory/TXT NPZ loading and nonzero global z-score |
| Downstream data | `src/data/downstream_dataset.py` | Subject ID extraction by regex and label CSV alignment |
| Model | `src/models/mae_model.py` | Adaptive MAE wrapper |
| Model tokenizer | `src/models/patch_tokenizer_3d.py`, `src/models/patch_embed_3d.py` | Dynamic 3D patch tokenization and embedding |
| CLI/config | `src/utils/cli_app.py`, `src/utils/config_overrides.py` | YAML-backed CLI and dotted config overrides |
| HPC launchers | `scripts/pretrain.sh`, `scripts/finetune.sh` | `torchrun` wrappers for training/fine-tuning |

## Confirmed Inference Interface

`extract_feat.py` is the correct inference entry point. It:

- loads `pretrain_checkpoint/checkpoint.pth` by default;
- uses checkpoint `config` if present, otherwise `configs/pretrain.yaml`;
- constructs `AdaptiveMAE` with `TokenizedZeroConvPatchAttn3D`;
- extracts the encoder/backbone output without training;
- accepts one `.npz` file or a directory of `.npz` files;
- supports layouts `auto`, `dhwt`, and `cdhw`;
- writes one `*_tokens.npz` per input sample.

Input array requirements confirmed from code:

- 4D NPZ array;
- default spatial shape `96 x 96 x 96`;
- default temporal/channel length `40`;
- `DHWT` arrays are transposed to `CDHW`;
- arrays longer than 40 frames use `--start-frame`;
- shorter `DHWT` arrays require `--pad-short`.

Output arrays confirmed from code:

| Array | Shape | Meaning |
| --- | --- | --- |
| `cls_token` | `(768,)` by default | Global sample representation |
| `patch_tokens` | `(num_patches, 768)` | Dynamic patch token embeddings |
| `patch_coords` | `(num_patches, 3)` | Patch coordinates in voxel space |

Subject ID alignment is not handled by `extract_feat.py`. It preserves file
names only. UKB `eid` alignment, one-row-per-subject aggregation, missing logs,
and merged embedding TSV generation are provided by
`scripts/omni_pipeline/extract_omni_embeddings.py`.

## Workflow Overview

```text
UKB subject list + rs-fMRI image/NPZ paths
-> manifest with pure eid
-> Omni-fMRI checkpointed inference
-> eid + emb_001 ... emb_768 TSV
-> embedding QC
-> covariate merge + RankINT
-> PLINK2 screening GWAS and SAIGE handoff
-> LDSC h2
-> loci counting and novelty relative to Big40/Zhao
-> structural Brain-IDP mapping
-> ENIGMA rg validation
-> integrated Omni priority table
-> NeuroSTORM tracker-compatible summary
```

## A. Embedding Extraction

Inputs:

- UKB subject list with pure `eid`.
- UKB rs-fMRI imaging path manifest or directory scan.
- Omni checkpoint, expected default `pretrain_checkpoint/checkpoint.pth`.
- Omni config, default `configs/pretrain.yaml`.

Outputs:

- `embeddings.tsv`: `eid`, `emb_001`, ..., `emb_768`.
- `embeddings.failures.tsv`: failed subjects and errors.
- `embeddings.missing_subjects.tsv`: missing image paths.
- `embeddings.qc_summary.tsv`: row counts, embedding width, finite fraction.
- `embeddings.segment_qc.tsv`: per-subject segment count and aggregation mode.

Implementation:

- `prepare_ukb_manifest.py` builds a manifest from an existing table or path scan.
- `extract_omni_embeddings.py` reuses Omni `extract_feat.py` functions for
  checkpoint loading, model construction, tensor conversion, and token extraction.
- If a subject has multiple 40-frame segments, the default subject-level
  embedding is the mean of segment CLS tokens. This is an explicit workflow
  assumption and can be changed with `--segment-aggregation first`.

TODO before production:

- Confirm whether UKB inputs are already preprocessed NPZ or raw NIfTI.
- Confirm the preferred handling of multiple UKB fMRI instances. The GWAS path
  must keep `eid` as the identifier and must not use image-instance IDs.

## B. GWAS Input Preparation

Inputs:

- Omni `embeddings.tsv`.
- Covariate table with pure `eid`.

Default covariates aligned with the existing NeuroSTORM PLINK2 route:

- `age`
- `sex`
- `PC1` to `PC10`
- `scanner_age_time_since_first_mri`

Outputs:

- `embeddings.tsv`: `FID`, `IID`, `eid`, RankINT `emb_001` ... `emb_768`.
- `covariates.tsv`: `FID`, `IID`, `eid`, selected covariates.
- `sample_inclusion_summary.tsv`.
- `pheno_manifest.tsv`.

Rules:

- `FID=IID=pure_UKB_eid`.
- Never use image-instance IDs such as `1013356_20227_2_0` as FID/IID.
- RankINT is applied separately to each embedding after covariate complete-case
  filtering.

## C. GWAS

Two routes are defined.

### PLINK2 Screening Route

Purpose:

- fast model-level screening;
- locus prioritization;
- preparation for LDSC h2.

PBS template:

- `scripts/omni_pipeline/submit_plink2_gwas.pbs`

Default model:

```text
plink2 --glm hide-covar --maf 0.01 --mac 10
```

Interpretation limit:

PLINK2 does not model UKB relatedness. Treat PLINK2 loci as screening or
prioritization signals only, not final discovery claims.

### SAIGE Handoff Route

Purpose:

- formal relatedness-aware GWAS;
- final comparison route for locus discovery claims.

Handoff layout should reuse:

- same RankINT `embeddings.tsv`;
- same `covariates.tsv`;
- pure `eid` FID/IID;
- per-embedding phenotype names `emb_001` ... `emb_768`;
- explicit sample inclusion summary.

TODO:

- Confirm Santiago's exact SAIGE phenotype/covariate file schema and GRM/null
  model inputs.
- Add a thin SAIGE handoff writer only after the required SAIGE column names are
  confirmed.

## D. LDSC h2

Inputs:

- merged genome-wide PLINK2 summary statistics per embedding;
- LDSC software;
- HapMap3 merge-alleles SNP list;
- EUR LD score reference and regression weights.

PBS template:

- `scripts/omni_pipeline/submit_ldsc_h2.pbs`

Outputs:

- munged sumstats per embedding;
- LDSC h2 logs/results;
- summary with mean h2, h2 range, intercept, and top h2 embeddings.

## E. Loci Counting

Script:

- `scripts/omni_pipeline/count_loci.py`

Outputs:

- `per_embedding_loci.tsv`
- `unique_regions_p5e8.tsv`
- `unique_regions_strict.tsv`
- `loci_summary.tsv`

Thresholds:

- standard: `P < 5e-8`;
- strict: `P < 5e-8 / 768`.

Approximate regions are generated by merging significant variants on the same
chromosome within the configured base-pair window. This matches the current
screening-summary role and is not a substitute for formal fine-mapping.

## F. Benchmark And Comparison

Compare Omni-fMRI with:

- NeuroSTORM old / anchor embeddings;
- NeuroSTORM updated MAE-5ds;
- Big40/Zhao ICA reference loci where available.

Allowed comparisons:

- h2 distributions;
- loci count distributions;
- strict loci distributions;
- unique genomic region overlap/recovery;
- region novelty relative to Big40/Zhao;
- ENIGMA and structural validation burden.

Disallowed comparison:

- Do not compare `emb_001` to `emb_001` across models. Embedding dimensions are
  model-specific latent axes.

## G. Structural Brain-IDP Mapping

Reuse the NeuroSTORM structural mapping design:

- residualize embeddings and scalar UKB brain IDPs on covariates;
- test all 768 Omni embeddings against UKB scalar brain IDPs;
- summarize conservative anatomical subset separately;
- report top IDP hits, residual/partial `r`, FDR, and modality summary.

Output expectations:

- per-embedding association tables;
- all-embedding top hit table;
- conservative anatomical summary;
- modality summary.

Important caveat:

ICV/head-size sensitivity must be added before strong morphology claims,
especially for global volume, surface area, or head-size-sensitive IDPs.

## H. ENIGMA rg

Reuse the existing ENIGMA 77 usable trait workflow if available.

Target grid:

```text
768 Omni embedding GWAS traits x 77 ENIGMA traits = 59,136 LDSC rg tests
```

Summaries:

- parsed tests;
- nominal rg;
- FDR hits;
- positive/negative rg;
- sensitivity excluding ICV/global traits;
- top trait per embedding;
- report/plots if existing plotting scripts are available.

## I. Integrated Priority Table

Script:

- `scripts/omni_pipeline/build_priority_table.py`

Required output columns:

```text
embedding_id
h2
h2_se
loci_count
strict_loci_count
top_locus
ENIGMA_top_trait
ENIGMA_top_rg
structural_top_idp
structural_top_abs_r
novelty_status
cluster/module if available
priority_reason
```

This table is the main Omni result surface for collaborator review.

## J. Tracker Update

Script:

- `scripts/omni_pipeline/summarize_omni_results.py`

Output:

- `omni_tracker_compatible_summary.tsv`

Schema:

```text
big_parts
detailed_part
neurostorm_old
neurostorm_updated
omni_fmri
```

The Omni column should move from `pending` to completed numeric summaries as
each stage finishes.
