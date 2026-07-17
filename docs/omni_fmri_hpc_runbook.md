# Omni-fMRI HPC Runbook

This runbook gives concrete local dry-run and PBS submission commands for the
Omni-fMRI UKB analysis workflow. Replace paths with the server paths used for
your UKB data, checkpoint, genotype, LDSC, and output directories.

All commands assume the repository root is:

```bash
cd /Users/junzhou/Desktop/Main_Project/Omni-fMRI
```

On HPC, use the cloned repository path on the cluster instead.

## 1. Local Syntax And Help Checks

These commands do not run large jobs:

```bash
python -m py_compile scripts/omni_pipeline/*.py

python scripts/omni_pipeline/prepare_header_ready_ukb_20227_manifest.py --help
python scripts/omni_pipeline/prepare_ukb_manifest.py --help
python scripts/omni_pipeline/extract_omni_embeddings.py --help
python scripts/omni_pipeline/merge_tsv_shards.py --help
python scripts/omni_pipeline/prepare_gwas_inputs.py --help
python scripts/omni_pipeline/prepare_saige_handoff.py --help
python scripts/omni_pipeline/merge_plink2_sumstats.py --help
python scripts/omni_pipeline/parse_ldsc_h2_results.py --help
python scripts/omni_pipeline/count_loci.py --help
python scripts/omni_pipeline/summarize_omni_results.py --help
python scripts/omni_pipeline/build_priority_table.py --help
```

## 2. Prepare UKB Manifest

For the current UKB 20227 first-pass Omni run, do not wait for every NIfTI batch.
Start only from the header-ready MNI 4D NIfTI batches:

```text
mni_4d_20227_casebatch_0001_rest9800
mni_4d_20227_casebatch_0002
mni_4d_20227_casebatch_0003
mni_4d_20227_casebatch_0009_missing_afterbench100
```

Do not include these until header repair and re-audit pass:

```text
mni_4d_20227_casebatch_0004
mni_4d_20227_casebatch_0005
mni_4d_20227_casebatch_0006
mni_4d_20227_casebatch_0007
mni_4d_20227_casebatch_0008
```

Generate the audited header-ready all-case manifest on HPC. The audit uses
`np.dtype(img.header.get_data_dtype()).name`, so endian-aware dtypes such as
`<f4` or `>f4` are normalized to `float32` before readiness testing.

```bash
cd /working/lab_puyag/bingjinZ/Omni-fMRI

export UKB_ROOT=/working/lab_puyag/bingjinZ/UKBB
export OMNI_OUT=/working/lab_puyag/bingjinZ/UKBB/omni_fmri

mkdir -p ${OMNI_OUT}/manifests

python scripts/omni_pipeline/prepare_header_ready_ukb_20227_manifest.py \
  --ukb-root ${UKB_ROOT} \
  --output ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv \
  --summary-output ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.summary.tsv \
  --failed-output ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.failed_header.tsv \
  --force
```

Expected ready rows for the current four-batch pass:

```text
8854 + 9016 + 9037 + 178 = 27085
```

The manifest contains both the requested columns:

```text
eid case_id tag nifti_path
```

and Omni wrapper aliases:

```text
subject_id sample_id image_path batch input_kind
```

Check the audit summary before any inference:

```bash
cat ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.summary.tsv
wc -l ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.failed_header.tsv
```

The failed-header file should contain only the header line. If not, stop and
inspect the listed rows before submitting inference:

```bash
head -n 20 ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.failed_header.tsv \
  | cut -f1-4,11-17
```

Generic manifest modes are still available below for later reruns or alternate
input tables.

From an existing imaging table:

```bash
python scripts/omni_pipeline/prepare_ukb_manifest.py \
  --subject-list /path/to/ukb_subjects.tsv \
  --subject-eid-column eid \
  --input-table /path/to/ukb_fmri_paths.tsv \
  --input-eid-column eid \
  --input-path-column image_path \
  --output outputs/omni/manifests/ukb_omni_manifest.tsv \
  --missing-output outputs/omni/manifests/ukb_omni_missing.tsv
```

From a directory scan:

```bash
python scripts/omni_pipeline/prepare_ukb_manifest.py \
  --subject-list /path/to/ukb_subjects.tsv \
  --scan-root /path/to/ukb_fmri_npz \
  --scan-glob '**/*.npz' \
  --eid-regex '(?P<eid>[0-9]{7})' \
  --output outputs/omni/manifests/ukb_omni_manifest.tsv \
  --missing-output outputs/omni/manifests/ukb_omni_missing.tsv
```

## 3. Embedding Extraction

Manifest dry-run:

```bash
python scripts/omni_pipeline/extract_omni_embeddings.py \
  --manifest outputs/omni/manifests/ukb_omni_manifest.tsv \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --output-tsv outputs/omni/embeddings/embeddings.tsv \
  --work-dir outputs/omni/embeddings/work \
  --input-kind npz \
  --dry-run
```

Small local/HPC smoke run:

```bash
python scripts/omni_pipeline/extract_omni_embeddings.py \
  --manifest outputs/omni/manifests/ukb_omni_manifest.tsv \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --output-tsv outputs/omni/embeddings/embeddings.head10.tsv \
  --work-dir outputs/omni/embeddings/work_head10 \
  --input-kind npz \
  --device cuda:0 \
  --limit 10 \
  --force
```

Full extraction should be submitted as a cluster job after the small run passes.
If raw NIfTI is used instead of NPZ, set `--input-kind nifti`; the wrapper will
call the existing Omni preprocessing function and aggregate segment CLS tokens.

Dry-run one PBS extraction shard:

```bash
DRY_RUN=1 \
PBS_ARRAY_INDEX=1 \
MANIFEST=${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv \
CHECKPOINT=/working/lab_puyag/bingjinZ/Omni-fMRI/pretrain_checkpoint/checkpoint.pth \
OMNI_OUT_DIR=${OMNI_OUT}/embeddings/header_ready_shards \
WORK_ROOT=${OMNI_OUT}/embeddings/work_header_ready_shards \
INPUT_KIND=nifti \
PATH_COLUMN=nifti_path \
N_SHARDS=64 \
bash scripts/omni_pipeline/submit_omni_extraction.pbs
```

Submit sharded extraction:

```bash
qsub -J 1-64 \
  -v MANIFEST=${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv,\
CHECKPOINT=/working/lab_puyag/bingjinZ/Omni-fMRI/pretrain_checkpoint/checkpoint.pth,\
OMNI_OUT_DIR=${OMNI_OUT}/embeddings/header_ready_shards,\
WORK_ROOT=${OMNI_OUT}/embeddings/work_header_ready_shards,\
INPUT_KIND=nifti,\
PATH_COLUMN=nifti_path,\
N_SHARDS=64,\
FORCE=1 \
  scripts/omni_pipeline/submit_omni_extraction.pbs
```

After all shards complete, merge shard TSVs:

```bash
python scripts/omni_pipeline/merge_tsv_shards.py \
  --shards-glob "${OMNI_OUT}/embeddings/header_ready_shards/omni_embeddings_shard_*_of_064.tsv" \
  --output ${OMNI_OUT}/embeddings/omni_header_ready_all_cases.tsv \
  --summary ${OMNI_OUT}/embeddings/omni_header_ready_all_cases.merge_summary.tsv \
  --force
```

Also inspect all `*.failures.tsv`, `*.missing_subjects.tsv`, and
`*.qc_summary.tsv` files before using the merged table for GWAS.

## 4. GWAS Inputs

```bash
python scripts/omni_pipeline/prepare_gwas_inputs.py \
  --embeddings-tsv outputs/omni/embeddings/embeddings.tsv \
  --covariates-tsv /path/to/ukb_covariates.tsv \
  --outdir outputs/omni/gwas_inputs \
  --covariates age,sex,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,scanner_age_time_since_first_mri \
  --force
```

Expected outputs:

```text
outputs/omni/gwas_inputs/embeddings.tsv
outputs/omni/gwas_inputs/covariates.tsv
outputs/omni/gwas_inputs/sample_inclusion_summary.tsv
outputs/omni/gwas_inputs/pheno_manifest.tsv
```

Verify that `FID` and `IID` equal pure UKB `eid` before GWAS.

## 5. PLINK2 Screening GWAS

Dry-run one array task:

```bash
DRY_RUN=1 \
PBS_ARRAY_INDEX=1 \
GWAS_INPUT_DIR=/path/to/outputs/omni/gwas_inputs \
PLINK_OUT_DIR=/path/to/outputs/omni/plink2 \
BGEN_PATTERN='/reference/data/UKBB_500k/versions/bgen201803/ukb_imp_chr{CHR}_v3.bgen' \
SAMPLE_PATTERN='/reference/data/UKBB_500k/versions/sample201803/ukb25331_imp_chr{CHR}_v2_s487395.sample' \
bash scripts/omni_pipeline/submit_plink2_gwas.pbs
```

Submit full 768-embedding screening array:

```bash
qsub -J 1-768 \
  -v GWAS_INPUT_DIR=/path/to/outputs/omni/gwas_inputs,\
PLINK_OUT_DIR=/path/to/outputs/omni/plink2,\
BGEN_PATTERN=/reference/data/UKBB_500k/versions/bgen201803/ukb_imp_chr{CHR}_v3.bgen,\
SAMPLE_PATTERN=/reference/data/UKBB_500k/versions/sample201803/ukb25331_imp_chr{CHR}_v2_s487395.sample \
  scripts/omni_pipeline/submit_plink2_gwas.pbs
```

PLINK2 is screening only. Do not use it as the final discovery route without
SAIGE or another relatedness-aware validation.

## 6. SAIGE Handoff

Use the same files:

```text
outputs/omni/gwas_inputs/embeddings.tsv
outputs/omni/gwas_inputs/covariates.tsv
outputs/omni/gwas_inputs/sample_inclusion_summary.tsv
```

Handoff requirements:

- phenotype names `emb_001` ... `emb_768`;
- `FID=IID=pure_UKB_eid`;
- same inclusion set as PLINK2 where possible;
- clear note that embeddings were RankINT transformed.

Create validated handoff files:

```bash
python scripts/omni_pipeline/prepare_saige_handoff.py \
  --gwas-input-dir outputs/omni/gwas_inputs \
  --output-dir outputs/omni/saige_handoff \
  --force
```

TODO:

- Fill Santiago's exact SAIGE null model, sparse GRM, phenotype, and covariate
  command requirements once available.

## 7. Merge PLINK2 Chromosome Outputs

Dry-run the expected chr1-22 merge layout:

```bash
python scripts/omni_pipeline/merge_plink2_sumstats.py \
  --plink-dir /path/to/outputs/omni/plink2 \
  --output-dir /path/to/outputs/omni/sumstats_merged \
  --embedding-count 768 \
  --dry-run
```

Merge genome-wide sumstats after PLINK2 finishes:

```bash
python scripts/omni_pipeline/merge_plink2_sumstats.py \
  --plink-dir /path/to/outputs/omni/plink2 \
  --output-dir /path/to/outputs/omni/sumstats_merged \
  --embedding-count 768 \
  --force
```

Expected output:

```text
/path/to/outputs/omni/sumstats_merged/emb_001.sumstats.tsv.gz
/path/to/outputs/omni/sumstats_merged/merge_plink2_sumstats_summary.tsv
```

The default input glob matches files like
`emb_001/emb_001.chr1*.glm.linear*`. Override `--input-template` if the target
HPC PLINK2 suffix differs.

## 8. LDSC h2

Dry-run one array task:

```bash
DRY_RUN=1 \
PBS_ARRAY_INDEX=1 \
LDSC_DIR=/path/to/ldsc \
SUMSTATS_DIR=/path/to/outputs/omni/sumstats_merged \
MUNGED_DIR=/path/to/outputs/omni/ldsc/munged \
H2_OUT_DIR=/path/to/outputs/omni/ldsc/h2 \
REF_LD_CHR=/path/to/eur_w_ld_chr/ \
W_LD_CHR=/path/to/eur_w_ld_chr/ \
SNP_LIST=/path/to/w_hm3.snplist \
bash scripts/omni_pipeline/submit_ldsc_h2.pbs
```

Submit full 768-embedding LDSC array:

```bash
qsub -J 1-768 \
  -v LDSC_DIR=/path/to/ldsc,\
SUMSTATS_DIR=/path/to/outputs/omni/sumstats_merged,\
MUNGED_DIR=/path/to/outputs/omni/ldsc/munged,\
H2_OUT_DIR=/path/to/outputs/omni/ldsc/h2,\
REF_LD_CHR=/path/to/eur_w_ld_chr/,\
W_LD_CHR=/path/to/eur_w_ld_chr/,\
SNP_LIST=/path/to/w_hm3.snplist \
  scripts/omni_pipeline/submit_ldsc_h2.pbs
```

Check allele columns before real munging. The template defaults to `A1` and
`REF`; change `A2_COL` if the merged sumstats contain a better non-effect allele
column.

Parse LDSC h2 logs:

```bash
python scripts/omni_pipeline/parse_ldsc_h2_results.py \
  --logs-glob '/path/to/outputs/omni/ldsc/h2/emb_*.log' \
  --output outputs/omni/ldsc/omni_h2_summary.tsv \
  --model-summary outputs/omni/ldsc/omni_h2_model_summary.tsv
```

## 9. Loci Counting

```bash
python scripts/omni_pipeline/count_loci.py \
  --sumstats-glob '/path/to/outputs/omni/sumstats_merged/emb_*.sumstats.tsv.gz' \
  --output-dir outputs/omni/loci \
  --embedding-count 768 \
  --window-bp 1000000
```

Outputs:

```text
outputs/omni/loci/per_embedding_loci.tsv
outputs/omni/loci/unique_regions_p5e8.tsv
outputs/omni/loci/unique_regions_strict.tsv
outputs/omni/loci/loci_summary.tsv
```

## 10. ENIGMA rg

Reuse the existing NeuroSTORM ENIGMA LDSC rg inputs if available:

```text
77 h2-usable ENIGMA traits
768 Omni embedding GWAS traits
```

Expected grid:

```text
768 x 77 = 59,136 rg tests
```

TODO:

- Point the existing ENIGMA rg runner to Omni munged sumstats.
- Reuse existing plotting/report scripts if their column names match.

## 11. Structural Brain-IDP Mapping

Reuse the NeuroSTORM mapping logic with:

```text
--embeddings-tsv outputs/omni/gwas_inputs/embeddings.tsv
--covariates-tsv outputs/omni/gwas_inputs/covariates.tsv
```

Run all 768 embeddings, then summarize:

- top IDP per embedding;
- conservative anatomical subset;
- residual `r`;
- FDR;
- modality summary;
- ICV/head-size sensitivity before strong morphology claims.

## 12. Priority Table

```bash
python scripts/omni_pipeline/build_priority_table.py \
  --h2-summary outputs/omni/ldsc/omni_h2_summary.tsv \
  --loci-summary outputs/omni/loci/per_embedding_loci.tsv \
  --enigma-rg outputs/omni/enigma/omni_enigma_rg_summary.tsv \
  --structural-glob 'outputs/omni/structural/*.associations.tsv' \
  --novelty outputs/omni/benchmark/omni_big40_zhao_novelty.tsv \
  --clusters outputs/omni/qc/omni_embedding_clusters.tsv \
  --output outputs/omni/priority/omni_priority_table.tsv
```

## 13. Tracker-Compatible Summary

```bash
python scripts/omni_pipeline/summarize_omni_results.py \
  --output-dir outputs/omni/summary \
  --embedding-qc outputs/omni/embeddings/embeddings.qc_summary.tsv \
  --gwas-summary outputs/omni/gwas_inputs/sample_inclusion_summary.tsv \
  --ldsc-summary outputs/omni/ldsc/omni_h2_summary.tsv \
  --loci-summary outputs/omni/loci/loci_summary.tsv \
  --per-embedding-loci outputs/omni/loci/per_embedding_loci.tsv \
  --enigma-summary outputs/omni/enigma/omni_enigma_rg_summary.tsv \
  --structural-summary outputs/omni/structural/omni_structural_summary.tsv
```

The output table has the tracker-compatible columns:

```text
big_parts
detailed_part
neurostorm_old
neurostorm_updated
omni_fmri
```
