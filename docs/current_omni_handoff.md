# Current Omni-fMRI Handoff

Last updated: 2026-07-17

This handoff is for restarting the Omni-fMRI UKB 20227 analysis in a new Codex task. It records the current repository state, scientific decisions, external paths, exact restart commands, and known pitfalls.

## Repository State

Local project path:

```text
/Users/junzhou/Desktop/Main_Project/Omni-fMRI
```

Current branch:

```text
codex/junzhou-work
```

Core handoff/readme files:

```text
README.md
README.codex.md
docs/omni_fmri_analysis_workflow.md
docs/omni_fmri_hpc_runbook.md
docs/omni_fmri_tracker_summary.md
docs/current_omni_handoff.md
```

Pipeline scripts currently expected under:

```text
scripts/omni_pipeline/
  prepare_header_ready_ukb_20227_manifest.py
  prepare_ukb_manifest.py
  extract_omni_embeddings.py
  merge_tsv_shards.py
  filter_omni_embeddings_by_subject_list.py
  prepare_gwas_inputs.py
  prepare_saige_handoff.py
  submit_omni_extraction.pbs
  submit_plink2_gwas.pbs
  merge_plink2_sumstats.py
  submit_ldsc_h2.pbs
  parse_ldsc_h2_results.py
  count_loci.py
  summarize_omni_results.py
  build_priority_table.py
```

Before continuing in a new Codex task, run:

```bash
cd /Users/junzhou/Desktop/Main_Project/Omni-fMRI
git status --short --branch
ls -la docs scripts/omni_pipeline
```

## Current Decision

Use this policy:

```text
Omni inference: keep all case-level 4D NIfTI rows.
Omni GWAS: filter Omni all-case embeddings to NeuroSTORM MAE-5ds one-instance-per-eid subject_id list.
GWAS FID/IID: pure UKB eid only.
```

Rationale:

- NeuroSTORM MAE-5ds final GWAS/reporting used `--participant-duplicate-policy first`.
- The corresponding subject list is the exact one-instance-per-eid case selection.
- Running Omni inference for all case-level NIfTI first preserves a complete reusable inference archive.
- Filtering after inference gives the same GWAS cohort definition as NeuroSTORM without losing case-level Omni outputs.

Do not use NeuroSTORM `.pt` frame files as Omni input. Omni should start from 4D MNI NIfTI or NPZ segments generated from those NIfTI files.

## Confirmed Omni Interface

Confirmed from code:

- `extract_feat.py` is the correct frozen backbone extraction entry point.
- Default config is `configs/pretrain.yaml`.
- Default model input is 4D data with spatial shape `96 x 96 x 96` and temporal/channel length `40`.
- NIfTI preprocessing is handled by `data_preparation/preprocessing.py`.
- Inference actually consumes NPZ arrays; the wrapper can call preprocessing for NIfTI.
- Default `model.embed_dim` is `768`.
- Subject-level embedding should use `cls_token`, written as `emb_001` ... `emb_768`.

Current wrapper behavior:

```text
scripts/omni_pipeline/prepare_header_ready_ukb_20227_manifest.py
  Produces the current audited header-ready all-case manifest with eid, case_id, tag, nifti_path, and header audit columns.

scripts/omni_pipeline/prepare_ukb_manifest.py
  Produces case-level manifest with eid, subject_id, sample_id, image_path, batch, input_kind.

scripts/omni_pipeline/extract_omni_embeddings.py
  Reads manifest and outputs eid, subject_id, sample_id, batch, image_path, emb_001..emb_768.

scripts/omni_pipeline/merge_tsv_shards.py
  Merges same-schema PBS array TSV shards after validating one consistent header.

scripts/omni_pipeline/filter_omni_embeddings_by_subject_list.py
  Filters all-case Omni embeddings to an explicit NeuroSTORM subject_id keep list.

scripts/omni_pipeline/prepare_gwas_inputs.py
  Merges filtered embeddings with covariates, applies RankINT per embedding, and writes GWAS-ready phenotype/covariate tables.

scripts/omni_pipeline/submit_omni_extraction.pbs
  Runs extract_omni_embeddings.py as a sharded PBS array using --shard-index and --num-shards.

scripts/omni_pipeline/prepare_saige_handoff.py
  Packages validated RankINT phenotype/covariate tables for Santiago's relatedness-aware SAIGE route without assuming final SAIGE command paths.

scripts/omni_pipeline/merge_plink2_sumstats.py
  Merges chr1-22 PLINK2 screening outputs into one genome-wide sumstats file per embedding.

scripts/omni_pipeline/parse_ldsc_h2_results.py
  Parses LDSC h2 logs into per-embedding and model-level h2/intercept summary tables.
```

## UKB 20227 NIfTI Status

UKB root on HPC:

```text
/working/lab_puyag/bingjinZ/UKBB/
```

Header-ready MNI 4D NIfTI batches:

```text
mni_4d_20227_casebatch_0001_rest9800             8,854 files, TR=0.735, dtype=float32
mni_4d_20227_casebatch_0002                      9,016 files, TR=0.735, dtype=float32
mni_4d_20227_casebatch_0003                      9,037 files, TR=0.735, dtype=float32
mni_4d_20227_casebatch_0009_missing_afterbench100 178 files, TR=0.735, dtype=float32
```

Total header-ready files so far:

```text
27,085 NIfTI files
```

Header-fix-needed batches before Omni use:

```text
mni_4d_20227_casebatch_0004  9,039 files, TR=0, dtype=float64
mni_4d_20227_casebatch_0005  9,087 files, TR=0, dtype=float64
mni_4d_20227_casebatch_0006  9,017 files, TR=0, dtype=float64
mni_4d_20227_casebatch_0007  9,089 files, TR=0, dtype=float64
mni_4d_20227_casebatch_0008    171 files, TR=0, dtype=float64
```

Total header-fix-needed files:

```text
36,403 NIfTI files
```

Header read errors from audit:

```text
0
```

## NeuroSTORM First-Case List

Preferred GWAS cohort selection source:

```text
/mnt/lustre/working/lab_puyag/bingjinZ/UKBB/outputs/neurostorm_embeddings_20227_mae_5ds/neurostorm_mae_5ds_7batch_one_instance_per_eid.subjects.txt
```

Associated files from prior NeuroSTORM records:

```text
/mnt/lustre/working/lab_puyag/bingjinZ/UKBB/outputs/neurostorm_embeddings_20227_mae_5ds/neurostorm_mae_5ds_7batch_one_instance_per_eid.tsv
/mnt/lustre/working/lab_puyag/bingjinZ/UKBB/outputs/neurostorm_embeddings_20227_mae_5ds/neurostorm_mae_5ds_7batch_one_instance_per_eid.eids.txt
```

NeuroSTORM prior counts:

```text
Rows before participant dedup: 63,135
Rows after participant dedup: 58,734
Rows dropped by participant dedup: 4,401
```

The `subject_id` format is expected to be:

```text
eid_20227_instance_array
```

Example regex used by Omni manifest generation:

```text
(?P<subject_id>[0-9]+_20227_[0-9]+_[0-9]+)
```

## Restart Commands

Run these on the HPC clone of Omni-fMRI. Adjust only `cd`, `CHECKPOINT`, and covariate paths if needed.

### 1. Environment Variables

```bash
cd /working/lab_puyag/bingjinZ/Omni-fMRI

export UKB_ROOT=/working/lab_puyag/bingjinZ/UKBB
export OMNI_OUT=/working/lab_puyag/bingjinZ/UKBB/omni_fmri
export CHECKPOINT=/working/lab_puyag/bingjinZ/Omni-fMRI/pretrain_checkpoint/checkpoint.pth
export NS_FIRST_SUBJECTS=/mnt/lustre/working/lab_puyag/bingjinZ/UKBB/outputs/neurostorm_embeddings_20227_mae_5ds/neurostorm_mae_5ds_7batch_one_instance_per_eid.subjects.txt

mkdir -p ${OMNI_OUT}/manifests
mkdir -p ${OMNI_OUT}/embeddings
mkdir -p ${OMNI_OUT}/gwas_inputs_neurostorm_first_cases
```

### 2. Generate Header-Ready All-Case Manifests

Important: include `--allow-duplicates`. Without it, `prepare_ukb_manifest.py` keeps only one row per `eid`, which is not the desired all-case inference behavior.

```bash
python scripts/omni_pipeline/prepare_ukb_manifest.py \
  --scan-root ${UKB_ROOT} \
  --scan-glob 'mni_4d_20227_casebatch_0001_rest9800/*.nii.gz' \
  --eid-regex '(?P<eid>[0-9]{7})' \
  --subject-id-regex '(?P<subject_id>[0-9]+_20227_[0-9]+_[0-9]+)' \
  --input-kind nifti \
  --allow-duplicates \
  --output ${OMNI_OUT}/manifests/manifest_0001_rest9800.tsv
```

```bash
python scripts/omni_pipeline/prepare_ukb_manifest.py \
  --scan-root ${UKB_ROOT} \
  --scan-glob 'mni_4d_20227_casebatch_0002/*.nii.gz' \
  --eid-regex '(?P<eid>[0-9]{7})' \
  --subject-id-regex '(?P<subject_id>[0-9]+_20227_[0-9]+_[0-9]+)' \
  --input-kind nifti \
  --allow-duplicates \
  --output ${OMNI_OUT}/manifests/manifest_0002.tsv
```

```bash
python scripts/omni_pipeline/prepare_ukb_manifest.py \
  --scan-root ${UKB_ROOT} \
  --scan-glob 'mni_4d_20227_casebatch_0003/*.nii.gz' \
  --eid-regex '(?P<eid>[0-9]{7})' \
  --subject-id-regex '(?P<subject_id>[0-9]+_20227_[0-9]+_[0-9]+)' \
  --input-kind nifti \
  --allow-duplicates \
  --output ${OMNI_OUT}/manifests/manifest_0003.tsv
```

```bash
python scripts/omni_pipeline/prepare_ukb_manifest.py \
  --scan-root ${UKB_ROOT} \
  --scan-glob 'mni_4d_20227_casebatch_0009_missing_afterbench100/*.nii.gz' \
  --eid-regex '(?P<eid>[0-9]{7})' \
  --subject-id-regex '(?P<subject_id>[0-9]+_20227_[0-9]+_[0-9]+)' \
  --input-kind nifti \
  --allow-duplicates \
  --output ${OMNI_OUT}/manifests/manifest_0009_missing_afterbench100.tsv
```

### 3. Merge Manifests

```bash
head -n 1 ${OMNI_OUT}/manifests/manifest_0001_rest9800.tsv \
  > ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv

tail -n +2 ${OMNI_OUT}/manifests/manifest_0001_rest9800.tsv \
  >> ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv

tail -n +2 ${OMNI_OUT}/manifests/manifest_0002.tsv \
  >> ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv

tail -n +2 ${OMNI_OUT}/manifests/manifest_0003.tsv \
  >> ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv

tail -n +2 ${OMNI_OUT}/manifests/manifest_0009_missing_afterbench100.tsv \
  >> ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv
```

Expected `wc -l` values:

```bash
wc -l ${OMNI_OUT}/manifests/manifest_0001_rest9800.tsv
wc -l ${OMNI_OUT}/manifests/manifest_0002.tsv
wc -l ${OMNI_OUT}/manifests/manifest_0003.tsv
wc -l ${OMNI_OUT}/manifests/manifest_0009_missing_afterbench100.tsv
wc -l ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv
```

Expected output:

```text
8855
9017
9038
179
27086
```

### 4. Dry-Run Manifest Check

```bash
python scripts/omni_pipeline/extract_omni_embeddings.py \
  --manifest ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv \
  --checkpoint ${CHECKPOINT} \
  --output-tsv ${OMNI_OUT}/embeddings/omni_header_ready_all_cases.tsv \
  --work-dir ${OMNI_OUT}/embeddings/work_header_ready_all_cases \
  --input-kind nifti \
  --dry-run
```

### 5. Head10 Real Inference

```bash
python scripts/omni_pipeline/extract_omni_embeddings.py \
  --manifest ${OMNI_OUT}/manifests/manifest_header_ready_all_cases.tsv \
  --checkpoint ${CHECKPOINT} \
  --output-tsv ${OMNI_OUT}/embeddings/omni_header_ready_head10.tsv \
  --work-dir ${OMNI_OUT}/embeddings/work_head10 \
  --input-kind nifti \
  --device cuda:0 \
  --limit 10 \
  --force
```

Check outputs:

```bash
head -n 2 ${OMNI_OUT}/embeddings/omni_header_ready_head10.tsv
cat ${OMNI_OUT}/embeddings/omni_header_ready_head10.qc_summary.tsv
cat ${OMNI_OUT}/embeddings/omni_header_ready_head10.failures.tsv
```

Expected embedding table columns:

```text
eid subject_id sample_id batch image_path emb_001 ... emb_768
```

### 6. First Full Batch Test: 0009

`casebatch_0009_missing_afterbench100` has only 178 files and is the safest complete-batch test.

```bash
python scripts/omni_pipeline/extract_omni_embeddings.py \
  --manifest ${OMNI_OUT}/manifests/manifest_0009_missing_afterbench100.tsv \
  --checkpoint ${CHECKPOINT} \
  --output-tsv ${OMNI_OUT}/embeddings/omni_0009_missing_afterbench100_all_cases.tsv \
  --work-dir ${OMNI_OUT}/embeddings/work_0009_missing_afterbench100 \
  --input-kind nifti \
  --device cuda:0 \
  --force
```

Check outputs:

```bash
head -n 2 ${OMNI_OUT}/embeddings/omni_0009_missing_afterbench100_all_cases.tsv
cat ${OMNI_OUT}/embeddings/omni_0009_missing_afterbench100_all_cases.qc_summary.tsv
cat ${OMNI_OUT}/embeddings/omni_0009_missing_afterbench100_all_cases.failures.tsv
```

### 7. All Header-Ready Inference

Do not run all 27,085 NIfTI in one large interactive job. After head10 and 0009 pass, create a PBS array/sharded runner.

For now, the intended eventual all-case output path is:

```text
${OMNI_OUT}/embeddings/omni_header_ready_all_cases.tsv
```

### 8. Filter Omni All-Case Embeddings To NeuroSTORM First Cases

Run this after all-case inference output exists:

```bash
python scripts/omni_pipeline/filter_omni_embeddings_by_subject_list.py \
  --embeddings-tsv ${OMNI_OUT}/embeddings/omni_header_ready_all_cases.tsv \
  --keep-subject-list ${NS_FIRST_SUBJECTS} \
  --output-tsv ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.tsv \
  --summary-tsv ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.filter_summary.tsv \
  --unmatched-output ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.unmatched.tsv \
  --force
```

Check filter results:

```bash
cat ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.filter_summary.tsv
head -n 2 ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.tsv
head ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.unmatched.tsv
```

### 9. Prepare GWAS Inputs

Set the real covariate file path before running:

```bash
export UKB_COVARIATES=/path/to/ukb_covariates.tsv
```

Then run:

```bash
python scripts/omni_pipeline/prepare_gwas_inputs.py \
  --embeddings-tsv ${OMNI_OUT}/embeddings/omni_header_ready_neurostorm_first_cases.tsv \
  --covariates-tsv ${UKB_COVARIATES} \
  --outdir ${OMNI_OUT}/gwas_inputs_neurostorm_first_cases \
  --force
```

Check GWAS inputs:

```bash
head -n 2 ${OMNI_OUT}/gwas_inputs_neurostorm_first_cases/embeddings.tsv
head -n 2 ${OMNI_OUT}/gwas_inputs_neurostorm_first_cases/covariates.tsv
cat ${OMNI_OUT}/gwas_inputs_neurostorm_first_cases/sample_inclusion_summary.tsv
```

GWAS phenotype table must have:

```text
FID IID eid emb_001 ... emb_768
```

with:

```text
FID = IID = pure UKB eid
```

## Known Pitfalls

1. Do not omit `--allow-duplicates` during manifest generation if the goal is all-case inference.
   - Without it, the manifest is deduplicated by `eid` and will be smaller.
   - This caused header-ready rows to drop from 27,085 NIfTI to 25,260 rows.

2. All-case Omni inference output is expected to contain duplicate `eid` values.
   - This is intentional.
   - Duplicate `eid` is resolved only before GWAS by filtering to NeuroSTORM first-case `subject_id` list.

3. `subject_id` must match NeuroSTORM format.

```text
eid_20227_instance_array
```

4. Do not use image-instance IDs as GWAS FID/IID.
   - GWAS FID/IID must be pure UKB `eid`.

5. Do not use NeuroSTORM `.pt` files as Omni input.
   - Omni path should be 4D MNI NIfTI -> Omni NPZ segment -> Omni embedding.

6. Large inference should be sharded/PBS array.
   - First pass should be `head10` and then the small 0009 batch.

7. Header-fix-needed batches 0004-0008 should not enter Omni preprocessing until header audit confirms:

```text
pixdim4/TR = 0.735
dtype = float32
```

## Immediate Next Steps

1. In the new Codex task, confirm current branch and scripts exist.
2. Generate the audited all-case manifest for 0001, 0002, 0003, 0009 with `prepare_header_ready_ukb_20227_manifest.py`.
3. Confirm the manifest has 27,086 lines including header and failed-header output has only its header.
4. Run dry-run extraction.
5. Run head10 real inference.
6. Run complete 0009 real inference.
7. Use `submit_omni_extraction.pbs` for sharded full 27,085 header-ready case extraction.
8. After all-case Omni embeddings exist, filter to NeuroSTORM first-case subject list.
9. Prepare GWAS inputs with RankINT and pure `eid` FID/IID.

## Open Items

- Confirm the exact conda environment name used on HPC for Omni dependencies.
- Confirm checkpoint path and whether checkpoint contains its own config.
- Confirm UKB covariate TSV path for GWAS input preparation.
- Build full-scale PBS/sharded inference runner after 0009 succeeds.
- Add header-fixed batches 0004-0008 only after re-audit passes.
