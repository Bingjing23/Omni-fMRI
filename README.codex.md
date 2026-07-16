# Codex Handoff README

This file is for future Codex sessions working on this local Omni-fMRI project.
It records the repository state, important entry points, expected workflows, and
guardrails for making changes without rediscovering the project structure.

## Repository State

- Local path: `/Users/junzhou/Desktop/Main_Project/Omni-fMRI`
- Upstream base: `https://github.com/Bingjing23/Omni-fMRI/tree/main`
- Current working branch: `codex/junzhou-work`
- Default upstream branch: `main`
- Primary human-facing README: `README.md`
- Codex handoff file: `README.codex.md`

Before editing, always check:

```bash
git status --short --branch
git remote -v
```

The repository was cloned as a separate local project under
`/Users/junzhou/Desktop/Main_Project`. It is outside the default Codex writable
workspace used in this session, so future Codex tasks may need filesystem
approval before writing files there.

## Project Purpose

Omni-fMRI provides an atlas-free fMRI foundation model with dynamic 3D patching.
The main workflows are:

1. Preprocess raw 4D NIfTI data into model-ready NPZ files.
2. Extract frozen encoder features from `.npz` files.
3. Pre-train the MAE model on multi-dataset fMRI data.
4. Fine-tune the encoder for downstream classification or regression tasks.

## Key Entry Points

```text
data_preparation/preprocessing.py  Convert raw NIfTI files to NPZ segments
extract_feat.py                    Extract CLS and patch tokens from NPZ inputs
pretrain.py                        MAE pre-training entry point
finetune.py                        Downstream fine-tuning entry point
scripts/pretrain.sh                torchrun wrapper for pre-training
scripts/finetune.sh                torchrun wrapper for fine-tuning
quickstart/quickstart_smoke.py     Synthetic end-to-end smoke test
scripts/docker_smoke_test.py       Container smoke test
```

Configuration files:

```text
configs/pretrain.yaml              Default pre-training configuration
configs/finetune.yaml              Default downstream configuration
```

Core modules:

```text
src/data/pretrain_dataset.py       Pre-training dataset loading
src/data/downstream_dataset.py     Fine-tuning dataset loading
src/models/mae_model.py            Adaptive MAE implementation
src/models/vision_transformer.py   ViT encoder components
src/models/patch_embed_3d.py       3D patch embedding
src/models/patch_tokenizer_3d.py   Dynamic patch tokenization
src/utils/cli_app.py               YAML-backed CLI handling
src/utils/config_overrides.py      CLI override helpers
src/utils/dist_ddp.py              Distributed training helpers
src/utils/optim.py                 Optimizer and LR helpers
```

## Environment Assumptions

- Python target: 3.11
- PyTorch target: 2.4.1
- CUDA target: 12.4
- Typical execution target: Linux GPU workstation or HPC node
- `requirements.txt` includes a Linux CUDA `flash-attn` wheel URL.

On macOS, use repository inspection, lightweight Python syntax checks, and
documentation edits only unless a compatible environment is available.

## Data and Checkpoint Assumptions

Do not assume real data is present in the repository.

Expected checkpoint location:

```text
pretrain_checkpoint/checkpoint.pth
```

Download command:

```bash
mkdir -p pretrain_checkpoint
huggingface-cli download OneMore1/Omni-fMRI checkpoint.pth \
  --local-dir pretrain_checkpoint
```

Default model input:

```text
Spatial shape: 96 x 96 x 96
Temporal length / channels: 40
Common NPZ layouts: DHWT or CDHW
```

Feature output files contain:

```text
cls_token
patch_tokens
patch_coords
```

## Validation Commands

Use the lightest relevant check first.

Syntax/import-oriented check:

```bash
python -m compileall src pretrain.py finetune.py extract_feat.py
```

Synthetic end-to-end smoke test, requires checkpoint and model dependencies:

```bash
python quickstart/quickstart_smoke.py \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --work-dir outputs/quickstart_smoke \
  --device cuda:0
```

CPU smoke testing may be possible for small checks, but full feature extraction
is expected to be slow and may fail if CUDA-specific dependencies are missing.

## Common Workflows

Preprocess raw NIfTI:

```bash
python data_preparation/preprocessing.py \
  --input_dir /path/to/raw_nifti \
  --output_dir outputs/my_npz \
  --pattern .nii.gz \
  --target_shape 96 96 96 \
  --segment_length 40
```

Extract frozen features:

```bash
python extract_feat.py outputs/my_npz \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --output-dir outputs/my_tokens \
  --device cuda:0 \
  --overwrite
```

Pre-train:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 \
bash scripts/pretrain.sh \
  --output_dir outputs/pretrain \
  --data_root /path/to/data_root \
  --data_mode directory \
  --datasets HCP ABIDE \
  --batch_size 8 \
  --epochs 400
```

Fine-tune:

```bash
bash scripts/finetune.sh \
  --output_dir outputs/finetune \
  --data_root /path/to/data_root \
  --data_mode directory \
  --datasets ABIDE \
  --task_csv /path/to/labels.csv \
  --target_col gender \
  --subject_id_regex '(\\d{7})' \
  --task_type classification \
  --num_classes 2
```

## Editing Guidelines

- Keep scripts non-interactive and HPC-friendly.
- Preserve CLI-first workflows and YAML-backed configuration.
- Avoid hard-coded user-specific paths in source files.
- Put generated artifacts under `outputs/` or another ignored output directory.
- Do not commit real subject data, checkpoints, model outputs, or large derived files.
- When changing data loading, validate both directory mode and TXT mode when possible.
- When changing model or checkpoint logic, keep backward compatibility with checkpoint
  dictionaries containing either `model_state_dict`, `state_dict`, or a raw state dict.
- Prefer small, reviewable commits tied to a single workflow change.

## Likely Failure Points

- Missing checkpoint at `pretrain_checkpoint/checkpoint.pth`.
- `flash-attn` install failure outside Linux/CUDA.
- NPZ files with unexpected array keys or layouts.
- fMRI arrays not shaped as `96 x 96 x 96 x 40` or `40 x 96 x 96 x 96`.
- Subject IDs in file names not matching `data.subject_id_regex`.
- Task CSV missing `Subject` or the selected target column.
- Distributed runs failing because `NUM_GPUS`, `CUDA_VISIBLE_DEVICES`, or
  `MASTER_PORT` is inconsistent with the execution environment.

## Suggested First Actions for Future Codex Sessions

1. Run `git status --short --branch`.
2. Read `README.md` and this file.
3. Inspect the specific entry point relevant to the requested change.
4. Make the smallest coherent code or documentation change.
5. Run the narrowest available validation command.
6. Report any checks that could not be run because of missing CUDA, checkpoint,
   data, or dependencies.
