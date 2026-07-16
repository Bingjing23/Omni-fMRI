# Omni-fMRI Local Project

Omni-fMRI is an atlas-free fMRI foundation model for pre-training, downstream
fine-tuning, and feature extraction from 4D fMRI volumes. This local project is
based on the upstream Omni-fMRI repository and is intended for reproducible
research workflows on local workstations or HPC environments.

Upstream resources:

- Paper: https://arxiv.org/abs/2601.23090
- Project page: https://onemore1.github.io/Omni-fMRI/
- Checkpoint: https://huggingface.co/OneMore1/Omni-fMRI
- Upstream repository: https://github.com/Bingjing23/Omni-fMRI

## Repository Layout

```text
configs/                  Default YAML configuration files
data_preparation/          NIfTI-to-NPZ preprocessing utilities
docs/                      Static project page assets
examples/                  Notes for example and synthetic demo data
notebooks/                 Quick-start notebook
pretrain_checkpoint/       Default checkpoint location
quickstart/                End-to-end smoke test with synthetic data
scripts/                   Shell entry points for training and validation
src/data/                  Pre-training and downstream datasets
src/models/                MAE, ViT, patching, and tokenization modules
src/utils/                 CLI, distributed, logging, and optimizer helpers
extract_feat.py            Frozen encoder feature extraction
pretrain.py                MAE pre-training entry point
finetune.py                Downstream fine-tuning entry point
```

## Environment

The default project stack targets Python 3.11, PyTorch 2.4.1, CUDA 12.4, and
Linux-compatible GPU training. CPU execution is useful for lightweight checks,
but real training and feature extraction should use CUDA.

```bash
conda create -n omnifmri python=3.11
conda activate omnifmri

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -U huggingface_hub
```

The `requirements.txt` file includes a Linux CUDA wheel URL for `flash-attn`.
On macOS or CPU-only systems, install dependencies in a Linux/CUDA environment
or adjust the attention dependency before running the full model.

## Checkpoint

Pre-trained weights are not stored in Git. Download the released checkpoint and
place it at the default path:

```bash
mkdir -p pretrain_checkpoint
huggingface-cli download OneMore1/Omni-fMRI checkpoint.pth \
  --local-dir pretrain_checkpoint
```

The expected default checkpoint path is:

```text
pretrain_checkpoint/checkpoint.pth
```

## Quick Smoke Test

Run the synthetic end-to-end check before using real data. It creates a small
synthetic NIfTI file, preprocesses it to NPZ, extracts encoder tokens, and
validates the output arrays.

```bash
python quickstart/quickstart_smoke.py \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --work-dir outputs/quickstart_smoke \
  --device cuda:0
```

Expected terminal summary:

```text
cls_token (768,) float32
patch_tokens (880, 768) float32
patch_coords (880, 3) int64
QUICKSTART_SMOKE_PASS outputs/quickstart_smoke
```

Generated files:

```text
outputs/quickstart_smoke/
  raw/sample_fmri.nii.gz
  processed_npz/sample_fmri_seg000.npz
  features/sample_fmri_seg000_tokens.npz
```

## Input Data Format

The model expects 4D fMRI arrays with spatial size `96 x 96 x 96` and temporal
length `40` by default.

For raw NIfTI files, preprocess first:

```bash
python data_preparation/preprocessing.py \
  --input_dir /path/to/raw_nifti \
  --output_dir outputs/my_npz \
  --pattern .nii.gz \
  --target_shape 96 96 96 \
  --segment_length 40
```

If data is already in NPZ format, each `.npz` should contain one 4D array. The
feature extraction script accepts either `DHWT` or `CDHW` layout:

- `DHWT`: depth, height, width, time
- `CDHW`: channel/time, depth, height, width

## Feature Extraction

Run the frozen Omni-fMRI encoder on a single `.npz` file or a directory of
`.npz` files:

```bash
python extract_feat.py outputs/my_npz \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --output-dir outputs/my_tokens \
  --device cuda:0 \
  --overwrite
```

Useful options:

- `--npz-key arr`: select a specific array key inside each NPZ file.
- `--layout dhwt`: force input layout when auto-detection is not desired.
- `--layout cdhw`: use channel-first 4D arrays.
- `--pad-short`: zero-pad short `DHWT` samples instead of failing.
- `--no-recursive`: scan only the top-level input directory.

Each output file is named `*_tokens.npz` and contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `cls_token` | `(768,)` | Global sample representation |
| `patch_tokens` | `(num_patches, 768)` | Dynamic patch-level token embeddings |
| `patch_coords` | `(num_patches, 3)` | Patch coordinates in voxel space |

## Pre-training

Directory mode expects split directories named with dataset and suffix, such as:

```text
data_root/
  HCP_train_40/
  HCP_val_40/
  ABIDE_train_40/
  ABIDE_val_40/
```

Run distributed pre-training:

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

TXT mode uses one path per line and does not require `--datasets`:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 \
bash scripts/pretrain.sh \
  --output_dir outputs/pretrain \
  --data_root /path/to/data_root \
  --data_mode txt \
  --train_txt /path/to/train.txt \
  --val_txt /path/to/val.txt \
  --batch_size 8 \
  --epochs 400
```

Default pre-training values are defined in `configs/pretrain.yaml`.

## Fine-tuning

The task CSV must contain:

- `Subject`: subject identifier column.
- The selected target column, for example `age` or `gender`.

Classification example:

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

Regression example:

```bash
bash scripts/finetune.sh \
  --output_dir outputs/finetune_age \
  --data_root /path/to/data_root \
  --data_mode txt \
  --train_txt /path/to/train.txt \
  --val_txt /path/to/val.txt \
  --test_txt /path/to/test.txt \
  --task_csv /path/to/labels.csv \
  --target_col age \
  --subject_id_regex '(\\d{7})' \
  --task_type regression \
  --num_classes 1
```

Default fine-tuning values are defined in `configs/finetune.yaml`.
Override the checkpoint with:

```bash
bash scripts/finetune.sh --pretrained_checkpoint /path/to/checkpoint.pth
```

## Docker

Pull the published image:

```bash
docker pull onemore1/onmi-fmri:py3.11-pytorch2.4.1-cuda12.4-cudnn9
```

Run with mounted code, data, and output directories:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -v /path/to/data_root:/data \
  -v /path/to/outputs:/outputs \
  -w /workspace \
  onemore1/onmi-fmri:py3.11-pytorch2.4.1-cuda12.4-cudnn9 \
  bash
```

Run the Docker smoke test:

```bash
docker run --rm \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  onemore1/onmi-fmri:py3.11-pytorch2.4.1-cuda12.4-cudnn9 \
  python scripts/docker_smoke_test.py --work-dir /tmp/omnifmri-smoke
```

## Development Notes

Use a feature branch for local modifications. The current local working branch
is intended to be separate from upstream `main`.

Recommended checks after code changes:

```bash
python -m compileall src pretrain.py finetune.py extract_feat.py
python quickstart/quickstart_smoke.py \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --work-dir outputs/quickstart_smoke \
  --device cuda:0
```

Do not commit real subject data, generated checkpoints, or large outputs.

## Citation

```bibtex
@article{wang2026omni,
  title={Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model},
  author={Wang, Mo and Ye, Wenhao and Xia, Junfeng and Zhang, Junxiang and Pan, Xuanye and Xu, Minghao and Deng, Haotian and Wen, Hongkai and Liu, Quanying},
  journal={arXiv preprint arXiv:2601.23090},
  year={2026}
}
```
