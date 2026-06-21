<div align="center">

# Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2601.23090-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.23090)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/OneMore1/Omni-fMRI)
[![Project Page](https://img.shields.io/badge/Project-Page-168b88?style=flat-square)](https://onemore1.github.io/Omni-fMRI/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/OneMore1/Omni-fMRI)
[![Docker Pulls](https://img.shields.io/docker/pulls/onemore1/onmi-fmri?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/onemore1/onmi-fmri)

</div>

[Paper](https://arxiv.org/abs/2601.23090) |
[Project Page](https://onemore1.github.io/Omni-fMRI/) |
[Checkpoint](https://huggingface.co/OneMore1/Omni-fMRI) |
[Quick Start Notebook](notebooks/01_quick_start_feature_extraction.ipynb) |
[Docker](https://hub.docker.com/r/onemore1/onmi-fmri) |
[Citation](#citation)

[ICML 2026] Official implementation of Omni-fMRI, a universal atlas-free fMRI foundation model with dynamic patching to reduce compute while preserving informative spatial structure.

<p align="center">
  <img src="pipeline.png" width="800" alt="framework">
</p>

</div>

## Quick Start

The fastest way to check the release is to run a synthetic, non-subject demo
from raw NIfTI creation through preprocessing and feature extraction. No real
fMRI data or checkpoint files are stored in GitHub; the checkpoint is hosted on
Hugging Face.

### 1. Install

```bash
git clone https://github.com/OneMore1/Omni-fMRI.git
cd Omni-fMRI
conda create -n omnifmri python=3.11
conda activate omnifmri
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -U huggingface_hub
```

### 2. Download the checkpoint

Pre-trained weights are available at
[Hugging Face](https://huggingface.co/OneMore1/Omni-fMRI). Put the released
checkpoint at `pretrain_checkpoint/checkpoint.pth`; feature extraction uses
this path by default.

```bash
mkdir -p pretrain_checkpoint
huggingface-cli download OneMore1/Omni-fMRI checkpoint.pth \
  --local-dir pretrain_checkpoint
```

### 3. Run the 10-minute smoke test

This script creates a synthetic `96 x 96 x 96 x 40` NIfTI file, converts it to
NPZ with the official preprocessing script, extracts features with
`extract_feat.py`, and verifies the saved arrays.

```bash
python quickstart/quickstart_smoke.py \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --work-dir outputs/quickstart_smoke \
  --device cuda:0
```

Expected output for the synthetic demo:

```text
cls_token (768,) float32
patch_tokens (880, 768) float32
patch_coords (880, 3) int64
QUICKSTART_SMOKE_PASS outputs/quickstart_smoke
```

The generated files are written under:

```text
outputs/quickstart_smoke/
  raw/sample_fmri.nii.gz
  processed_npz/sample_fmri_seg000.npz
  features/sample_fmri_seg000_tokens.npz
```

### 4. Extract features from your own fMRI data

If you have raw NIfTI files, first convert them to the NPZ format used by
extraction and training:

```bash
python data_preparation/preprocessing.py \
  --input_dir /path/to/raw_nifti \
  --output_dir outputs/my_npz \
  --pattern .nii.gz \
  --target_shape 96 96 96 \
  --segment_length 40
```

Then run the frozen Omni-fMRI encoder:

```bash
python extract_feat.py outputs/my_npz \
  --checkpoint pretrain_checkpoint/checkpoint.pth \
  --output-dir outputs/my_tokens \
  --device cuda:0 \
  --overwrite
```

If you already have `.npz` files shaped `(96, 96, 96, 40)`, you can skip
preprocessing and pass the NPZ file or folder directly to `extract_feat.py`.
Useful extraction options include `--npz-key arr`, `--layout dhwt`,
`--layout cdhw`, `--pad-short`, `--overwrite`, and `--no-recursive`.

Each output token file contains:

| Array | Meaning |
| --- | --- |
| `cls_token` | Global subject/session representation. |
| `patch_tokens` | Dynamic patch-level token embeddings. |
| `patch_coords` | Coordinates for the extracted dynamic patches. |

For an executable walkthrough, see
[`notebooks/01_quick_start_feature_extraction.ipynb`](notebooks/01_quick_start_feature_extraction.ipynb).

### Data Format

Directory mode expects folders named `{DATASET}_{SPLIT}`:

```text
data_root/
  HCP_train_40/
  HCP_val_40/
  ABIDE_train/
  ABIDE_val/
  ABIDE_test/
```

Pre-training defaults to `train_40/val_40`; downstream defaults to `train/val/test`.

TXT mode expects one path per line. Each line can be an `.npz` file or a directory containing `.npz` files. Relative paths are resolved under `--data_root`.

### Pre-training

Directory mode:

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

TXT mode does not use `--datasets`:

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

### Downstream Fine-tuning

`--task_csv` must contain a `Subject` column and the selected `--target_col`.

Directory mode:

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

TXT mode does not use `--datasets`:

```bash
bash scripts/finetune.sh \
  --output_dir outputs/finetune \
  --data_root /path/to/data_root \
  --data_mode txt \
  --train_txt /path/to/train.txt \
  --val_txt /path/to/val.txt \
  --test_txt /path/to/test.txt \
  --task_csv /path/to/labels.csv \
  --target_col gender \
  --subject_id_regex '(\\d{7})' \
  --task_type classification \
  --num_classes 2
```

For regression, use `--task_type regression --num_classes 1 --target_col age`. Label mean and standard deviation are computed automatically from the training split.

Fine-tuning loads `pretrain_checkpoint/checkpoint_epoch_32.pth` by default. Use `--pretrained_checkpoint /path/to/checkpoint.pth` to override it.

All training entry points are CLI-first. Omitted arguments fall back to defaults in `configs/*.yaml`. For uncommon options, use dotted overrides:

```bash
python pretrain.py --add-arg model.thresholds='[0.23]' training.warmup_epochs=5
python finetune.py --add-arg training.freeze_encoder=true data.batch_size=8
```

### Docker

Pull the published image:

```bash
docker pull onemore1/onmi-fmri:py3.11-pytorch2.4.1-cuda12.4-cudnn9
```

Run it with GPU, mounted code, data, and outputs:

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

Inside the container, run commands with paths mounted above, for example:

```bash
bash scripts/finetune.sh \
  --data_root /data \
  --task_csv /data/labels.csv \
  --target_col gender \
  --subject_id_regex '(\\d{7})' \
  --output_dir /outputs/finetune
```

Before real training, run the lightweight pipeline check:

```bash
docker run --rm \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  onemore1/onmi-fmri:py3.11-pytorch2.4.1-cuda12.4-cudnn9 \
  python scripts/docker_smoke_test.py --work-dir /tmp/omnifmri-smoke
```

To rebuild the image locally instead of pulling Docker Hub:

```bash
docker build -t omnifmri:local .
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -v /path/to/data_root:/data \
  -v /path/to/outputs:/outputs \
  -w /workspace \
  omnifmri:local \
  bash
```

## Repository Layout

```text
configs/                  # Default YAML values
data_preparation/          # NIfTI to NPZ preprocessing
docs/                      # Static project page for GitHub Pages
examples/                  # Notes about synthetic demo data
notebooks/                 # Quick Start notebooks
pretrain_checkpoint/       # Default location for released checkpoints
quickstart/                # End-to-end smoke test scripts
scripts/                   # Launchers and smoke test
src/data/                  # Pre-training and downstream datasets
src/models/                # MAE and ViT modules
src/utils/                 # CLI, logging, optimizer helpers
extract_feat.py            # Backbone feature extraction
pretrain.py                # MAE pre-training
finetune.py                # Downstream fine-tuning
```

## Citation

```bibtex
@article{wang2026omni,
  title={Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model},
  author={Wang, Mo and Ye, Wenhao and Xia, Junfeng and Zhang, Junxiang and Pan, Xuanye and Xu, Minghao and Deng, Haotian and Wen, Hongkai and Liu, Quanying},
  journal={arXiv preprint arXiv:2601.23090},
  year={2026}
}
```
