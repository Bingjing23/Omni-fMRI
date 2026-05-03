<div align="center">

# Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2601.23090-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.23090)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/OneMore1/Omni-fMRI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/OneMore1/Omni-fMRI)
[![Docker Pulls](https://img.shields.io/docker/pulls/onemore1/onmi-fmri?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/onemore1/onmi-fmri)

</div>

[ICML 2026] Official implementation of Omni-fMRI, a universal atlas-free fMRI foundation model with dynamic patching to reduce compute while preserving informative spatial structure.

<p align="center">
  <img src="pipeline.png" width="800" alt="framework">
</p>

</div>

## Quick Start

### Installation

```bash
conda create -n omnifmri python=3.11
conda activate omnifmri
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Pre-trained weights are available at https://huggingface.co/OneMore1/Omni-fMRI. Put the released checkpoint at `pretrain_checkpoint/checkpoint.pth`; feature extraction and fine-tuning use this path by default.

### Feature Extraction

Use this when you want to apply the released Omni-fMRI checkpoint directly and export backbone tokens for downstream analysis.

```bash
python extract_feat.py \
  /path/to/input_npz_or_folder \
  --output-dir /path/to/output_tokens
```

Inputs are `.npz` files shaped `(96, 96, 96, 40)` by default. Useful options: `--checkpoint /path/to/checkpoint.pth`, `--npz-key arr`, `--layout dhwt`, `--layout cdhw`, `--pad-short`, `--overwrite`, `--no-recursive`.

Each output contains `cls_token`, `patch_tokens`, and `patch_coords`.

### Data Format

Raw NIfTI files can be converted to the NPZ format used by extraction and training:

```bash
python data_preparation/preprocessing.py \
  --input_dir /path/to/raw_nifti \
  --output_dir /path/to/processed_npz \
  --segment_length 40
```

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
pretrain_checkpoint/       # Default location for released checkpoints
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
