<div align="center">

# Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2601.23090-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.23090)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/OneMore1/Omni-fMRI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/OneMore1/Omni-fMRI)
[![Docker Pulls](https://img.shields.io/docker/pulls/onemore1/onmi-fmri?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/onemore1/onmi-fmri)

</div>

Official implementation of Omni-fMRI, a universal atlas-free fMRI foundation model with dynamic patching to reduce compute while preserving informative spatial structure.

<p align="center">
  <img src="pipeline.png" width="800" alt="framework">
</p>

</div>

## Installation

The repository targets Python 3.11 and PyTorch 2.4.1 with CUDA 12.4:

```bash
conda create -n omnifmri python=3.11
conda activate omnifmri

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Project Structure

```text
configs/
  finetune.yaml          # Downstream fine-tuning config
  pretrain.yaml          # Pre-training config
data_preparation/
  preprocessing.py       # NIfTI to segmented NPZ preprocessing pipeline
  MNI152_T1_1mm_brain_mask.nii.gz
scripts/
  docker_smoke_test.py   # Dummy-data Docker preflight
  finetune.sh            # Torchrun launcher for downstream tasks
  pretrain.sh            # Torchrun launcher for pre-training
src/
  data/                  # Dataset loaders
  models/                # Model architecture
  utils/                 # Config, logging, optimization, distributed helpers
extract_feat.py          # Extract CLS and patch tokens from NPZ inputs
finetune.py              # Main fine-tuning entry point
pretrain.py              # Main pre-training entry point
heatmap_visualize.py     # Heatmap visualization
visual_3d.py             # 3D visualization
```

## Data Preparation

Use [data_preparation/preprocessing.py](data_preparation/preprocessing.py) to convert raw NIfTI files into the segmented NPZ format used by training and feature extraction.

The script does the following:

- recursively scans an input directory for `.nii` or `.nii.gz` files
- symmetrically pads or crops spatial dimensions to `(96, 96, 96)` by default
- preserves world coordinates by updating the affine after padding or cropping
- applies Z-score normalization to non-zero voxels
- splits 4D time series into fixed-length segments, `40` frames by default
- writes compressed `.npz` outputs with the segment data, TR, affine, and metadata

Basic usage:

```bash
python data_preparation/preprocessing.py \
  --input_dir /path/to/raw_nifti_dir \
  --output_dir /path/to/processed_npz_dir
```

Common options:

```bash
--pattern .nii.gz            # Match input filenames by suffix
--target_shape 96 96 96      # Target spatial size
--segment_length 40          # Frames per NPZ segment
--log_level INFO             # DEBUG, INFO, WARNING, ERROR
```

If your raw file names end with a custom suffix, for example `_preproc.nii.gz`, use:

```bash
python data_preparation/preprocessing.py \
  --input_dir /path/to/raw_nifti_dir \
  --output_dir /path/to/processed_npz_dir \
  --pattern _preproc.nii.gz
```

Each output `.npz` contains:

```text
data           # Segment array, shape (96, 96, 96, 40) by default
tr             # Repetition time from the source header
affine         # Updated affine after spatial normalization
segment_index  # Zero-based segment id for the source file
timepoints     # [start, end) frame range from the source timeseries
subject_id     # Stem of the source NIfTI filename
metadata       # Original shape and padding/cropping bookkeeping
```

For pretraining or finetuning, the script does not create train/val/test splits by itself. Run it separately for each split and write into the dataset folders expected by the loaders, for example:

```text
data_root/
  ABIDE_train_40/
  ABIDE_val_40/
  HCP_train_40/
  HCP_val_40/
```

Example:

```bash
python data_preparation/preprocessing.py \
  --input_dir /raw/HCP_train \
  --output_dir /data/HCP_train_40 \
  --segment_length 40
```

The training code expects NPZ inputs with spatial size `(96, 96, 96)` and 40-frame segments unless you intentionally change the model and config settings.

## Extract Backbone Features

`extract_feat.py` loads a pre-trained checkpoint, runs each NPZ sample through the encoder backbone, and writes one output NPZ per sample.

For a folder of NPZ files:

```bash
python extract_feat.py \
  /path/to/input_npz_or_folder \
  --checkpoint /path/to/checkpoint.pth \
  --output-dir /path/to/output_tokens
```

For a single NPZ file:

```bash
python extract_feat.py \
  /path/to/sample.npz \
  --checkpoint /path/to/checkpoint.pth \
  --output-dir /path/to/output_tokens
```

Useful options:

```bash
--checkpoint /path/to/checkpoint.pth  # Override checkpoint path
--npz-key arr                         # Array key inside NPZ; default is the first key
--layout dhwt                         # Input layout: dhwt, cdhw, or auto
--start-frame 0                       # Start frame when DHWT has more than 40 frames
--pad-short                           # Zero-pad samples shorter than 40 frames
--overwrite                           # Overwrite existing output NPZ files
--no-recursive                        # Do not recursively scan input directories
```

Input assumptions:

- Default input layout is `(D, H, W, T)`, with spatial shape `(96, 96, 96)`.
- The model expects 40 temporal frames.
- `--layout cdhw` can be used when the input is already `(40, 96, 96, 96)`.

Each output file is named like `<input_stem>_tokens.npz` and contains:

```text
cls_token     # Shape: (768,)
patch_tokens  # Shape: (num_patches, 768)
patch_coords  # Shape: (num_patches, 3), top-left voxel coords in (z, y, x)
```

## Training

### Pre-training

Expected directory layout:

```text
data_root/
  ABIDE_train_40/
  ABIDE_val_40/
  HCP_train_40/
  HCP_val_40/
    0010001/
      0010001_run-1_0000-0199_1.npz
      0010001_run-1_0000-0199_2.npz
```

Update `configs/pretrain.yaml`:

```yaml
data:
  data_root: /path/to/data_root
  datasets: ["HCP", "ABIDE"]
```

Start pre-training:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 CONFIG_FILE=configs/pretrain.yaml OUTPUT_DIR=outputs/pretrain \
  bash scripts/pretrain.sh --cfg-options model.thresholds='[0.23]' training.warmup_epochs=5
```

### Downstream Evaluation

Directory mode:

```yaml
task:
  csv: /path/to/data_csv

data:
  data_root: /path/to/data_root
  datasets: ["HCP"]
  mode: "directory"
```

TXT mode:

```yaml
task:
  csv: /path/to/data_csv

data:
  train_txt: /path/to/train_txt
  val_txt: /path/to/val_txt
  test_txt: /path/to/test_txt
  mode: "txt"
```

Start downstream training:

```bash
bash scripts/finetune.sh \
  --pretrained_checkpoint /path/to/pretrain_checkpoint.pth \
  --data_root /path/to/data_root \
  --task_csv /path/to/data_csv \
  --data_mode directory \
  --task_type classification \
  --num_classes 2 \
  --batch_size 16
```

## Docker

### Build The Image

Run from the repository root:

```bash
docker build -t omnifmri:local .
```

### Start A Bash Shell

Linux or macOS:

```bash
docker run --rm -it \
  --ipc=host \
  -v "$(pwd):/workspace" \
  -w /workspace \
  omnifmri:local \
  bash
```

PowerShell:

```powershell
docker run --rm -it `
  --ipc=host `
  -v "${PWD}:/workspace" `
  -w /workspace `
  omnifmri:local `
  bash
```

If you want to mount real data and outputs for training, add:

```text
-v /path/to/data_root:/data
-v /path/to/outputs:/outputs
```

and then point configs to `/data` and `/outputs` inside the container.

### GPU Training In Docker

When you are ready to train on GPU, add `--gpus all`:

```bash
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

Inside the container, use `bash`, not `sh`, because the launcher scripts rely on Bash arrays:

```bash
bash scripts/pretrain.sh
# or
bash scripts/finetune.sh
```

### Docker Smoke Test

The repository includes a non-training preflight that creates dummy data and validates:

- config loading
- model construction
- dataset discovery
- dataloader output shapes

Run it inside the container:

```bash
python scripts/docker_smoke_test.py --work-dir /tmp/omnifmri-smoke
```

This does not launch real training and does not require real data. It is intended to verify that the Docker runtime and the basic pipeline wiring are healthy before you run on GPUs.

### Published Image

```bash
docker pull onemore1/onmi-fmri:py3.11-pytorch2.4.1-cuda12.4-cudnn9
```

The published image uses the same runtime layout. You can launch it with the same `docker run ... bash` commands above.

At the time of writing, `onemore1/onmi-fmri:latest` is not available, so prefer the explicit tag above.

## Model Checkpoints

Pre-trained weights are available on Hugging Face:

https://huggingface.co/OneMore1/Omni-fMRI

## Docker Image

Docker Hub:

https://hub.docker.com/r/onemore1/onmi-fmri

## Citation

```bibtex
@article{wang2026omni,
  title={Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model},
  author={Wang, Mo and Ye, Wenhao and Xia, Junfeng and Zhang, Junxiang and Pan, Xuanye and Xu, Minghao and Deng, Haotian and Wen, Hongkai and Liu, Quanying},
  journal={arXiv preprint arXiv:2601.23090},
  year={2026}
}
```
