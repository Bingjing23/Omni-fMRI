<div align="center">

# Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model

<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2601.23090-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2601.23090)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github)](https://github.com/OneMore1/Omni-fMRI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/OneMore1/Omni-fMRI)
[![Docker Pulls](https://img.shields.io/docker/pulls/onemore1/onmi-fmri?style=flat-square&logo=docker&logoColor=white)](https://hub.docker.com/r/onemore1/onmi-fmri)

</div>

This repository contains the official implementation of Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model, which introduces a dynamic patching mechanism that significantly reduces computational costs while preserving informative spatial structures.

<p align="center">
  <img src="pipeline.png" width="800" alt="framework">
</p>

</div>

## Installation

Setting up the environment requires Python 3.10 and CUDA-compatible PyTorch for GPU acceleration:

```
conda create -n omnifmri python=3.11

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```


## Project Structure

The codebase is organized into modular components for easy navigation and extension:

```
├── configs/                # Configuration files
│   ├── finetune.yaml       # Configuration for the fine-tuning phase
│   └── pretrain.yaml       # Configuration for the pre-training phase
├── scripts/                # Shell scripts for automated execution
│   ├── finetune.sh         
│   └── pretrain.sh         
├── src/                    # Source code core
│   ├── data/               # Data loadind
│   ├── models/             # Models Architecture 
│   └── utils/              # Helper functions (Logging, metrics, checkpoints)
├── finetune.py             # Main entry point for model fine-tuning
├── heatmap_visualize.py    # Visualization tool for generating heatmaps
├── extract_feat.py # Extract backbone CLS and patch tokens from NPZ inputs
├── pretrain.py             # Main entry point for model pre-training
└── visual_3d.py            # Visualization tool
```

## Data Preparation

### Preprocessing Pipeline

See  data_preparation.ipynb in data-preparation. The input fMRI data should be in the MNI coordinate system.

fMRI data were resampled with cubic spline interpolation to a $96\times96\times96$ grid at 2 mm isotropic resolution in MNI space. Time series with TR outside 0.7–0.8 s were voxel-wise resampled to 0.72 s with cubic splines, and signals were globally z-scored within the brain mask.

## Extract Backbone Feature

`extract_backbone_tokens.py` loads a pre-trained checkpoint, runs each NPZ sample through the encoder backbone, and saves one output NPZ per sample.


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
--checkpoint /path/to/checkpoint.pth  # Override checkpoint path
--npz-key arr                         # Array key inside NPZ; default is the first key
--layout dhwt                         # Input layout: dhwt, cdhw, or auto
--start-frame 0                       # Start frame when DHWT has more than 40 frames
--pad-short                           # Zero-pad samples shorter than 40 frames
--overwrite                           # Overwrite existing output NPZ files
--no-recursive                        # Do not recursively scan input directories
```

Input assumptions:

- Default input layout is `(D, H, W, T)`, with spatial shape `(96, 96, 96)`.
- The model uses 40 temporal frames. If an input has more than 40 frames, the script uses `--start-frame` to choose a contiguous 40-frame window.
- `--layout cdhw` can be used when the input is already `(40, 96, 96, 96)`.

Each output file is named like `<input_stem>_tokens.npz` and contains:

```text
cls_token     # Shape: (768,)
patch_tokens  # Shape: (num_patches, 768)
patch_coords  # Shape: (num_patches, 3), top-left voxel coords in (z, y, x)
```


## Training

### Pre-training

1. Ensure your pre-train data structure as follow:

```
data_root/
├── ABIDE_train/                
├── ABIDE_val/                  
├── HCP_val/              
└── HCP_train/              
    ├── 0010001/                # Subject ID
    └── 0010002/                
        ├── 0010002_run-1_0000-0199_1.npz  # Data chunk 1 
        ├── 0010002_run-1_0000-0199_2.npz  # Data chunk 2
```

2. Edit `configs/pretrain.yaml` and update the `data_root` and `datasets` 

```yaml
data:
  data_root: /path/to/data_root
  datasets: ["HCP", "ABIDE"]
```

3. Start pre-training from unlabeled fMRI data using multi-scale masked prediction tasks:

```bash
# running pretrain
sh scripts/pretrain.sh
```

### Downstream evaluation

We have provided serval downstream dataloader as follow:

1. Loading downstream datasets as pre-training data structure:

```yaml
task:
  csv: "/path/to/data_csv"

data:
  data_root: /path/to/data_root
  datasets: ["HCP"]
  mode: "directory"
```

2. Loading dowwnstream datasets with txt:

```yaml
task:
  csv: "/path/to/data_csv"

data:
  train_txt: /path/to/train_txt
  val_txt: /path/to/val_txt
  test_txt: /path/to/test_txt
  mode: "txt"
```

3. Loading downstream datasets with txt and directory mapping:

```yaml
data:
  train_txt: /path/to/train_txt
  val_txt: /path/to/val_txt
  test_txt: /path/to/test_txt
  mode: "txt_mapping"
```

Start downstream training:

```bash
# running downstream training
sh scripts/finetune.sh
```

### How to use Docker

#### Option A: Build Locally

Run from the Omni-fMRI repository root:

```bash
docker build -t omnifmri:local .
```

Start the container:

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  -v "$PWD":/workspace \
  -v /path/to/data_root:/data \
  -v /path/to/outputs:/outputs \
  omnifmri:local \
  bash
```

Inside the container, edit config paths to container paths, for example:

```yaml
data:
  data_root: /data
```

Then run:

```bash
sh scripts/pretrain.sh
# or
sh scripts/finetune.sh
```


#### Option B: Use The Published Image

```bash
docker pull onemore1/onmi-fmri:latest
```

Run from the Omni-fMRI repository root:

```bash
docker run --gpus all --rm -it \
  --ipc=host \
  -v "$PWD":/workspace \
  -v /path/to/data_root:/data \
  -v /path/to/outputs:/outputs \
  onemore1/onmi-fmri:latest \
  bash
```

#### Model Checkpoints

Our pre-trained model weights can be found in Huggingface.  https://huggingface.co/OneMore1/Omni-fMRI

#### Model Docker

Docker can be found in Dockerhub.   https://hub.docker.com/r/onemore1/onmi-fmri


#### Citation
Citations and discussions are welcome.

@article{wang2026omni,
  title={Omni-fMRI: A Universal Atlas-Free fMRI Foundation Model},
  author={Wang, Mo and Ye, Wenhao and Xia, Junfeng and Zhang, Junxiang and Pan, Xuanye and Xu, Minghao and Deng, Haotian and Wen, Hongkai and Liu, Quanying},
  journal={arXiv preprint arXiv:2601.23090},
  year={2026}
}




