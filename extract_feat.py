#!/usr/bin/env python3
"""Extract Omni-fMRI backbone tokens from NPZ samples.

The script loads a pre-trained checkpoint, runs each input sample through the
encoder backbone, and writes one NPZ per sample with:
  - cls_token: final-layer CLS token, shape (embed_dim,)
  - patch_tokens: final-layer patch tokens, shape (num_patches, embed_dim)
  - patch_coords: top-left patch coordinates in voxel space, shape (num_patches, 3)
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.models.mae_model import AdaptiveMAE
from src.models.patch_embed_3d import TokenizedZeroConvPatchAttn3D
from src.utils.cli_app import YamlBackedCliApp
from src.utils.config_overrides import add_pretrain_override_args, apply_pretrain_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CLS token, patch tokens, and patch coordinates from Omni-fMRI NPZ data."
    )
    parser.add_argument("input", type=Path, help="Input .npz file or a directory containing .npz files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where per-sample token NPZ files will be written",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrain_checkpoint/checkpoint.pth"),
        help="Checkpoint path (default: pretrain_checkpoint/checkpoint.pth)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pretrain.yaml"),
        help="Fallback config used only if the checkpoint does not contain one. CLI args override both.",
    )
    parser.add_argument("--npz-key", default=None, help="Array key inside NPZ (default: first key)")
    parser.add_argument(
        "--layout",
        choices=("auto", "dhwt", "cdhw"),
        default="auto",
        help="Input array layout. auto accepts DHWT or CDHW when one axis matches in_chans.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start index when a DHWT sample has more frames than model in_chans",
    )
    parser.add_argument(
        "--pad-short",
        action="store_true",
        help="Zero-pad DHWT samples shorter than model in_chans instead of raising an error",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively scan input directories for .npz files (default: true)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files that already exist",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0. Use CUDA_VISIBLE_DEVICES to bind a physical GPU.",
    )
    add_pretrain_override_args(parser)
    return parser


def load_checkpoint(checkpoint_path: Path) -> dict:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def as_3tuple(value: object) -> tuple[int, int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise ValueError(f"Expected a 3-item spatial tuple, got {value}")
        return tuple(int(x) for x in value)
    return (int(value), int(value), int(value))


def create_model(config: dict) -> nn.Module:
    model_config = config["model"]

    return AdaptiveMAE(
        img_size=as_3tuple(model_config["img_size"]),
        patch_size=as_3tuple(model_config["patch_size"]),
        in_chans=int(model_config["in_chans"]),
        embed_dim=int(model_config["embed_dim"]),
        depth=int(model_config["depth"]),
        qkv_bias=bool(model_config["qkv_bias"]),
        qk_norm=bool(model_config["qk_norm"]),
        num_heads=int(model_config["num_heads"]),
        decoder_embed_dim=int(model_config["decoder_embed_dim"]),
        drop_path_rate=float(model_config["drop_path_rate"]),
        decoder_depth=int(model_config["decoder_depth"]),
        decoder_num_heads=int(model_config["decoder_num_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        mask_ratio=float(model_config["mask_ratio"]),
        mixed_patch_embed=TokenizedZeroConvPatchAttn3D,
        patch_norm=bool(model_config["enable_patch_norm"]),
        gate_attention=model_config["gate_attention"],
    )


def state_dict_from_checkpoint(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return checkpoint


def create_backbone(checkpoint: dict, config: dict, device: torch.device) -> nn.Module:
    model = create_model(config)
    state_dict = state_dict_from_checkpoint(checkpoint)
    load_msg = model.load_state_dict(state_dict, strict=False)

    unexpected = len(load_msg.unexpected_keys)
    missing = len(load_msg.missing_keys)
    print(f"Loaded checkpoint weights with {missing} missing keys and {unexpected} unexpected keys.")
    if unexpected:
        print("Unexpected keys are ignored by strict=False loading.")

    if hasattr(model, "encoder"):
        backbone = model.encoder
    elif hasattr(model, "backbone"):
        backbone = model.backbone
    else:
        backbone = model

    backbone.to(device)
    backbone.eval()
    return backbone


def iter_npz_files(input_path: Path, recursive: bool) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Input file must be .npz: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*.npz" if recursive else "*.npz"
    files = sorted(input_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {input_path}")
    return files


def load_npz_array(npz_path: Path, key: Optional[str]) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        array_key = key or data.files[0]
        if array_key not in data.files:
            raise KeyError(f"Key '{array_key}' not found in {npz_path}; available keys: {data.files}")
        return np.asarray(data[array_key], dtype=np.float32)


def array_to_model_tensor(
    array: np.ndarray,
    *,
    in_chans: int,
    spatial_size: tuple[int, int, int],
    layout: str,
    start_frame: int,
    pad_short: bool,
) -> torch.Tensor:
    if array.ndim != 4:
        raise ValueError(f"Expected a 4D array, got shape {array.shape}")

    if layout == "auto":
        if array.shape[-1] >= in_chans:
            layout = "dhwt"
        elif array.shape[0] == in_chans:
            layout = "cdhw"
        else:
            raise ValueError(
                f"Cannot infer layout from shape {array.shape}; expected last axis or first axis to match {in_chans}"
            )

    if layout == "dhwt":
        if array.shape[:3] != spatial_size:
            raise ValueError(f"Expected spatial shape {spatial_size}, got {array.shape[:3]}")
        total_frames = array.shape[-1]
        if total_frames < in_chans:
            if not pad_short:
                raise ValueError(
                    f"Need at least {in_chans} frames, got {total_frames}. Use --pad-short to zero-pad."
                )
            padded = np.zeros((*array.shape[:3], in_chans), dtype=np.float32)
            padded[..., :total_frames] = array
            array = padded
        else:
            if start_frame < 0 or start_frame + in_chans > total_frames:
                raise ValueError(
                    f"--start-frame {start_frame} is invalid for {total_frames} frames and length {in_chans}"
                )
            array = array[..., start_frame : start_frame + in_chans]
        array = np.transpose(array, (3, 0, 1, 2))
    else:
        if array.shape[0] != in_chans:
            raise ValueError(f"Expected channel axis length {in_chans}, got {array.shape[0]}")
        if array.shape[1:] != spatial_size:
            raise ValueError(f"Expected spatial shape {spatial_size}, got {array.shape[1:]}")

    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def output_path_for(npz_path: Path, input_root: Optional[Path], output_dir: Path) -> Path:
    if input_root is not None:
        rel = npz_path.resolve().relative_to(input_root)
        return output_dir / rel.parent / f"{rel.stem}_tokens.npz"
    return output_dir / f"{npz_path.stem}_tokens.npz"


@torch.no_grad()
def extract_tokens(backbone: nn.Module, sample: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    input_dict = backbone.patch_tokenizer(sample)
    current_img_size = sample.shape[2:]

    tokens, _, _, _, _ = backbone.mixed_patch(
        sample,
        backbone.pos_embed,
        input_dict,
        current_img_size=current_img_size,
    )

    seqlens = torch.as_tensor(input_dict["seqlens"], device=sample.device, dtype=torch.long)
    arange = torch.arange(tokens.shape[1], device=sample.device).unsqueeze(0)
    valid_mask = arange < seqlens.unsqueeze(1)

    tokens_packed = tokens[valid_mask]
    cu_seqlens = torch.cat(
        [
            torch.zeros(1, device=sample.device, dtype=torch.int32),
            seqlens.cumsum(0, dtype=torch.int32),
        ]
    ).contiguous()
    max_seqlen = int(seqlens.max().item())

    layer_outputs = backbone.forward_features(
        tokens_packed,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    last_tokens = layer_outputs[-1]

    if seqlens.numel() != 1:
        raise RuntimeError("Internal error: this extractor expects one sample per forward pass.")

    seq_len = int(seqlens[0].item())
    sequence = last_tokens[:seq_len]
    cls_token = sequence[0].detach().float().cpu().numpy()
    patch_tokens = sequence[1:].detach().float().cpu().numpy()
    patch_coords = input_dict["patch_coords"][0, : seq_len - 1].detach().cpu().numpy().astype(np.int64)
    return cls_token, patch_tokens, patch_coords


class ExtractFeaturesApp(YamlBackedCliApp):
    default_config_path = Path("configs/pretrain.yaml")

    def build_parser(self) -> argparse.ArgumentParser:
        return build_parser()

    def configure(self) -> dict:
        assert self.args is not None
        checkpoint = load_checkpoint(self.args.checkpoint.expanduser().resolve())
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if config is None:
            config = self.load_base_config(self.args)
        apply_pretrain_overrides(config, self.args)
        return config

    def run(self) -> None:
        self.args = self.parse_args()
        config = self.configure()

        input_path = self.args.input.expanduser().resolve()
        output_dir = self.args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        device = torch.device(self.args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        checkpoint = load_checkpoint(self.args.checkpoint.expanduser().resolve())
        backbone = create_backbone(checkpoint, config, device)

        in_chans = int(backbone.patch_embed.proj.in_channels)
        spatial_size = tuple(int(x) for x in backbone.img_size)

        files = iter_npz_files(input_path, self.args.recursive)
        input_root = input_path if input_path.is_dir() else None

        print(f"Found {len(files)} NPZ file(s). Writing outputs to {output_dir}")
        for npz_path in tqdm(files, desc="Extracting tokens"):
            out_path = output_path_for(npz_path, input_root, output_dir)
            if out_path.exists() and not self.args.overwrite:
                continue

            try:
                array = load_npz_array(npz_path, self.args.npz_key)
                sample = array_to_model_tensor(
                    array,
                    in_chans=in_chans,
                    spatial_size=spatial_size,
                    layout=self.args.layout,
                    start_frame=self.args.start_frame,
                    pad_short=self.args.pad_short,
                ).to(device=device, non_blocking=True)

                cls_token, patch_tokens, patch_coords = extract_tokens(backbone, sample)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_path,
                    cls_token=cls_token,
                    patch_tokens=patch_tokens,
                    patch_coords=patch_coords,
                )
            except Exception as exc:
                print(f"[ERR] {npz_path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    ExtractFeaturesApp.main()
