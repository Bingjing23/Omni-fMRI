#!/usr/bin/env python3
"""Split 200-frame NPZ volumes into five 40-frame chunks.

Usage
-----
python npz_200_to_40.py \
    --src /path/to/npz_dir \
    --dst /path/to/output_dir

Each input file like ``0010001_run-1_0000-0199.npz`` is expected to contain a
4D array of shape (..., 200). The script slices along the last axis into five
non-overlapping 40-frame segments and saves them as
``0010001_run-1_0000-0199_1.npz`` ... ``_5.npz`` in ``--dst``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split 200-frame NPZ files into 5×40 frames")
    parser.add_argument("--src", required=True, help="Directory containing original 200-frame npz files")
    parser.add_argument("--dst", required=True, help="Directory to store 40-frame chunks")
    parser.add_argument(
        "--key",
        default=None,
        help="Array key inside npz (default: first key)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing chunk files")
    return parser.parse_args()


def load_array(npz_path: Path, key: Optional[str]) -> np.ndarray:
    with np.load(npz_path) as data:
        arr_key = key or data.files[0]
        if arr_key not in data:
            raise KeyError(f"Key '{arr_key}' not found in {npz_path}")
        array = data[arr_key]
    return array


def split_array(array: np.ndarray, segments: int = 5) -> list[np.ndarray]:
    frames = array.shape[-1]
    if frames % segments != 0:
        raise ValueError(f"Array last axis length {frames} not divisible by {segments}")
    chunk = frames // segments
    return [array[..., i * chunk : (i + 1) * chunk] for i in range(segments)]


def process_file(npz_path: Path, dst_dir: Path, key: Optional[str], overwrite: bool) -> list[Path]:
    array = load_array(npz_path, key)
    chunks = split_array(array, segments=5)
    written = []
    for idx, chunk in enumerate(chunks, 1):
        out_name = f"{npz_path.stem}_{idx}.npz"
        out_path = dst_dir / out_name
        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} exists (use --overwrite to replace)")
            continue
        np.savez_compressed(out_path, arr=chunk)
        written.append(out_path)
    return written


def main() -> None:
    args = parse_args()
    src_dir = Path(args.src).expanduser().resolve()
    dst_dir = Path(args.dst).expanduser().resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(src_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"No npz files found in {src_dir}")

    total_written = 0
    for npz_path in npz_files:
        try:
            written = process_file(npz_path, dst_dir, args.key, args.overwrite)
            total_written += len(written)
            print(f"[OK] {npz_path.name} -> {len(written)} chunks")
        except Exception as exc:
            print(f"[ERR] {npz_path.name}: {exc}")

    print(f"Finished. Wrote {total_written} chunk files to {dst_dir}")


if __name__ == "__main__":  
    main()
