#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
npz_to_nii_strict.py

将 <root_in>/<subject>/*.npz|*.npy 转为 <root_out>/<subject>/*.nii.gz
- 不改数据内容
- 不改 header 字段（不 set_zooms、不 set_xyzt_units）
- 仅复制 template NIfTI 的 header 和 affine
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import nibabel as nib


def load_array(p: Path, npz_key: Optional[str]) -> np.ndarray:
    if p.suffix == ".npz":
        with np.load(p, allow_pickle=False) as z:
            if npz_key and npz_key in z.files:
                arr = z[npz_key]
            else:
                arr = z[z.files[0]]  # 取第一个数组
    elif p.suffix == ".npy":
        arr = np.load(p, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported file type: {p}")
    arr = np.asarray(arr)
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expect 3D/4D volume, got shape {arr.shape} at {p}")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_in", required=True, type=str, help="输入根目录（被试子目录下包含 .npz/.npy）")
    ap.add_argument("--root_out", required=True, type=str, help="输出根目录（将复刻层级写出 .nii.gz）")
    ap.add_argument("--template_nii", required=True, type=str, help="用于复制 header/affine 的模板 NIfTI")
    ap.add_argument("--npz_key", type=str, default="data", help="当 .npz 时的主键名（默认 data）")
    ap.add_argument("--pattern", type=str, default="", help="仅处理匹配的文件（如 'frames_*.npz'）")
    ap.add_argument("--skip_exists", action="store_true", help="若目标存在则跳过")
    ap.add_argument("--dry_run", action="store_true", help="只打印不写文件")
    args = ap.parse_args()

    root_in  = Path(args.root_in).resolve()
    root_out = Path(args.root_out).resolve()
    tmpl     = Path(args.template_nii).resolve()

    assert root_in.is_dir(), f"root_in not found: {root_in}"
    assert tmpl.is_file(),   f"template_nii not found: {tmpl}"
    root_out.mkdir(parents=True, exist_ok=True)

    template_img = nib.load(str(tmpl))
    template_hdr = template_img.header.copy()
    template_aff = template_img.affine.copy()

    subjects = [p for p in sorted(root_in.iterdir()) if p.is_dir()]
    total = 0
    for subj in subjects:
        files = sorted(subj.glob(args.pattern)) if args.pattern else \
                sorted(list(subj.glob("*.npz")) + list(subj.glob("*.npy")))
        if not files:
            print(f"[INFO] no npz/npy under: {subj}")
            continue

        for f in files:
            rel = f.relative_to(root_in)
            out_dir = (root_out / rel.parent)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f.with_suffix("").name + ".nii.gz"
            out_path = out_dir / out_name

            if args.skip_exists and out_path.exists():
                print(f"[SKIP][EXIST] {out_path}")
                continue

            try:
                vol = load_array(f, args.npz_key)   # 不改 dtype/数据
            except Exception as e:
                print(f"[ERR] load {f}: {e}")
                continue

            print(f"[MAKE] {rel} -> {out_path.name}  shape={tuple(vol.shape)} dtype={vol.dtype}")
            if args.dry_run:
                total += 1
                continue

            # 只复制 template 的 header & affine；不额外设置 zooms/units
            img = nib.Nifti1Image(vol, template_aff, header=template_hdr.copy())

            try:
                # 同步 qform/sform 与 code（复制，不计算）
                img.set_qform(template_img.get_qform(), code=int(template_hdr.get("qform_code", 1)))
                img.set_sform(template_img.get_sform(), code=int(template_hdr.get("sform_code", 1)))
            except Exception:
                pass

            try:
                nib.save(img, str(out_path))
            except Exception as e:
                print(f"[ERR] save {out_path}: {e}")
                continue

            total += 1

    print(f"[DONE] wrote {total} files to {root_out}")


if __name__ == "__main__":
    main()
