"""
Neuroimaging Data Preprocessing Pipeline
========================================

Processes NIfTI format brain imaging data for machine learning applications.

Pipeline Steps:
1. Recursively scan for NIfTI files (.nii/.nii.gz)
2. Normalize spatial dimensions to target cube size (default: 96³)
3. Preserve world coordinates via affine matrix adjustment
4. Apply Z-score normalization
5. Split 4D time series into fixed-length segments
6. Export as compressed NPZ format

Example Usage:
    python data_preparation/preprocessing.py \
        --input_dir /data/raw \
        --output_dir /data/processed \
        --target_shape 96 96 96 \
        --segment_length 40

Requirements:
    - nibabel >= 3.0
    - numpy >= 1.20
    - tqdm >= 4.60
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import nibabel as nib
from nibabel.affines import apply_affine
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def find_nifti_files(
    root_dir: str,
    pattern: str = ".nii.gz",
    case_insensitive: bool = False
) -> List[Path]:
    """
    Recursively find NIfTI files matching pattern.

    Args:
        root_dir: Root directory to search
        pattern: File suffix to match (e.g., ".nii", ".nii.gz", "_mc.nii.gz")
        case_insensitive: Whether to ignore case in matching

    Returns:
        List of Path objects sorted by directory depth and name
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Invalid directory: {root_dir}")

    hits = []
    target = pattern.lower() if case_insensitive else pattern

    for path in root.rglob("*"):
        if path.is_file():
            name = path.name.lower() if case_insensitive else path.name
            if name.endswith(target) or (case_insensitive and target in name):
                hits.append(path.resolve())

    # Sort by directory depth, then path length, then name
    hits.sort(key=lambda p: (len(p.parts), len(str(p)), str(p)))
    return hits


def normalize_spatial_dimensions(
    img: nib.Nifti1Image,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
    fill_value: float = 0.0
) -> Tuple[nib.Nifti1Image, Dict[str, Any]]:
    """
    Normalize image to target spatial dimensions while preserving world coordinates.

    Uses symmetric padding/cropping to maintain brain centering. Adjusts affine
    matrix to preserve spatial mapping: t_new = t + M @ crop_offset - M @ pad_offset

    Args:
        img: Input NIfTI image (3D or 4D)
        target_shape: Target spatial dimensions (x, y, z)
        fill_value: Value for padding

    Returns:
        Tuple of (normalized_image, metadata_dict)
    """
    data = np.asarray(img.dataobj)
    affine = img.affine.copy()
    rotation_scale = affine[:3, :3].copy()
    translation = affine[:3, 3].copy()

    # Handle dimensionality
    if data.ndim == 3:
        x, y, z = data.shape
        t = 1
        data = data[..., np.newaxis]
    elif data.ndim == 4:
        x, y, z, t = data.shape
    else:
        raise ValueError(f"Expected 3D or 4D input, got shape {data.shape}")

    tx, ty, tz = target_shape

    def compute_transform(old: int, new: int) -> Tuple[int, int, str]:
        """Compute symmetric padding/cropping for single dimension."""
        if old == new:
            return 0, 0, 'identity'
        elif old < new:
            total = new - old
            left = total // 2
            right = total - left
            return left, right, 'pad'
        else:
            total = old - new
            left = total // 2
            right = total - left
            return left, right, 'crop'

    # Compute transforms for each dimension
    px_l, px_r, mx = compute_transform(x, tx)
    py_l, py_r, my = compute_transform(y, ty)
    pz_l, pz_r, mz = compute_transform(z, tz)

    # Apply transformations
    d = data

    # Crop first (reverse order to maintain correct indexing)
    if mz == 'crop':
        d = d[:, :, pz_l:z - pz_r, :]
    if my == 'crop':
        d = d[:, py_l:y - py_r, :, :]
    if mx == 'crop':
        d = d[px_l:x - px_r, :, :, :]

    # Then pad
    pad_width = (
        (px_l if mx == 'pad' else 0, px_r if mx == 'pad' else 0),
        (py_l if my == 'pad' else 0, py_r if my == 'pad' else 0),
        (pz_l if mz == 'pad' else 0, pz_r if mz == 'pad' else 0),
        (0, 0)
    )

    if any(p != (0, 0) for p in pad_width):
        d = np.pad(d, pad_width=pad_width, mode='constant', constant_values=fill_value)

    # Update affine: adjust translation for padding/cropping
    pad_offset = np.array([
        px_l if mx == 'pad' else 0,
        py_l if my == 'pad' else 0,
        pz_l if mz == 'pad' else 0
    ], dtype=float)

    crop_offset = np.array([
        px_l if mx == 'crop' else 0,
        py_l if my == 'crop' else 0,
        pz_l if mz == 'crop' else 0
    ], dtype=float)

    new_translation = translation + rotation_scale @ crop_offset - rotation_scale @ pad_offset
    affine[:3, 3] = new_translation

    # Restore 3D if input was 3D
    if t == 1:
        d = d[..., 0]

    # Create output image with preserved header
    header = img.header.copy()
    qcode = int(header.get('qform_code', 1)) or 1
    scode = int(header.get('sform_code', 1)) or 1

    out_img = nib.Nifti1Image(d, affine, header=header)
    out_img.set_qform(affine, code=qcode)
    out_img.set_sform(affine, code=scode)

    metadata = {
        "original_shape": (x, y, z, t),
        "normalized_shape": d.shape if d.ndim == 4 else (*d.shape, 1),
        "transforms": {
            "x": {"left": px_l, "right": px_r, "mode": mx},
            "y": {"left": py_l, "right": py_r, "mode": my},
            "z": {"left": pz_l, "right": pz_r, "mode": mz},
        },
        "affine_translation": {
            "original": tuple(translation.tolist()),
            "new": tuple(new_translation.tolist())
        }
    }

    return out_img, metadata


def zscore_normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply Z-score normalization only to non-zero values: z = (x - μ) / (σ + eps)
    Background (zero values) remains zero.

    Args:
        data: Input array
        eps: Small constant to prevent division by zero

    Returns:
        Normalized array
    """
    mask = data != 0
    if not np.any(mask):
        return data

    values = data[mask]
    mean = float(values.mean())
    std = float(values.std())

    normalized = np.zeros_like(data)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized[mask] = (values - mean) / (std + eps)

    return normalized


def segment_timeseries(
    data: np.ndarray,
    segment_length: int = 40
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Split 4D time series into fixed-length segments.

    Args:
        data: 4D array with shape (x, y, z, time)
        segment_length: Number of timepoints per segment

    Returns:
        List of (start_idx, end_idx, segment_data) tuples
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D input, got {data.ndim}D")

    n_timepoints = data.shape[3]
    n_segments = n_timepoints // segment_length

    if n_segments == 0:
        logger.warning(f"Insufficient timepoints ({n_timepoints} < {segment_length})")
        return []

    segments = []
    for i in range(n_segments):
        start = i * segment_length
        end = (i + 1) * segment_length
        segment = data[..., start:end]
        segments.append((start, end, segment))

    return segments


def process_single_subject(
    nifti_path: Path,
    output_dir: Path,
    target_shape: Tuple[int, int, int] = (96, 96, 96),
    segment_length: int = 40
) -> bool:
    """
    Process a single subject's NIfTI file through the full pipeline.

    Args:
        nifti_path: Path to input NIfTI file
        output_dir: Directory for output files
        target_shape: Target spatial normalization dimensions
        segment_length: Timepoints per segment

    Returns:
        True if processing succeeded
    """
    try:
        logger.info(f"Processing: {nifti_path.name}")

        # Load image
        img = nib.load(str(nifti_path))
        header = img.header
        tr = float(header.get_zooms()[3]) if len(header.get_zooms()) > 3 else 1.0

        # Spatial normalization
        norm_img, metadata = normalize_spatial_dimensions(
            img, target_shape=target_shape
        )
        data = np.asarray(norm_img.dataobj, dtype=np.float32)

        # Z-score normalization
        normalized_data = zscore_normalize(data)

        # Quality metrics
        zero_ratio = (normalized_data == 0).sum() / normalized_data.size
        if normalized_data.ndim == 4:
            zero_voxel_ratio = (normalized_data == 0).all(axis=3).mean()
        else:
            zero_voxel_ratio = 0.0

        logger.info(f"  Zero element ratio: {zero_ratio:.2%}")
        logger.info(f"  Zero voxel ratio: {zero_voxel_ratio:.2%}")

        # Temporal segmentation
        if normalized_data.ndim == 3:
            logger.info(
                f"  3D data detected, expanding to pseudo-timeseries length {segment_length}"
            )
            expanded = np.repeat(
                normalized_data[..., np.newaxis],
                repeats=segment_length,
                axis=3
            )
            segments = [(0, segment_length, expanded)]
        else:
            segments = segment_timeseries(normalized_data, segment_length)
            if not segments:
                logger.warning("  No segments generated, skipping")
                return False

        # Generate output filename base
        if nifti_path.name.endswith(".nii.gz"):
            stem = nifti_path.name[:-7]
        elif nifti_path.name.endswith(".nii"):
            stem = nifti_path.name[:-4]
        else:
            stem = nifti_path.stem

        # Save segments
        for idx, (start, end, segment) in enumerate(segments):
            out_name = f"{stem}_seg{idx:03d}.npz"
            out_path = output_dir / out_name

            np.savez_compressed(
                out_path,
                data=segment.astype(np.float32),
                tr=float(tr),
                affine=norm_img.affine.astype(np.float32),
                segment_index=idx,
                timepoints=(start, end),
                subject_id=stem,
                metadata=metadata
            )
            logger.info(f"  Saved: {out_name} (shape: {segment.shape})")

        return True

    except Exception as e:
        logger.error(f"  Failed to process {nifti_path.name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess neuroimaging data for machine learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s -i /data/raw -o /data/processed

  # Custom target shape and segment length
  %(prog)s -i /data/raw -o /data/processed --target_shape 128 128 128 -s 50

  # Process specific file pattern
  %(prog)s -i /data/raw -o /data/processed --pattern "_preproc.nii.gz"
        """
    )

    parser.add_argument(
        "-i", "--input_dir",
        required=True,
        help="Root directory containing NIfTI files"
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Output directory for processed NPZ files"
    )
    parser.add_argument(
        "-p", "--pattern",
        default=".nii.gz",
        help="File pattern to match (default: .nii.gz)"
    )
    parser.add_argument(
        "--target_shape",
        nargs=3,
        type=int,
        default=[96, 96, 96],
        metavar=("X", "Y", "Z"),
        help="Target spatial dimensions (default: 96 96 96)"
    )
    parser.add_argument(
        "-s", "--segment_length",
        type=int,
        default=40,
        help="Number of timepoints per segment (default: 40)"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO)"
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Find files
    try:
        files = find_nifti_files(input_dir, pattern=args.pattern)
        logger.info(f"Found {len(files)} files matching pattern '{args.pattern}'")
    except Exception as e:
        logger.error(f"Failed to scan directory: {e}")
        sys.exit(1)

    if not files:
        logger.warning("No files found to process")
        sys.exit(0)

    # Process files
    success_count = 0
    fail_count = 0

    for filepath in tqdm(files, desc="Processing subjects"):
        success = process_single_subject(
            filepath,
            output_dir,
            target_shape=tuple(args.target_shape),
            segment_length=args.segment_length
        )
        if success:
            success_count += 1
        else:
            fail_count += 1

    # Summary
    logger.info("=" * 50)
    logger.info("Processing complete")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Output directory: {output_dir}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
