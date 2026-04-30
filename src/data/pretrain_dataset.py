import os
import glob
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


def global_zscore_nonzero(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score non-zero voxels while keeping zero-valued background at zero."""
    mask = data != 0
    if not np.any(mask):
        return data

    normalized = np.zeros_like(data, dtype=np.float32)
    foreground = data[mask].astype(np.float32, copy=False)
    mean = foreground.mean()
    std = foreground.std()
    if std < eps:
        return normalized

    normalized[mask] = (foreground - mean) / std
    return normalized


class fMRIDataset(Dataset):
    def __init__(self, 
                 data_root, datasets, split_suffixes, crop_length=40, downstream=False):
        """
        Dataset
        """
        self.file_paths = []
        self.crop_length = crop_length
        self.downstream = downstream
        for dataset_name in datasets:
            for suffix in split_suffixes:
                folder_name = f"{dataset_name}_{suffix}"
                folder_path = os.path.join(data_root, folder_name)
                if not os.path.exists(folder_path):
                    print(f"Warning: Folder not found: {folder_path}")
                    continue

                for root, dirs, files in os.walk(folder_path):
                    npz_files = glob.glob(os.path.join(root, "*.npz"))
                    if len(npz_files) > 1:
                        # sample_size = max(1, int(len(npz_files) * 0.5)) 
                        # npz_files = random.sample(npz_files, sample_size)
                        npz_files = sorted(npz_files)[:1]
                    self.file_paths.extend(npz_files)

        print(f"Dataset loaded. Total files found: {len(self.file_paths)}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with np.load(file_path) as data_file:
                key = list(data_file.keys())[0]
                fmri_data = data_file[key] 
                fmri_data = fmri_data.astype(np.float32)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

        total_time_frames = fmri_data.shape[-1]
        if total_time_frames > self.crop_length:
            start_idx = np.random.randint(0, total_time_frames - self.crop_length + 1)
            end_idx = start_idx + self.crop_length
            cropped_data = fmri_data[..., start_idx:end_idx]
        else:
            cropped_data = fmri_data[..., :self.crop_length]

        cropped_data = global_zscore_nonzero(cropped_data)
        data_tensor = torch.from_numpy(cropped_data)
        data_tensor = data_tensor.permute(3, 0, 1, 2)

        return data_tensor


class fMRITxtDataset(fMRIDataset):
    """Pre-training dataset loaded from an explicit txt list of files or folders."""

    def __init__(self, data_root, subject_list_txt, crop_length=40, downstream=False):
        if not subject_list_txt:
            raise ValueError("TXT mode requires a train_txt or val_txt path.")

        self.file_paths = self._load_files_from_txt(subject_list_txt, data_root)
        self.crop_length = crop_length
        self.downstream = downstream

        print(f"Dataset Mode: TXT List ({subject_list_txt})")
        print(f"Dataset loaded. Total files found: {len(self.file_paths)}")

    def _resolve_txt_entry(self, entry: str, data_root: str) -> Path:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = Path(data_root).expanduser() / path
        return path

    def _load_files_from_txt(self, txt_path: str, data_root: str) -> list[str]:
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as handle:
            entries = [line.strip() for line in handle if line.strip()]

        file_paths: list[str] = []
        for entry in entries:
            path = self._resolve_txt_entry(entry, data_root)
            if path.is_dir():
                file_paths.extend(sorted(glob.glob(str(path / "**" / "*.npz"), recursive=True)))
            elif path.is_file() and path.suffix == ".npz":
                file_paths.append(str(path))
            else:
                print(f"Warning: TXT entry not found or not an .npz file/folder: {path}")

        return file_paths
