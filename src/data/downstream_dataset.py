import glob
import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .pretrain_dataset import fMRIDataset


def normalize_subject_id(subject_id: object) -> str:
    subject_id = str(subject_id).strip()
    if subject_id.isdigit():
        return subject_id.lstrip("0") or "0"
    return subject_id


def extract_subject_id(path: str | os.PathLike, subject_id_regex: str) -> str:
    if not subject_id_regex:
        raise ValueError("subject_id_regex must be provided.")

    path = Path(path)
    candidates = [path.name, path.parent.name, str(path)]
    for candidate in candidates:
        match = re.search(subject_id_regex, candidate)
        if match:
            if "subject" in match.groupdict():
                return normalize_subject_id(match.group("subject"))
            if match.groups():
                return normalize_subject_id(match.group(1))
            return normalize_subject_id(match.group(0))
    return ""


def load_label_map(
    csv_path: str,
    *,
    task_type: Literal["classification", "regression"],
    target_col: str,
    subject_col: str = "Subject",
) -> dict[str, torch.Tensor]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Label CSV file not found at: {csv_path}")
    if not target_col:
        raise ValueError("--target_col is required for downstream datasets.")

    print(f"Loading labels from {csv_path}...")
    df = pd.read_csv(csv_path)
    if subject_col not in df.columns:
        raise ValueError(f"CSV must contain subject column '{subject_col}'.")
    if target_col not in df.columns:
        raise ValueError(f"CSV must contain target column '{target_col}'.")

    df = df.dropna(subset=[subject_col, target_col]).copy()
    df[subject_col] = df[subject_col].map(normalize_subject_id)

    labels_map: dict[str, torch.Tensor] = {}
    if task_type == "classification":
        values = df[target_col]
        sex_mapping = {"F": 0, "M": 1, "f": 0, "m": 1}

        non_null = values.dropna()
        if non_null.empty:
            raise ValueError(f"Target column '{target_col}' does not contain any valid values.")

        sample = str(non_null.iloc[0]).strip()
        if sample in sex_mapping:
            df = df[df[target_col].isin(sex_mapping.keys())].copy()
            df[target_col] = df[target_col].map(sex_mapping)
        else:
            numeric_values = pd.to_numeric(df[target_col], errors="coerce")
            if numeric_values.notna().all():
                df[target_col] = numeric_values.astype(int)
            else:
                categories = sorted(str(value) for value in df[target_col].dropna().unique())
                label_to_id = {label: idx for idx, label in enumerate(categories)}
                print(f"Encoding '{target_col}' categories: {label_to_id}")
                df[target_col] = df[target_col].map(lambda value: label_to_id[str(value)])

        for _, row in df.iterrows():
            labels_map[row[subject_col]] = torch.tensor(row[target_col], dtype=torch.long)

    elif task_type == "regression":
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna(subset=[target_col])
        for _, row in df.iterrows():
            labels_map[row[subject_col]] = torch.tensor(row[target_col], dtype=torch.float32).view(1)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    print(f"Using column '{target_col}' as target.")
    print(f"Successfully loaded {len(labels_map)} subjects' labels.")
    return labels_map


class fMRITaskDataset(fMRIDataset):
    """Downstream dataset discovered from data_root/dataset_split folders."""

    def __init__(
        self,
        data_root: str,
        datasets: list[str],
        split_suffixes: list[str],
        crop_length: int,
        label_csv_path: str,
        target_col: str,
        subject_id_regex: str,
        task_type: Literal["classification", "regression"] = "classification",
        downstream: bool = True,
    ):
        super().__init__(data_root, datasets, split_suffixes, crop_length, downstream)

        self.task_type = task_type
        self.target_col = target_col
        self.subject_id_regex = subject_id_regex
        self.labels_map = load_label_map(
            label_csv_path,
            task_type=task_type,
            target_col=target_col,
        )

        initial_file_count = len(self.file_paths)
        self.file_paths = [
            path for path in self.file_paths
            if self._extract_subject_id(path) in self.labels_map
        ]

        dropped = initial_file_count - len(self.file_paths)
        if dropped:
            print(f"Warning: Dropped {dropped} files due to missing labels in CSV.")

        print(f"Task Dataset ready for {self.task_type}. Usable files: {len(self.file_paths)}")

    def _extract_subject_id(self, file_path: str) -> str:
        return extract_subject_id(file_path, self.subject_id_regex)

    def target_values(self) -> torch.Tensor:
        labels = [
            self.labels_map[self._extract_subject_id(path)].float().view(-1)
            for path in self.file_paths
        ]
        if not labels:
            raise RuntimeError("Cannot compute target statistics from an empty dataset.")
        return torch.cat(labels)

    def __getitem__(self, idx: int):
        retries = 0
        max_retries = 100
        while retries < max_retries:
            try:
                data_tensor = super().__getitem__(idx)
                if data_tensor is None:
                    raise ValueError(f"Failed to load data at index {idx} (super returned None)")

                subject_id = self._extract_subject_id(self.file_paths[idx])
                if subject_id in self.labels_map:
                    return data_tensor, self.labels_map[subject_id]
                raise KeyError(f"Label not found for subject ID: {subject_id}")
            except Exception:
                idx = np.random.randint(0, len(self))
                retries += 1

        raise RuntimeError(f"Failed to load any valid data after {max_retries} retries.")


class fMRITaskDataset1(fMRIDataset):
    """Downstream dataset loaded from an explicit txt list of files or folders."""

    def __init__(
        self,
        data_root: str,
        crop_length: int,
        label_csv_path: str,
        target_col: str,
        subject_id_regex: str,
        task_type: Literal["classification", "regression"] = "classification",
        downstream: bool = True,
        subject_list_txt: str | None = None,
    ):
        if not subject_list_txt:
            raise ValueError("TXT mode requires train_txt, val_txt, and test_txt paths.")

        self.file_paths: list[str] = []
        self.crop_length = crop_length
        self.downstream = downstream
        self.task_type = task_type
        self.target_col = target_col
        self.subject_id_regex = subject_id_regex
        self.labels_map = load_label_map(
            label_csv_path,
            task_type=task_type,
            target_col=target_col,
        )
        self.file_paths = self._load_files_from_txt(subject_list_txt, data_root)

        print(f"Task Dataset ready for {self.task_type}. Total usable files: {len(self.file_paths)}")

    def _resolve_txt_entry(self, entry: str, data_root: str) -> Path:
        path = Path(entry).expanduser()
        if not path.is_absolute():
            path = Path(data_root).expanduser() / path
        return path

    def _load_files_from_txt(self, txt_path: str, data_root: str) -> list[str]:
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"TXT file not found: {txt_path}")

        valid_files: list[str] = []
        with open(txt_path, "r", encoding="utf-8") as handle:
            entries = [line.strip() for line in handle if line.strip()]

        print(f"Dataset Mode: TXT List ({txt_path}); entries={len(entries)}")
        for entry in entries:
            path = self._resolve_txt_entry(entry, data_root)
            if path.is_dir():
                candidate_files = sorted(glob.glob(str(path / "**" / "*.npz"), recursive=True))
            elif path.is_file():
                candidate_files = [str(path)] if path.suffix == ".npz" else []
            else:
                print(f"Warning: TXT entry not found: {path}")
                continue

            for file_path in candidate_files:
                subject_id = self._extract_subject_id(file_path)
                if subject_id in self.labels_map:
                    valid_files.append(file_path)

        return valid_files

    def _extract_subject_id(self, path: str) -> str:
        return extract_subject_id(path, self.subject_id_regex)

    def target_values(self) -> torch.Tensor:
        labels = [
            self.labels_map[self._extract_subject_id(path)].float().view(-1)
            for path in self.file_paths
        ]
        if not labels:
            raise RuntimeError("Cannot compute target statistics from an empty dataset.")
        return torch.cat(labels)

    def __getitem__(self, idx: int):
        retries = 0
        max_retries = 30
        while retries < max_retries:
            try:
                data_tensor = super().__getitem__(idx)
                if data_tensor is None:
                    raise ValueError("Super class returned None")

                subject_id = self._extract_subject_id(self.file_paths[idx])
                if subject_id in self.labels_map:
                    return data_tensor, self.labels_map[subject_id]
                raise KeyError(f"Label not found for subject ID: {subject_id}")
            except Exception:
                idx = np.random.randint(0, len(self))
                retries += 1

        raise RuntimeError(f"Failed to load data after {max_retries} retries.")
