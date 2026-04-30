#!/usr/bin/env python3
"""Create dummy data and validate the Omni-fMRI Docker runtime without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import finetune
import pretrain


SPATIAL_SHAPE = (96, 96, 96)
TIME_LENGTH = 40
SUBJECT_DIR = "0000001"
SUBJECT_ID = "1"
SAMPLE_NAME = "0000001_run-1_0000-0039_1.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("docker_dummy"),
        help="Directory used for generated dummy data and smoke-test outputs.",
    )
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((*SPATIAL_SHAPE, TIME_LENGTH), dtype=np.uint8)
    array[0, 0, 0, 0] = 1
    np.savez_compressed(path, arr=array)


def create_dummy_data(root: Path) -> Path:
    split_dirs = [
        "ABIDE_train_40",
        "ABIDE_val_40",
        "ABIDE_train",
        "ABIDE_val",
        "ABIDE_test",
    ]
    for split_dir in split_dirs:
        write_npz(root / split_dir / SUBJECT_DIR / SAMPLE_NAME)

    (root / "train.txt").write_text(str(root / "ABIDE_train" / SUBJECT_DIR) + "\n", encoding="utf-8")
    (root / "val.txt").write_text(str(root / "ABIDE_val" / SUBJECT_DIR) + "\n", encoding="utf-8")
    (root / "test.txt").write_text(str(root / "ABIDE_test" / SUBJECT_DIR) + "\n", encoding="utf-8")

    labels_csv = root / "labels.csv"
    labels_csv.write_text(f"Subject,gender,age\n{SUBJECT_ID},M,21\n", encoding="utf-8")
    return labels_csv


def make_pretrain_config(data_root: Path, output_root: Path) -> dict:
    config = load_yaml("configs/pretrain.yaml")
    config["experiment"]["output_dir"] = str(output_root / "pretrain")
    config["data"]["data_root"] = str(data_root)
    config["data"]["datasets"] = ["ABIDE"]
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False
    return config


def make_finetune_config(data_root: Path, labels_csv: Path, output_root: Path) -> dict:
    config = load_yaml("configs/finetune.yaml")
    config["experiment"]["output_dir"] = str(output_root / "finetune")
    config["experiment"]["pretrained_checkpoint"] = None
    config["task"]["csv"] = str(labels_csv)
    config["task"]["task_type"] = "classification"
    config["task"]["target_col"] = "gender"
    config["task"]["num_classes"] = 2
    config["data"]["data_root"] = str(data_root)
    config["data"]["datasets"] = ["ABIDE"]
    config["data"]["mode"] = "directory"
    config["data"]["subject_id_regex"] = r"(\d{7})"
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    config["data"]["pin_memory"] = False
    return config


def make_finetune_txt_config(data_root: Path, labels_csv: Path, output_root: Path) -> dict:
    config = make_finetune_config(data_root, labels_csv, output_root)
    config["experiment"]["output_dir"] = str(output_root / "finetune_txt")
    config["data"]["mode"] = "txt"
    config["data"]["train_txt"] = str(data_root / "train.txt")
    config["data"]["val_txt"] = str(data_root / "val.txt")
    config["data"]["test_txt"] = str(data_root / "test.txt")
    return config


def validate_pretrain_pipeline(config: dict) -> None:
    model = pretrain.create_model(config)
    train_loader, val_loader, _ = pretrain.create_dataloaders(config, False, 0, 1)

    assert len(train_loader.dataset) == 1, "Expected one dummy pretrain sample."
    assert len(val_loader.dataset) == 1, "Expected one dummy pretrain validation sample."

    batch = next(iter(train_loader))
    assert tuple(batch.shape) == (1, TIME_LENGTH, *SPATIAL_SHAPE), batch.shape

    print(
        f"[pretrain] model={type(model).__name__} "
        f"train_samples={len(train_loader.dataset)} "
        f"batch_shape={tuple(batch.shape)}"
    )


def validate_finetune_pipeline(config: dict) -> None:
    model = finetune.create_model(config)
    train_loader, val_loader, test_loader, _ = finetune.create_dataloaders(config, False, 0, 1)

    assert len(train_loader.dataset) == 1, "Expected one dummy downstream train sample."
    assert len(val_loader.dataset) == 1, "Expected one dummy downstream val sample."
    assert len(test_loader.dataset) == 1, "Expected one dummy downstream test sample."

    samples, labels = next(iter(train_loader))
    assert tuple(samples.shape) == (1, TIME_LENGTH, *SPATIAL_SHAPE), samples.shape
    assert tuple(labels.shape) == (1,), labels.shape

    print(
        f"[finetune:{config['data']['mode']}] model={type(model).__name__} "
        f"train_samples={len(train_loader.dataset)} "
        f"batch_shape={tuple(samples.shape)} "
        f"label_shape={tuple(labels.shape)}"
    )


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    data_root = work_dir / "data"
    output_root = work_dir / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    labels_csv = create_dummy_data(data_root)
    print(f"Dummy data created under {data_root}")

    validate_pretrain_pipeline(make_pretrain_config(data_root, output_root))
    validate_finetune_pipeline(make_finetune_config(data_root, labels_csv, output_root))
    validate_finetune_pipeline(make_finetune_txt_config(data_root, labels_csv, output_root))

    print("Docker smoke test passed. Dummy data, configs, dataloaders, and model construction are all valid.")


if __name__ == "__main__":
    main()
