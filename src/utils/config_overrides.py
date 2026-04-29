import argparse

import yaml


def _set_nested(config, dotted_key, value):
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or current[key] is None:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _parse_value(value):
    if isinstance(value, str):
        return yaml.safe_load(value)
    return value


def apply_cfg_options(config, options):
    if not options:
        return
    for option in options:
        if "=" not in option:
            raise ValueError(f"Invalid --cfg-options item '{option}', expected key=value")
        key, value = option.split("=", 1)
        _set_nested(config, key, _parse_value(value))


def add_pretrain_override_args(parser):
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--train_split_suffixes", nargs="+", default=None)
    parser.add_argument("--val_split_suffixes", nargs="+", default=None)
    parser.add_argument("--input_seq_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=None)

    parser.add_argument("--img_size", nargs=3, type=int, default=None)
    parser.add_argument("--patch_size", nargs="+", type=int, default=None)
    parser.add_argument("--in_chans", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--drop_path_rate", type=float, default=None)
    parser.add_argument("--num_scales", type=int, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--gate_attention", type=str, default=None)
    parser.add_argument("--model_chose", type=str, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--val_freq", type=int, default=None)
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=None,
        help="Override any config entry with dotted key=value, e.g. data.batch_size=8 model.thresholds='[0.2]'",
    )


def add_finetune_override_args(parser):
    add_pretrain_override_args(parser)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--task_csv", type=str, default=None)
    parser.add_argument("--task_type", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--task_mean", type=float, default=None)
    parser.add_argument("--task_std", type=float, default=None)
    parser.add_argument("--data_mode", type=str, default=None)
    parser.add_argument("--test_split_suffixes", nargs="+", default=None)
    parser.add_argument("--train_txt", type=str, default=None)
    parser.add_argument("--val_txt", type=str, default=None)
    parser.add_argument("--test_txt", type=str, default=None)
    parser.add_argument("--global_pool", type=str, default=None)
    parser.add_argument("--fusion_mode", type=str, default=None)
    parser.add_argument("--freeze_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--head_lr", type=float, default=None)
    parser.add_argument("--layer_decay", type=float, default=None)


def apply_pretrain_overrides(config, args):
    apply_cfg_options(config, getattr(args, "cfg_options", None))

    experiment_map = {"seed": "seed", "resume": "resume", "output_dir": "output_dir"}
    data_map = {
        "data_root": "data_root",
        "datasets": "datasets",
        "train_split_suffixes": "train_split_suffixes",
        "val_split_suffixes": "val_split_suffixes",
        "input_seq_len": "input_seq_len",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "pin_memory": "pin_memory",
        "prefetch_factor": "prefetch_factor",
    }
    model_map = {
        "img_size": "img_size",
        "patch_size": "patch_size",
        "in_chans": "in_chans",
        "embed_dim": "embed_dim",
        "depth": "depth",
        "num_heads": "num_heads",
        "drop_path_rate": "drop_path_rate",
        "num_scales": "num_scales",
        "thresholds": "thresholds",
        "method": "method",
        "gate_attention": "gate_attention",
        "model_chose": "model_chose",
        "mask_ratio": "mask_ratio",
    }
    training_map = {
        "epochs": "epochs",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "warmup_epochs": "warmup_epochs",
        "min_lr": "min_lr",
    }
    logging_map = {"save_freq": "save_freq"}
    validation_map = {"val_freq": "val_freq"}

    _apply_map(config, "experiment", experiment_map, args)
    _apply_map(config, "data", data_map, args)
    _apply_map(config, "model", model_map, args)
    _apply_map(config, "training", training_map, args)
    _apply_map(config, "logging", logging_map, args)
    _apply_map(config, "validation", validation_map, args)

    patch_size = getattr(args, "patch_size", None)
    if patch_size is not None and len(patch_size) == 1:
        config["model"]["patch_size"] = patch_size[0]


def apply_finetune_overrides(config, args):
    apply_pretrain_overrides(config, args)

    finetune_experiment = {"pretrained_checkpoint": "pretrained_checkpoint"}
    task_map = {
        "task_csv": "csv",
        "task_type": "task_type",
        "num_classes": "num_classes",
        "task_mean": "mean",
        "task_std": "std",
    }
    data_map = {
        "data_mode": "mode",
        "test_split_suffixes": "test_split_suffixes",
        "train_txt": "train_txt",
        "val_txt": "val_txt",
        "test_txt": "test_txt",
    }
    model_map = {"global_pool": "global_pool", "fusion_mode": "fusion_mode"}
    training_map = {
        "freeze_encoder": "freeze_encoder",
        "head_lr": "head_lr",
        "layer_decay": "layer_decay",
    }

    _apply_map(config, "experiment", finetune_experiment, args)
    _apply_map(config, "task", task_map, args)
    _apply_map(config, "data", data_map, args)
    _apply_map(config, "model", model_map, args)
    _apply_map(config, "training", training_map, args)


def _apply_map(config, section, mapping, args):
    if section not in config or config[section] is None:
        config[section] = {}
    for arg_name, config_name in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[section][config_name] = value

