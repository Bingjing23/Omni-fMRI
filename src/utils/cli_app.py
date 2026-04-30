from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml


class CliApp:
    """Small base class for object-oriented CLI entrypoints."""

    def __init__(self, argv: Optional[list[str]] = None) -> None:
        self.argv = argv
        self.args: argparse.Namespace | None = None

    def build_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def parse_args(self) -> argparse.Namespace:
        parser = self.build_parser()
        return parser.parse_args(self.argv)

    def run(self) -> None:
        raise NotImplementedError

    @classmethod
    def main(cls, argv: Optional[list[str]] = None) -> None:
        app = cls(argv=argv)
        app.run()


class YamlBackedCliApp(CliApp):
    """CLI app where YAML provides defaults and CLI arguments override them."""

    default_config_path: str | Path | None = None

    def build_parser(self) -> argparse.ArgumentParser:
        raise NotImplementedError

    def configure(self) -> dict:
        raise NotImplementedError

    def load_yaml_config(self, config_path: str | Path | None) -> dict:
        if config_path is None:
            return {}

        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data or {}

    def get_config_path_from_args(self, args: argparse.Namespace) -> str | Path | None:
        config_path = getattr(args, "config", None)
        if config_path is not None:
            return config_path
        return self.default_config_path

    def load_base_config(self, args: argparse.Namespace) -> dict:
        return self.load_yaml_config(self.get_config_path_from_args(args))
