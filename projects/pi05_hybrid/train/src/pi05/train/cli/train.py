"""CLI for Pi0.5 LoRA training."""

from __future__ import annotations

import argparse
from pathlib import Path

from pi05.common.config.schema import load_experiment_config
from pi05.common.utils.paths import bootstrap_project_paths, default_train_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven Pi0.5 LoRA training.")
    parser.add_argument("--config", type=Path, default=default_train_config_path(), help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    bootstrap_project_paths(include_project_src=False)
    from pi05.train.engine.trainer import train_from_config

    train_from_config(config)


if __name__ == "__main__":
    main()
