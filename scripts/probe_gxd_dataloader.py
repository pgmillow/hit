#!/usr/bin/env python3
"""Temporary dataloader probe for gxd_pi05."""

from __future__ import annotations

import os

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main() -> None:
    os.environ.setdefault("HF_LEROBOT_HOME", "/home/xudi_ge/data/lerobot_openpi")

    cfg = _config.get_config("gxd_pi05")
    loader = _data_loader.create_data_loader(
        cfg,
        shuffle=False,
        num_batches=1,
    )

    print(f"dataset_size: {loader.dataset_size}")

    obs, actions = next(iter(loader))
    print(f"actions: {actions.shape} {actions.dtype}")
    print(f"state: {obs.state.shape} {obs.state.dtype}")
    print(f"tokenized_prompt: {obs.tokenized_prompt.shape} {obs.tokenized_prompt.dtype}")
    print(f"tokenized_prompt_mask: {obs.tokenized_prompt_mask.shape} {obs.tokenized_prompt_mask.dtype}")

    for name, image in obs.images.items():
        print(f"image.{name}: {image.shape} {image.dtype}")

    for name, mask in obs.image_masks.items():
        print(f"image_mask.{name}: {mask.shape} {mask.dtype}")


if __name__ == "__main__":
    main()
