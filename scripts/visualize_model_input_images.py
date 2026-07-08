#!/usr/bin/env python3
"""Dump the 3 images (top / left_wrist / right_wrist) actually fed into the pi05 model
for the first N frames of a given config.

Uses create_torch_dataset + transform_dataset directly (no DataLoader workers, no JAX
sharding setup) so it stays light for tiny sample counts.

The transforms applied match the training pipeline:
  repack -> data_transforms.inputs (OctopusInputs, with right-side masks)
  -> Normalize(quantile) -> model_transforms.inputs (ClipState, ResizeImages(224,224),
  TokenizePrompt, PadStatesAndActions)
So `obs["image"]` after this is exactly the 224x224 letterboxed images that go into
SigLIP (the final /127.5-1 normalization happens inside model.preprocess_observation
and is trivially invertible for display).

Usage:
  python scripts/visualize_model_input_images.py --config-name dataV5_final_v3src_pad \
      --num-samples 2 --out-dir /home/xudi_ge/data/model_input_vis
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
from PIL import Image

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="dataV5_final_v3src_pad")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--out-dir", default="/home/xudi_ge/data/model_input_vis")
    parser.add_argument("--hf-lerobot-home", default="/home/xudi_ge/data/lerobot_openpi")
    args = parser.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", args.hf_lerobot_home)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _config.get_config(args.config_name)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)

    print(f"[info] creating torch dataset for repo_id={data_config.repo_id} ...")
    raw_dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    print(f"[info] raw_dataset len={len(raw_dataset)}; applying transforms ...")
    dataset = _data_loader.transform_dataset(raw_dataset, data_config)
    print(f"[info] transformed dataset ready. dumping {args.num_samples} samples ...")

    friendly = {"base_0_rgb": "top", "left_wrist_0_rgb": "left_wrist", "right_wrist_0_rgb": "right_wrist"}
    saved = 0
    for idx in range(len(dataset)):
        if saved >= args.num_samples:
            break
        sample = dataset[idx]
        if not isinstance(sample, dict) or "image" not in sample:
            print(f"[warn] sample {idx} unexpected type={type(sample)}; skip")
            continue
        for name, arr in sample["image"].items():
            arr8 = _to_uint8(arr)
            fn = out_dir / f"sample{saved:02d}_{friendly.get(name, name)}.png"
            Image.fromarray(arr8).save(fn)
            print(f"saved {fn}  shape={arr8.shape} dtype={arr8.dtype} "
                  f"min={arr8.min()} max={arr8.max()} mean={arr8.mean():.1f}")
        state = np.asarray(sample.get("state", np.array([])))
        actions = np.asarray(sample.get("actions", np.array([])))
        print(f"  sample{saved:02d} (raw_idx={idx}) state.shape={state.shape} "
              f"state[:14]={state[:14].tolist() if state.size else []}")
        print(f"  sample{saved:02d} (raw_idx={idx}) actions.shape={actions.shape} "
              f"actions[0]={actions[0].tolist() if actions.size else []}")
        saved += 1

    print(f"\nDone. {saved} samples x 3 images saved to {out_dir}")


if __name__ == "__main__":
    main()
