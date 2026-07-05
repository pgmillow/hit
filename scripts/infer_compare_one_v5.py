#!/usr/bin/env python3
"""Run inference on one V5 LeRobot sample and visualize pred vs GT actions.

Example:
  HF_LEROBOT_HOME=/home/xudi_ge/data/lerobot_openpi \\
  CUDA_VISIBLE_DEVICES=2 \\
  /home/xudi_ge/openpi/.venv/bin/python scripts/infer_compare_one_v5.py \\
    --config-name dataV5_final_v3src_pad \\
    --checkpoint-dir /data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/40000 \\
    --index 0 \\
    --out /tmp/v5_infer_compare_idx0.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import openpi.policies.policy_config as policy_config
import openpi.training.config as train_config
import openpi.training.data_loader as data_loader

ACTION_NAMES = [
    "left_arm_cmd_pos0",
    "left_arm_cmd_pos1",
    "left_arm_cmd_pos2",
    "left_arm_cmd_pos3",
    "left_arm_cmd_pos4",
    "left_arm_cmd_pos5",
    "right_arm_cmd_pos0",
    "right_arm_cmd_pos1",
    "right_arm_cmd_pos2",
    "right_arm_cmd_pos3",
    "right_arm_cmd_pos4",
    "right_arm_cmd_pos5",
    "left_hand_cmd_pos0",
    "right_hand_cmd_pos0",
]


def _build_obs(raw: dict) -> dict:
    """Build policy input from a raw LeRobot row (same keys as training dataset)."""
    obs = dict(raw)
    obs.pop("task_index", None)
    return obs


def _plot_compare(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    index: int,
    checkpoint_dir: Path,
    infer_ms: float,
    out_path: Path,
) -> None:
    """Plot GT vs prediction for each action dimension over the action horizon."""
    horizon = gt.shape[0]
    action_dim = gt.shape[1]
    steps = np.arange(horizon)

    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 2.2 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]

    mae_per_dim = np.mean(np.abs(pred - gt), axis=0)
    for i, ax in enumerate(axes):
        ax.plot(steps, gt[:, i], label="GT", color="#1f77b4", linewidth=2.0)
        ax.plot(steps, pred[:, i], label="Pred", color="#ff7f0e", linewidth=1.8, linestyle="--")
        ax.set_ylabel(ACTION_NAMES[i], fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.text(
            0.99,
            0.85,
            f"MAE={mae_per_dim[i]:.4f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("action horizon step")
    overall_mae = float(np.mean(np.abs(pred - gt)))
    overall_rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    fig.suptitle(
        f"V5 sample index={index} | checkpoint={checkpoint_dir.name}\n"
        f"MAE={overall_mae:.5f}  RMSE={overall_rmse:.5f}  infer={infer_ms:.1f} ms",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer one V5 sample and compare with GT.")
    parser.add_argument("--config-name", default="dataV5_final_v3src_pad")
    parser.add_argument(
        "--checkpoint-dir",
        default="/data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/40000",
    )
    parser.add_argument("--index", type=int, default=0, help="LeRobot dataset frame index")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching denoise steps")
    parser.add_argument("--out", type=Path, default=Path("/tmp/v5_infer_compare.png"))
    parser.add_argument(
        "--lerobot-home",
        default="/home/xudi_ge/data/lerobot_openpi",
        help="HF_LEROBOT_HOME path",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", args.lerobot_home)

    config = train_config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)

    index = args.index
    if index < 0 or index >= len(raw_dataset):
        raise IndexError(f"index={index} out of range [0, {len(raw_dataset) - 1}]")

    raw = dict(raw_dataset[index])
    gt = np.asarray(raw["action"], dtype=np.float32)[..., :14]

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_dir}")

    print(f"Loading policy from {checkpoint_dir} ...")
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        repack_transforms=data_config.repack_transforms,
        sample_kwargs={"num_steps": args.num_steps},
    )

    obs = _build_obs(raw)
    if "prompt" in raw:
        print(f"prompt: {raw['prompt']!r}")

    print(f"Running inference on dataset index={index}, action_horizon={gt.shape[0]} ...")
    result = policy.infer(obs)
    pred = np.asarray(result["actions"], dtype=np.float32)
    infer_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))

    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape}, gt={gt.shape}")

    mae = float(np.mean(np.abs(pred - gt)))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    print(f"MAE={mae:.6f}  RMSE={rmse:.6f}  infer={infer_ms:.1f} ms")

    _plot_compare(
        gt,
        pred,
        index=index,
        checkpoint_dir=checkpoint_dir,
        infer_ms=infer_ms,
        out_path=args.out,
    )
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
