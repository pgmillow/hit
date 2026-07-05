#!/usr/bin/env python3
"""Run OpenPI V5 inference on multiple LeRobot frames and produce an overview plot.

Loads the trained policy once, iterates a list of dataset frame indices, and
writes:
  - one combined overview figure (per-dim MAE bar chart + overall MAE/RMSE table)
  - one per-frame GT-vs-Pred comparison figure (same layout as infer_compare_one_v5)
  - a JSON summary with per-frame metrics

Example:
  CUDA_VISIBLE_DEVICES=2 HF_LEROBOT_HOME=/home/xudi_ge/data/lerobot_openpi \\
  /home/xudi_ge/openpi/.venv/bin/python scripts/infer_compare_v5_multi.py \\
    --config-name dataV5_final_v3src_pad \\
    --checkpoint-dir /data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/40000 \\
    --indices 500 2031 3673 5327 7061 \\
    --out-dir /tmp/v5_infer_multi
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import openpi.policies.policy_config as policy_config
import openpi.training.config as train_config
import openpi.training.data_loader as data_loader


def _load_episode_offsets(lerobot_home: Path, repo_id: str) -> list[tuple[int, int, int]]:
    """Return list of (episode_index, start_frame_idx, length) sorted by episode_index."""
    meta = lerobot_home / repo_id.replace("local/", "local/") / "meta" / "episodes.jsonl"
    if not meta.exists():
        # try alternate layout: <home>/local/<repo>/meta/...
        meta = lerobot_home / "local" / repo_id.split("/")[-1] / "meta" / "episodes.jsonl"
    eps = [json.loads(l) for l in meta.read_text().splitlines() if l.strip()]
    eps.sort(key=lambda e: e["episode_index"])
    cum = 0
    out = []
    for e in eps:
        out.append((int(e["episode_index"]), cum, int(e["length"])))
        cum += int(e["length"])
    return out


def _locate_frame(offsets: list[tuple[int, int, int]], frame_idx: int) -> tuple[int, int]:
    """Binary search frame_idx -> (episode_index, frame_in_episode)."""
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        ep_idx, start, length = offsets[mid]
        if frame_idx < start:
            hi = mid - 1
        elif frame_idx >= start + length:
            lo = mid + 1
        else:
            return ep_idx, frame_idx - start
    return -1, -1

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
    obs = dict(raw)
    obs.pop("task_index", None)
    return obs


def _plot_one(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    index: int,
    episode_index: int,
    frame_in_episode: int,
    checkpoint_dir: Path,
    infer_ms: float,
    out_path: Path,
) -> None:
    horizon, action_dim = gt.shape
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
        f"V5 idx={index} ep={episode_index} f={frame_in_episode} | ckpt={checkpoint_dir.name}\n"
        f"MAE={overall_mae:.5f}  RMSE={overall_rmse:.5f}  infer={infer_ms:.1f} ms",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_overview(records: list[dict], out_path: Path, checkpoint_dir: Path) -> None:
    n = len(records)
    dim = len(ACTION_NAMES)
    mae_mat = np.stack([r["mae_per_dim"] for r in records], axis=0)  # (n, dim)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])

    ax_bar = fig.add_subplot(gs[0, :])
    x = np.arange(dim)
    width = 0.8 / max(1, n)
    for i, r in enumerate(records):
        ax_bar.bar(
            x + i * width - 0.4 + width / 2,
            r["mae_per_dim"],
            width=width,
            label=f"idx={r['index']} ep={r['episode_index']}",
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("per-dim MAE")
    ax_bar.set_title(f"Per-dim MAE across frames | ckpt={checkpoint_dir.name}")
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(fontsize=8, loc="upper right", ncol=min(n, 4))

    ax_overall = fig.add_subplot(gs[1, 0])
    labels = [f"ep{r['episode_index']}\nf{r['frame_in_episode']}" for r in records]
    maes = [r["mae"] for r in records]
    rmses = [r["rmse"] for r in records]
    xpos = np.arange(n)
    ax_overall.bar(xpos - 0.2, maes, 0.4, label="MAE", color="#1f77b4")
    ax_overall.bar(xpos + 0.2, rmses, 0.4, label="RMSE", color="#ff7f0e")
    ax_overall.set_xticks(xpos)
    ax_overall.set_xticklabels(labels, fontsize=8)
    ax_overall.set_ylabel("error")
    ax_overall.set_title("Overall MAE / RMSE per frame")
    ax_overall.grid(True, axis="y", alpha=0.25)
    ax_overall.legend()

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    avg_mae = float(np.mean(maes))
    avg_rmse = float(np.mean(rmses))
    avg_infer = float(np.mean([r["infer_ms"] for r in records[1:]])) if n > 1 else float(records[0]["infer_ms"])
    cell_text = [[f"{r['index']}", f"{r['episode_index']}", f"{r['frame_in_episode']}",
                  f"{r['mae']:.5f}", f"{r['rmse']:.5f}", f"{r['infer_ms']:.1f}"] for r in records]
    cell_text.append(["mean", "-", "-", f"{avg_mae:.5f}", f"{avg_rmse:.5f}", f"{avg_infer:.1f}"])
    table = ax_text.table(
        cellText=cell_text,
        colLabels=["idx", "ep", "f_in_ep", "MAE", "RMSE", "infer_ms"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax_text.set_title("Summary", fontsize=10)

    fig.suptitle(
        f"V5 multi-frame inference overview | n={n} | meanMAE={avg_mae:.5f} meanRMSE={avg_rmse:.5f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer multiple V5 frames and produce overview.")
    parser.add_argument("--config-name", default="dataV5_final_v3src_pad")
    parser.add_argument(
        "--checkpoint-dir",
        default="/data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/40000",
    )
    parser.add_argument("--indices", type=int, nargs="+", required=True,
                        help="Dataset frame indices to evaluate.")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/v5_infer_multi"))
    parser.add_argument("--lerobot-home", default="/home/xudi_ge/data/lerobot_openpi")
    parser.add_argument("--skip-per-frame-plots", action="store_true",
                        help="Only write the overview figure and JSON summary.")
    args = parser.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", args.lerobot_home)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    config = train_config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    raw_dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)

    offsets = _load_episode_offsets(Path(args.lerobot_home), config.data.repo_id)

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

    records: list[dict] = []
    for idx in args.indices:
        if idx < 0 or idx >= len(raw_dataset):
            print(f"[skip] index={idx} out of range [0, {len(raw_dataset) - 1}]")
            continue
        raw = dict(raw_dataset[idx])
        gt = np.asarray(raw["action"], dtype=np.float32)[..., :14]

        episode_index, frame_index = _locate_frame(offsets, idx)
        prompt = raw.get("prompt")

        obs = _build_obs(raw)
        print(f"[idx={idx}] ep={episode_index} f={frame_index} prompt={prompt!r} running inference ...")
        result = policy.infer(obs)
        pred = np.asarray(result["actions"], dtype=np.float32)
        infer_ms = float(result.get("policy_timing", {}).get("infer_ms", 0.0))

        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch at idx={idx}: pred={pred.shape}, gt={gt.shape}")

        mae = float(np.mean(np.abs(pred - gt)))
        rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
        mae_per_dim = np.mean(np.abs(pred - gt), axis=0)
        print(f"        MAE={mae:.6f}  RMSE={rmse:.6f}  infer={infer_ms:.1f} ms")

        rec = {
            "index": int(idx),
            "episode_index": int(episode_index),
            "frame_in_episode": int(frame_index),
            "prompt": str(prompt) if prompt is not None else None,
            "mae": mae,
            "rmse": rmse,
            "infer_ms": infer_ms,
            "mae_per_dim": mae_per_dim.tolist(),
        }
        records.append(rec)

        if not args.skip_per_frame_plots:
            _plot_one(
                gt, pred,
                index=int(idx),
                episode_index=int(episode_index),
                frame_in_episode=int(frame_index),
                checkpoint_dir=checkpoint_dir,
                infer_ms=infer_ms,
                out_path=args.out_dir / f"frame_idx{idx}_ep{episode_index}.png",
            )

    if not records:
        raise RuntimeError("No frames were evaluated.")

    overview_path = args.out_dir / "overview.png"
    _plot_overview(records, overview_path, checkpoint_dir)
    print(f"Saved overview to {overview_path}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "config_name": args.config_name,
        "checkpoint_dir": str(checkpoint_dir),
        "num_steps": args.num_steps,
        "records": records,
        "mean_mae": float(np.mean([r["mae"] for r in records])),
        "mean_rmse": float(np.mean([r["rmse"] for r in records])),
    }, indent=2))
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
