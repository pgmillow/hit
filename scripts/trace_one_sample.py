#!/usr/bin/env python3
"""Trace exactly what happens to ONE real sample as it flows through the data
pipeline: raw LeRobot row -> repack_transforms -> data_transforms (OctopusInputs)
-> Normalize (quantile) -> model_transforms (ClipState/ResizeImages/Tokenize/Pad).

Prints actual per-dimension values (with names) at each stage so you can see
exactly what normalization does to real numbers.

Usage:
  /home/xudi_ge/openpi/.venv/bin/python scripts/trace_one_sample.py \
    --config-name dataV5_final_v3src_pad --index 0
"""

from __future__ import annotations

import argparse

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

STATE_NAMES = [
    "left_arm_qpos0", "left_arm_qpos1", "left_arm_qpos2", "left_arm_qpos3", "left_arm_qpos4", "left_arm_qpos5",
    "right_arm_qpos0", "right_arm_qpos1", "right_arm_qpos2", "right_arm_qpos3", "right_arm_qpos4", "right_arm_qpos5",
    "left_hand_qpos0", "right_hand_qpos0",
    "left_ee_pose_x", "left_ee_pose_y", "left_ee_pose_z", "left_ee_pose_roll", "left_ee_pose_pitch", "left_ee_pose_yaw",
    "right_ee_pose_x", "right_ee_pose_y", "right_ee_pose_z", "right_ee_pose_roll", "right_ee_pose_pitch", "right_ee_pose_yaw",
]
ACTION_NAMES = [
    "left_arm_cmd_pos0", "left_arm_cmd_pos1", "left_arm_cmd_pos2", "left_arm_cmd_pos3", "left_arm_cmd_pos4", "left_arm_cmd_pos5",
    "right_arm_cmd_pos0", "right_arm_cmd_pos1", "right_arm_cmd_pos2", "right_arm_cmd_pos3", "right_arm_cmd_pos4", "right_arm_cmd_pos5",
    "left_hand_cmd_pos0", "right_hand_cmd_pos0",
]


def _print_vec(names: list[str], raw, q01=None, q99=None, normed=None) -> None:
    for i, name in enumerate(names):
        parts = [f"  [{i:2d}] {name:22s} raw={raw[i]: .6f}"]
        if q01 is not None and q99 is not None:
            parts.append(f" q01={q01[i]: .6f} q99={q99[i]: .6f} span={q99[i]-q01[i]: .6f}")
        if normed is not None:
            parts.append(f" -> normalized={normed[i]: .6f}")
        print("".join(parts))


def main(config_name: str, index: int) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise RuntimeError("norm_stats not found; run compute_norm_stats.py first")

    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    raw = dict(raw_dataset[index])

    print("=" * 100)
    print(f"STAGE 0: raw sample from LeRobot dataset (index={index})")
    print("=" * 100)
    for k, v in raw.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)} dtype={getattr(v, 'dtype', type(v))}")
        else:
            print(f"  {k}: {v!r}"[:200])
    raw_state = np.asarray(raw["observation.state"], dtype=np.float64)
    raw_actions = np.asarray(raw["action"], dtype=np.float64)
    if raw_actions.ndim > 1:
        raw_actions_first = raw_actions[0]
    else:
        raw_actions_first = raw_actions
    print("\n  -- raw observation.state (named) --")
    for i, name in enumerate(STATE_NAMES):
        print(f"  [{i:2d}] {name:22s} = {raw_state[i]: .6f}")
    print("\n  -- raw action (named, first timestep of horizon if chunked) --")
    for i, name in enumerate(ACTION_NAMES):
        print(f"  [{i:2d}] {name:22s} = {raw_actions_first[i]: .6f}")

    print()
    print("=" * 100)
    print("STAGE 1: repack_transforms.inputs (rename dataset columns -> canonical keys)")
    print("=" * 100)
    d = dict(raw)
    for fn in data_config.repack_transforms.inputs:
        d = fn(d)
        print(f"  after {fn.__class__.__name__}: keys={list(d.keys())}")

    print()
    print("=" * 100)
    print("STAGE 2: data_transforms.inputs (OctopusInputs: build model-facing dict)")
    print("=" * 100)
    for fn in data_config.data_transforms.inputs:
        d = fn(d)
        print(f"  after {fn.__class__.__name__}: keys={list(d.keys())}")
        if "image" in d:
            for cam, img in d["image"].items():
                print(f"    image[{cam}] shape={img.shape} dtype={img.dtype}")

    pre_state = np.asarray(d["state"], dtype=np.float64)
    pre_actions = np.asarray(d["actions"], dtype=np.float64)
    pre_actions_first = pre_actions[0] if pre_actions.ndim > 1 else pre_actions
    print(f"\n  pre-normalize state shape={pre_state.shape}  actions shape={pre_actions.shape}")

    print()
    print("=" * 100)
    print("STAGE 3: Normalize (quantile, use_quantiles=True) — this is the actual training-time transform")
    print("=" * 100)
    stats = data_config.norm_stats
    state_stats = stats["state"]
    actions_stats = stats["actions"]
    q01_s, q99_s = np.asarray(state_stats.q01, dtype=np.float64), np.asarray(state_stats.q99, dtype=np.float64)
    q01_a, q99_a = np.asarray(actions_stats.q01, dtype=np.float64), np.asarray(actions_stats.q99, dtype=np.float64)

    normalize = _transforms.Normalize(stats, use_quantiles=data_config.use_quantile_norm)
    normed = normalize(dict(d))
    norm_state = np.asarray(normed["state"], dtype=np.float64)
    norm_actions = np.asarray(normed["actions"], dtype=np.float64)
    norm_actions_first = norm_actions[0] if norm_actions.ndim > 1 else norm_actions

    print("\n  formula per-dim: span = q99 - q01")
    print("    if span < 0.05 (near-constant dim): normalized = 0.0   (avoid noise amplification)")
    print("    else: normalized = (raw - q01) / (span + 1e-6) * 2.0 - 1.0")
    print("\n  -- state: raw -> q01/q99 -> normalized --")
    _print_vec(STATE_NAMES, raw_state, q01_s[: len(STATE_NAMES)], q99_s[: len(STATE_NAMES)], norm_state[: len(STATE_NAMES)])
    print("\n  -- actions (first horizon step): raw -> q01/q99 -> normalized --")
    _print_vec(ACTION_NAMES, raw_actions_first, q01_a[: len(ACTION_NAMES)], q99_a[: len(ACTION_NAMES)], norm_actions_first[: len(ACTION_NAMES)])

    print()
    print("=" * 100)
    print("STAGE 4: model_transforms.inputs (ClipState / ResizeImages / TokenizePrompt / PadStatesAndActions)")
    print("=" * 100)
    d2 = dict(normed)
    for fn in data_config.model_transforms.inputs:
        before_state = np.asarray(d2["state"], dtype=np.float64).copy() if "state" in d2 else None
        d2 = fn(d2)
        msg = f"  after {fn.__class__.__name__}: keys={list(d2.keys())}"
        if "state" in d2:
            msg += f"  state.shape={np.asarray(d2['state']).shape}"
        if "actions" in d2:
            msg += f"  actions.shape={np.asarray(d2['actions']).shape}"
        print(msg)
        if before_state is not None and "state" in d2:
            after_state = np.asarray(d2["state"], dtype=np.float64)
            if after_state.shape == before_state.shape and not np.allclose(after_state, before_state):
                n_changed = int(np.sum(~np.isclose(after_state, before_state)))
                print(f"    -> {fn.__class__.__name__} changed {n_changed} state values (e.g. clipping)")

    print()
    print("=" * 100)
    print("STAGE 5: Unnormalize (inverse of stage 3, applied at inference time to model output)")
    print("=" * 100)
    unnormalize = _transforms.Unnormalize(stats, use_quantiles=data_config.use_quantile_norm)
    recovered = unnormalize({"state": normed["state"], "actions": normed["actions"]})
    rec_state = np.asarray(recovered["state"], dtype=np.float64)
    rec_actions = np.asarray(recovered["actions"], dtype=np.float64)
    rec_actions_first = rec_actions[0] if rec_actions.ndim > 1 else rec_actions
    print("\n  -- state: normalized -> recovered (compare to STAGE 0 raw) --")
    for i, name in enumerate(STATE_NAMES):
        diff = rec_state[i] - raw_state[i]
        print(f"  [{i:2d}] {name:22s} normalized={norm_state[i]: .6f} -> recovered={rec_state[i]: .6f}  (raw={raw_state[i]: .6f}, diff={diff: .2e})")
    print("\n  -- actions (first horizon step): normalized -> recovered --")
    for i, name in enumerate(ACTION_NAMES):
        diff = rec_actions_first[i] - raw_actions_first[i]
        print(f"  [{i:2d}] {name:22s} normalized={norm_actions_first[i]: .6f} -> recovered={rec_actions_first[i]: .6f}  (raw={raw_actions_first[i]: .6f}, diff={diff: .2e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    main(args.config_name, args.index)
