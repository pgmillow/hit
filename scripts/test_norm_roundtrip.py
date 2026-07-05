#!/usr/bin/env python3
"""Sanity-check the normalize -> unnormalize round trip for a given TrainConfig,
using the *real* data pipeline (repack + data transforms + Normalize/Unnormalize)
and the *real* computed norm_stats.json for that config's assets dir.

This does NOT run the model. It only exercises:
  raw sample -> repack_transforms -> data_transforms -> Normalize -> Unnormalize
and checks that we recover the pre-normalize values (within float tolerance),
and reports whether any normalized values would need clipping.

Usage:
  /home/xudi_ge/openpi/.venv/bin/python scripts/test_norm_roundtrip.py --config-name dataV5_final_v3src_pad
"""

from __future__ import annotations

import argparse

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


def _apply(fns, data):
    for fn in fns:
        data = fn(data)
    return data


def main(config_name: str, num_samples: int = 5) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.norm_stats is None:
        raise RuntimeError(
            f"No norm_stats found for config '{config_name}' (assets_dir={config.assets_dirs}). "
            "Run scripts/compute_norm_stats.py first."
        )

    print(f"[INFO] config={config_name} repo_id={data_config.repo_id} use_quantile_norm={data_config.use_quantile_norm}")
    print(f"[INFO] norm_stats keys: {list(data_config.norm_stats.keys())}")

    raw_dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    n = len(raw_dataset)
    print(f"[INFO] raw dataset size: {n}")

    indices = sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1][:num_samples]))

    normalize = _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
    unnormalize = _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)

    max_abs_diff = {"state": 0.0, "actions": 0.0}
    max_norm_abs = {"state": 0.0, "actions": 0.0}
    n_clip_needed = {"state": 0, "actions": 0}

    for idx in indices:
        raw = dict(raw_dataset[idx])
        pre = _apply(data_config.repack_transforms.inputs, dict(raw))
        pre = _apply(data_config.data_transforms.inputs, pre)

        pre_state = np.asarray(pre["state"], dtype=np.float64)
        pre_actions = np.asarray(pre["actions"], dtype=np.float64)

        normed = normalize(dict(pre))
        recovered = unnormalize(dict(normed))

        norm_state = np.asarray(normed["state"], dtype=np.float64)
        norm_actions = np.asarray(normed["actions"], dtype=np.float64)

        diff_state = np.abs(np.asarray(recovered["state"], dtype=np.float64) - pre_state)
        diff_actions = np.abs(np.asarray(recovered["actions"], dtype=np.float64) - pre_actions)

        max_abs_diff["state"] = max(max_abs_diff["state"], float(diff_state.max()))
        max_abs_diff["actions"] = max(max_abs_diff["actions"], float(diff_actions.max()))
        max_norm_abs["state"] = max(max_norm_abs["state"], float(np.abs(norm_state).max()))
        max_norm_abs["actions"] = max(max_norm_abs["actions"], float(np.abs(norm_actions).max()))
        n_clip_needed["state"] += int((np.abs(norm_state) > 1.0 + 1e-6).sum())
        n_clip_needed["actions"] += int((np.abs(norm_actions) > 1.0 + 1e-6).sum())

        has_nan_inf = (
            not np.all(np.isfinite(norm_state))
            or not np.all(np.isfinite(norm_actions))
            or not np.all(np.isfinite(recovered["state"]))
            or not np.all(np.isfinite(recovered["actions"]))
        )
        print(
            f"[sample idx={idx}] "
            f"state roundtrip max|diff|={diff_state.max():.3e}  "
            f"actions roundtrip max|diff|={diff_actions.max():.3e}  "
            f"norm_state range=[{norm_state.min():.3f},{norm_state.max():.3f}]  "
            f"norm_actions range=[{norm_actions.min():.3f},{norm_actions.max():.3f}]  "
            f"nan_or_inf={has_nan_inf}"
        )
        if has_nan_inf:
            raise RuntimeError(f"NaN/Inf detected at sample idx={idx}")

    print()
    print("[SUMMARY]")
    print(f"  max roundtrip |diff|  state={max_abs_diff['state']:.3e}  actions={max_abs_diff['actions']:.3e}")
    print(f"  max |normalized value| state={max_norm_abs['state']:.3f}  actions={max_norm_abs['actions']:.3f}")
    print(f"  # values needing clip (|x|>1) across sampled frames: state={n_clip_needed['state']}  actions={n_clip_needed['actions']}")

    ok = max_abs_diff["state"] < 1e-3 and max_abs_diff["actions"] < 1e-3
    print()
    print("[RESULT] " + ("PASS" if ok else "FAIL") + " - normalize/unnormalize round trip " + ("matches" if ok else "DOES NOT match"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()
    main(args.config_name, args.num_samples)
