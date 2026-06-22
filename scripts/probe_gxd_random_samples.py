#!/usr/bin/env python3
"""Randomly sample transformed gxd_pi05 data and report outliers."""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import bisect
import os
from collections import defaultdict

import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi import transforms as _transforms


def _top_push(heap: list[tuple[float, int, str]], value: float, index: int, name: str, k: int) -> None:
    item = (float(value), int(index), name)
    if len(heap) < k:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=10.0)
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_LEROBOT_HOME", "/home/xudi_ge/data/lerobot_openpi")

    cfg = dataclasses.replace(_config.get_config("gxd_pi05"), num_workers=0)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw_dataset = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    dataset = _data_loader.transform_dataset(raw_dataset, data_config)

    rng = np.random.default_rng(args.seed)
    indices = rng.integers(0, len(dataset), size=args.samples)
    transform = dataset._transform
    prompt_transform = None
    base_dataset = raw_dataset
    if hasattr(raw_dataset, "_dataset"):
        prompt_transform = raw_dataset._transform
        base_dataset = raw_dataset._dataset
    grouped_indices: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index in indices:
        index = int(index)
        ep_pos = bisect.bisect_right(base_dataset._episode_ends, index)
        ep_start = 0 if ep_pos == 0 else base_dataset._episode_ends[ep_pos - 1]
        grouped_indices[ep_pos].append((index, index - ep_start))

    top: list[tuple[float, int, str]] = []
    bad_finite: list[tuple[int, str]] = []
    over_threshold: list[tuple[int, str, float]] = []
    state_max_values = []
    action_max_values = []

    n = 0
    for ep_pos in sorted(grouped_indices):
        ep_idx = int(base_dataset._episodes[ep_pos]["episode_index"])
        df = base_dataset._load_episode(ep_idx)
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        for index, local_idx in grouped_indices[ep_pos]:
            n += 1
            row = df.iloc[local_idx]
            action_indices = np.minimum(np.arange(local_idx, local_idx + cfg.model.action_horizon), len(df) - 1)
            sample = {
                "observation.images.top": base_dataset._parse_image(row["observation.images.top"]),
                "observation.images.left_wrist": base_dataset._parse_image(row["observation.images.left_wrist"]),
                "observation.images.right_wrist": base_dataset._parse_image(row["observation.images.right_wrist"]),
                "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
                "action": actions[action_indices],
                "task_index": np.asarray(row["task_index"], dtype=np.int64),
            }
            if args.skip_images:
                state = np.asarray(sample["observation.state"], dtype=np.float32)
                action = np.asarray(sample["action"], dtype=np.float32)
                norm_stats = data_config.norm_stats
                if data_config.use_quantile_norm:
                    state = _transforms.Normalize(norm_stats, use_quantiles=True)({"state": state})["state"]
                    action = _transforms.Normalize(norm_stats, use_quantiles=True)({"actions": action})["actions"]
                else:
                    state = _transforms.Normalize(norm_stats, use_quantiles=False)({"state": state})["state"]
                    action = _transforms.Normalize(norm_stats, use_quantiles=False)({"actions": action})["actions"]
                sample = {
                    "state": _transforms.pad_to_dim(state, cfg.model.action_dim, axis=-1),
                    "actions": _transforms.pad_to_dim(action, cfg.model.action_dim, axis=-1),
                }
            elif prompt_transform is not None:
                sample = prompt_transform(sample)
                sample = transform(sample)
            else:
                sample = transform(sample)
            arrays = {
                "state": np.asarray(sample["state"]),
                "actions": np.asarray(sample["actions"]),
            }
            if not args.skip_images:
                for image_name, image in sample["image"].items():
                    arrays[f"image.{image_name}"] = np.asarray(image)

            for name, array in arrays.items():
                if not np.isfinite(array).all():
                    bad_finite.append((int(index), name))
                    continue
                abs_max = float(np.max(np.abs(array)))
                _top_push(top, abs_max, int(index), name, args.top_k)
                if abs_max > args.threshold:
                    over_threshold.append((int(index), name, abs_max))

            state_max_values.append(float(np.max(np.abs(arrays["state"]))))
            action_max_values.append(float(np.max(np.abs(arrays["actions"]))))

            if n % 500 == 0:
                print(f"checked {n}/{args.samples}")

    print(f"dataset_size: {len(dataset)}")
    print(f"samples: {args.samples}, seed: {args.seed}")
    print(f"bad_finite_count: {len(bad_finite)}")
    if bad_finite:
        print("bad_finite_examples:")
        for item in bad_finite[: args.top_k]:
            print(item)

    for label, values in (("state_abs_max", state_max_values), ("actions_abs_max", action_max_values)):
        arr = np.asarray(values)
        print(
            f"{label}: mean={arr.mean():.4f} p95={np.quantile(arr, 0.95):.4f} "
            f"p99={np.quantile(arr, 0.99):.4f} max={arr.max():.4f}"
        )

    print(f"over_threshold_count(abs>{args.threshold}): {len(over_threshold)}")
    for index, name, value in sorted(over_threshold, key=lambda x: x[2], reverse=True)[: args.top_k]:
        print(f"over_threshold index={index} name={name} abs_max={value:.4f}")

    print("top_abs_max:")
    for value, index, name in sorted(top, reverse=True):
        print(f"index={index} name={name} abs_max={value:.4f}")


if __name__ == "__main__":
    main()
