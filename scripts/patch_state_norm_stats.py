#!/usr/bin/env python3
"""Force a specific `state` dimension to be treated as truly constant in a
computed norm_stats.json, by setting q01 == q99 == value.

Why this is needed: transforms.py's near-constant special-case
(`_normalize_quantile` / `_unnormalize_quantile`) only triggers when
`span = q99 - q01 < 0.05`. For `observation.state[12]` (left_hand_qpos0), the
*measured* value has small session-to-session drift (998~1000) plus rare
sensor glitches, giving span ~= 2.0 which is > 0.05, so it is NOT auto-detected
as near-constant -- even though the corresponding *commanded* action
(left_hand_cmd_pos0) is already an exact constant 1000.0 (std=0.0) in every
episode. This script patches the state stats to match, so:
  - normalize:   always maps to 0.0
  - unnormalize: always recovers exactly `value` (default 1000.0), regardless
    of sensor noise/drift/glitches in that dimension.

Usage:
  /home/xudi_ge/openpi/.venv/bin/python scripts/patch_state_norm_stats.py \
    --assets-dir /home/xudi_ge/openpi/assets/gxd_pi05_v3src_pad/local/openpi_V5_mcap0625_v3src_pad \
    --index 12 --value 1000.0

  # Dry run (just show what would change):
  ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True, help="Dir containing norm_stats.json")
    parser.add_argument("--key", default="state", choices=["state", "actions"])
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--value", type=float, default=1000.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = pathlib.Path(args.assets_dir) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text())
    stats = data["norm_stats"][args.key]

    old_q01 = stats["q01"][args.index]
    old_q99 = stats["q99"][args.index]
    old_mean = stats["mean"][args.index]
    old_std = stats["std"][args.index]

    print(f"[INFO] {path}")
    print(f"[INFO] {args.key}[{args.index}] BEFORE: mean={old_mean} std={old_std} q01={old_q01} q99={old_q99} span={old_q99 - old_q01}")
    print(f"[INFO] {args.key}[{args.index}] AFTER : mean={args.value} std=0.0 q01={args.value} q99={args.value} span=0.0")

    if args.dry_run:
        print("[INFO] --dry-run set, not writing changes.")
        return

    backup_path = path.with_suffix(".json.bak")
    if not backup_path.exists():
        shutil.copy(path, backup_path)
        print(f"[INFO] Backed up original to {backup_path}")

    stats["q01"][args.index] = args.value
    stats["q99"][args.index] = args.value
    stats["mean"][args.index] = args.value
    stats["std"][args.index] = 0.0

    path.write_text(json.dumps(data, indent=2))
    print(f"[INFO] Patched and wrote {path}")


if __name__ == "__main__":
    main()
