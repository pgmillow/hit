#!/usr/bin/env python3
"""Repack official LeRobot v3 datasets into OpenPI local-reader layout.

OpenPI ``LocalLeRobotParquetDataset`` expects:
  meta/tasks.jsonl
  meta/episodes.jsonl
  data/chunk-000/episode_XXXXXX.parquet   (also accepts file-XXX.parquet)

``/data/double_V1`` already has the correct numeric schema (60 Hz, state26, action14,
3x RGB). This script only rewrites metadata and links/copies parquet files.

Example:
  python scripts/repack_lerobot_v3_for_openpi.py \\
    --input-dir /data/double_V1 \\
    --output-dir /data/double_V1_openpi \\
    --repo-id local/double_V1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_COLUMNS = {
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.state",
    "action",
    "task_index",
}


def _load_tasks(input_dir: Path) -> list[dict]:
    tasks_parquet = input_dir / "meta" / "tasks.parquet"
    tasks_jsonl = input_dir / "meta" / "tasks.jsonl"
    if tasks_jsonl.exists():
        return [json.loads(line) for line in tasks_jsonl.read_text().splitlines() if line.strip()]
    if not tasks_parquet.exists():
        raise FileNotFoundError(f"Missing tasks meta under {input_dir / 'meta'}")
    table = pq.read_table(tasks_parquet)
    data = table.to_pydict()
    return [
        {"task_index": int(task_index), "task": str(task)}
        for task_index, task in zip(data["task_index"], data["task"], strict=True)
    ]


def _load_episodes(input_dir: Path) -> list[dict]:
    episodes_jsonl = input_dir / "meta" / "episodes.jsonl"
    if episodes_jsonl.exists():
        return [json.loads(line) for line in episodes_jsonl.read_text().splitlines() if line.strip()]

    episode_files = sorted((input_dir / "meta" / "episodes").rglob("*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"Missing episodes meta under {input_dir / 'meta'}")

    rows: list[dict] = []
    for path in episode_files:
        table = pq.read_table(path)
        data = table.to_pydict()
        for i in range(table.num_rows):
            tasks = data["tasks"][i]
            if isinstance(tasks, str):
                tasks = [tasks]
            rows.append(
                {
                    "episode_index": int(data["episode_index"][i]),
                    "tasks": list(tasks),
                    "length": int(data["length"][i]),
                    "data_chunk_index": int(data["data/chunk_index"][i])
                    if "data/chunk_index" in data
                    else 0,
                    "data_file_index": int(data["data/file_index"][i])
                    if "data/file_index" in data
                    else int(data["episode_index"][i]),
                }
            )
    rows.sort(key=lambda x: x["episode_index"])
    return rows


def _link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def repack(
    input_dir: Path,
    output_dir: Path,
    *,
    repo_id: str,
    link_mode: str,
    overwrite: bool,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    info_path = input_dir / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    features = set(info.get("features", {}))
    missing = REQUIRED_COLUMNS - features
    if missing:
        raise ValueError(f"Input dataset missing required features: {sorted(missing)}")

    tasks = _load_tasks(input_dir)
    episodes = _load_episodes(input_dir)

    # Validate first parquet schema quickly.
    sample_ep = episodes[0]
    sample_src = (
        input_dir
        / "data"
        / f"chunk-{sample_ep['data_chunk_index']:03d}"
        / f"file-{sample_ep['data_file_index']:03d}.parquet"
    )
    if not sample_src.exists():
        # fallback naming used by some exports
        sample_src = (
            input_dir
            / "data"
            / f"chunk-{sample_ep['data_chunk_index']:03d}"
            / f"episode_{sample_ep['episode_index']:06d}.parquet"
        )
    sample_cols = set(pq.read_schema(sample_src).names)
    missing_cols = REQUIRED_COLUMNS - sample_cols
    if missing_cols:
        raise ValueError(f"{sample_src} missing columns: {sorted(missing_cols)}")

    meta_out = output_dir / "meta"
    data_out = output_dir / "data" / "chunk-000"
    meta_out.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    # OpenPI-friendly info.json
    info_out = dict(info)
    info_out["codebase_version"] = info.get("codebase_version", "v3.0")
    info_out["data_path"] = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
    info_out["repo_id"] = repo_id
    info_out["total_episodes"] = len(episodes)
    info_out["total_frames"] = int(sum(ep["length"] for ep in episodes))
    info_out["total_tasks"] = len(tasks)
    (meta_out / "info.json").write_text(json.dumps(info_out, indent=2) + "\n")

    with (meta_out / "tasks.jsonl").open("w") as f:
        for item in tasks:
            f.write(json.dumps({"task_index": int(item["task_index"]), "task": item["task"]}, ensure_ascii=False) + "\n")

    with (meta_out / "episodes.jsonl").open("w") as f:
        for ep in episodes:
            f.write(
                json.dumps(
                    {
                        "episode_index": int(ep["episode_index"]),
                        "tasks": list(ep["tasks"]),
                        "length": int(ep["length"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    linked = 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        chunk_idx = int(ep["data_chunk_index"])
        file_idx = int(ep["data_file_index"])
        src_candidates = [
            input_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet",
            input_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet",
        ]
        src = next((p for p in src_candidates if p.exists()), None)
        if src is None:
            raise FileNotFoundError(
                f"No parquet for episode {ep_idx}. Tried: {[str(p) for p in src_candidates]}"
            )
        dst = data_out / f"episode_{ep_idx:06d}.parquet"
        _link_or_copy(src, dst, link_mode)
        linked += 1

    print(f"[ok] wrote OpenPI dataset -> {output_dir}")
    print(f"     episodes={linked} frames={info_out['total_frames']} fps={info_out.get('fps')}")
    print(f"     tasks={len(tasks)} link_mode={link_mode}")
    print(f"     repo_id={repo_id}")
    print()
    print("Next (symlink into HF_LEROBOT_HOME):")
    print(f"  mkdir -p \"$HOME/.cache/huggingface/lerobot/local\"")
    print(f"  ln -sfn {output_dir} \"$HOME/.cache/huggingface/lerobot/local/{repo_id.split('/')[-1]}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True, help="Official LeRobot v3 dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="OpenPI-ready output root")
    parser.add_argument("--repo-id", type=str, default="local/double_V1", help="Logical repo id for OpenPI")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place parquet files in the output dataset (default: symlink)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output-dir if it exists")
    args = parser.parse_args()
    repack(
        args.input_dir,
        args.output_dir,
        repo_id=args.repo_id,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
