#!/usr/bin/env python3
"""Merge V1/V2/V3 LeRobot datasets into one RGB-only hybrid dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCES = (
    Path("/home/gexudi/hitdata/input/dataV1/data_mcap1_water_long"),
    Path("/home/gexudi/hitdata/input/dataV2"),
    Path("/home/gexudi/hitdata/input/dataV3"),
)
DEFAULT_OUTPUT = Path("/home/gexudi/hitdata/input/dataHybrid123")
TASK_TEXT = "pour water from the bottle on the right into the cup on the left with the right hand"
KEEP_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
    "observation.state",
    "action",
)
VECTOR_DIMS = {
    "observation.state": 26,
    "action": 14,
}


@dataclass
class SourcePlan:
    root: Path
    files: list[Path]
    total_episodes: int
    total_frames: int


class VectorStats:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.count = 0
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"Expected vector batch shape (*, {self.dim}), got {values.shape}")
        if values.shape[0] == 0:
            return

        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))
        for row in values:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    def as_json(self) -> dict[str, list[Any]]:
        variance = self.m2 / self.count if self.count > 0 else np.zeros(self.dim, dtype=np.float64)
        std = np.sqrt(np.maximum(variance, 0.0))
        return {
            "min": self.min.astype(float).tolist(),
            "max": self.max.astype(float).tolist(),
            "mean": self.mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "count": [int(self.count)],
        }


class ScalarStats:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update_range(self, start: int, count: int) -> None:
        self.values.extend(float(value) for value in range(start, start + count))

    def update_constant(self, value: int, count: int) -> None:
        self.values.extend([float(value)] * count)

    def as_json(self) -> dict[str, list[Any]]:
        values = np.asarray(self.values, dtype=np.float64)
        return {
            "min": [float(values.min())],
            "max": [float(values.max())],
            "mean": [float(values.mean())],
            "std": [float(values.std())],
            "count": [int(values.size)],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Hybrid dataset output root.")
    parser.add_argument("--sources", nargs="*", type=Path, default=list(DEFAULT_SOURCES), help="Source dataset roots.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the merge plan without writing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated data/meta files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = [build_source_plan(path) for path in args.sources]
    total_episodes = sum(source.total_episodes for source in sources)
    total_frames = sum(source.total_frames for source in sources)
    print(f"[merge] sources={len(sources)} episodes={total_episodes} frames={total_frames}", flush=True)
    for source in sources:
        print(
            f"[merge] source {source.root}: files={len(source.files)} "
            f"episodes={source.total_episodes} frames={source.total_frames}",
            flush=True,
        )

    if args.dry_run:
        return 0

    prepare_output(args.output, overwrite=args.overwrite)
    rgb_schema_metadata = load_rgb_schema_metadata(sources[0].files[0])
    vector_stats = {key: VectorStats(dim) for key, dim in VECTOR_DIMS.items()}
    index_stats = ScalarStats()
    episode_stats = ScalarStats()

    global_episode = 0
    global_index = 0
    written_files: list[dict[str, Any]] = []

    for source in sources:
        for source_file in source.files:
            table = pq.read_table(source_file, columns=list(KEEP_COLUMNS))
            row_count = table.num_rows
            source_episodes = unique_ints(table["episode_index"])
            if len(source_episodes) != 1:
                raise ValueError(f"{source_file} contains multiple episode_index values: {source_episodes}")

            table = replace_column(table, "episode_index", pa.array([global_episode] * row_count, type=pa.int64()))
            table = replace_column(table, "index", pa.array(range(global_index, global_index + row_count), type=pa.int64()))
            table = replace_column(table, "task_index", pa.array([0] * row_count, type=pa.int64()))
            table = table.replace_schema_metadata(rgb_schema_metadata)

            for key, stats in vector_stats.items():
                stats.update(fixed_size_list_to_numpy(table[key], VECTOR_DIMS[key]))
            index_stats.update_range(global_index, row_count)
            episode_stats.update_constant(global_episode, row_count)

            output_file = args.output / "data" / "chunk-000" / f"file-{global_episode:03d}.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # Image columns are already PNG-compressed bytes; Parquet compression
            # spends CPU for little size benefit, so keep output uncompressed.
            pq.write_table(table, output_file, compression=None)
            written_files.append(
                {
                    "source": str(source_file),
                    "output": str(output_file),
                    "episode_index": global_episode,
                    "rows": row_count,
                    "index_from": global_index,
                    "index_to": global_index + row_count,
                }
            )

            global_episode += 1
            global_index += row_count
            if global_episode % 25 == 0 or global_episode == total_episodes:
                print(
                    f"[merge] wrote episodes={global_episode}/{total_episodes} "
                    f"frames={global_index}/{total_frames}",
                    flush=True,
                )

    if global_episode != total_episodes or global_index != total_frames:
        raise RuntimeError(
            f"Internal count mismatch: wrote episodes={global_episode}/{total_episodes}, "
            f"frames={global_index}/{total_frames}"
        )

    write_metadata(
        output=args.output,
        template_info_path=sources[0].root / "meta" / "info.json",
        total_episodes=total_episodes,
        total_frames=total_frames,
        data_files_size_mb=directory_size_mb(args.output / "data"),
        stats={
            "index": index_stats.as_json(),
            "episode_index": episode_stats.as_json(),
            **{key: stats.as_json() for key, stats in vector_stats.items()},
        },
    )
    write_episode_metadata(args.output, written_files)
    write_report(args.output, written_files, sources)
    print(f"[merge] done: {args.output}", flush=True)
    return 0


def build_source_plan(root: Path) -> SourcePlan:
    root = root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot meta/info.json: {root}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root / 'data'}")
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    if len(files) != total_episodes:
        raise ValueError(f"{root}: parquet files={len(files)} does not match total_episodes={total_episodes}")
    actual_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if actual_rows != total_frames:
        raise ValueError(f"{root}: parquet rows={actual_rows} does not match total_frames={total_frames}")
    return SourcePlan(root=root, files=files, total_episodes=total_episodes, total_frames=total_frames)


def prepare_output(output: Path, *, overwrite: bool) -> None:
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    generated = [output / "data", output / "meta", output / "merge_report.json"]
    existing = [path for path in generated if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Output already has generated files. Re-run with --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def load_rgb_schema_metadata(template_file: Path) -> dict[bytes, bytes] | None:
    table = pq.read_table(template_file, columns=list(KEEP_COLUMNS))
    return table.schema.metadata


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        raise KeyError(name)
    return table.set_column(idx, name, values)


def unique_ints(column: pa.ChunkedArray) -> list[int]:
    values = column.combine_chunks().to_pylist()
    return sorted({int(value) for value in values})


def fixed_size_list_to_numpy(column: pa.ChunkedArray, dim: int) -> np.ndarray:
    combined = column.combine_chunks()
    values = combined.values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(-1, dim)


def write_metadata(
    *,
    output: Path,
    template_info_path: Path,
    total_episodes: int,
    total_frames: int,
    data_files_size_mb: float,
    stats: dict[str, Any],
) -> None:
    meta_dir = output / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads(template_info_path.read_text(encoding="utf-8"))
    info["features"] = {key: info["features"][key] for key in KEEP_COLUMNS}
    info["total_episodes"] = int(total_episodes)
    info["total_frames"] = int(total_frames)
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{total_episodes}"}
    info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    info["video_path"] = None
    info["data_files_size_in_mb"] = int(math.ceil(data_files_size_mb))
    info["video_files_size_in_mb"] = 0
    (meta_dir / "info.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")

    tasks = pd.DataFrame({"task_index": [0]}, index=pd.Index([TASK_TEXT], name="task"))
    tasks.to_parquet(meta_dir / "tasks.parquet")

    (meta_dir / "stats.json").write_text(json.dumps(stats, indent=4, ensure_ascii=False), encoding="utf-8")


def write_report(output: Path, written_files: list[dict[str, Any]], sources: list[SourcePlan]) -> None:
    report = {
        "output": str(output),
        "task": TASK_TEXT,
        "rgb_only_columns": list(KEEP_COLUMNS),
        "dropped_columns": [
            "observation.images.left_tactile",
            "observation.images.right_tactile",
        ],
        "sources": [
            {
                "root": str(source.root),
                "episodes": source.total_episodes,
                "frames": source.total_frames,
            }
            for source in sources
        ],
        "files": written_files,
    }
    (output / "merge_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def write_episode_metadata(output: Path, written_files: list[dict[str, Any]]) -> None:
    episodes = pd.DataFrame(
        {
            "episode_index": [entry["episode_index"] for entry in written_files],
            "tasks": [[TASK_TEXT] for _ in written_files],
            "length": [entry["rows"] for entry in written_files],
            "data/chunk_index": [0 for _ in written_files],
            "data/file_index": [entry["episode_index"] for entry in written_files],
            "dataset_from_index": [entry["index_from"] for entry in written_files],
            "dataset_to_index": [entry["index_to"] for entry in written_files],
            "meta/episodes/chunk_index": [0 for _ in written_files],
            "meta/episodes/file_index": [0 for _ in written_files],
        }
    )
    episodes_path = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(episodes_path, index=False)


def directory_size_mb(path: Path) -> float:
    total = 0
    for file_path in path.glob("**/*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total / (1024 * 1024)


if __name__ == "__main__":
    raise SystemExit(main())
