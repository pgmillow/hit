#!/usr/bin/env python3
"""Audit local LeRobot-style datasets before hybrid merging.

The script is intentionally read-only.  It checks metadata, parquet schemas,
sampled frame data, vector validity, image references/payloads, and basic
episode continuity.  Use --full-scan when you want every parquet file checked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing dependency guard
    missing = exc.name or str(exc)
    print(
        f"Missing Python dependency: {missing}\n"
        "Run this script inside the training environment, for example:\n"
        "  source ~/miniconda3/etc/profile.d/conda.sh\n"
        "  conda activate lerobot-pi\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


DEFAULT_DATASETS = (
    Path("/home/gexudi/hitdata/input/dataV3"),
    Path("/home/gexudi/hitdata/input/dataV2"),
    Path("/home/gexudi/hitdata/input/dataV1"),
)

REQUIRED_IMAGE_KEYS = (
    "observation.images.top",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
OPTIONAL_IMAGE_KEYS = (
    "observation.images.left_tactile",
    "observation.images.right_tactile",
)
REQUIRED_VECTOR_DIMS = {
    "observation.state": 26,
    "action": 14,
}
REQUIRED_COLUMNS = (
    "timestamp",
    "frame_index",
    "episode_index",
    "task_index",
    *REQUIRED_IMAGE_KEYS,
    "observation.state",
    "action",
)
EXPECTED_FPS = 60
EXPECTED_IMAGE_SHAPE = (224, 224, 3)


@dataclass
class Issue:
    level: str
    message: str


@dataclass
class FileCheck:
    path: str
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


@dataclass
class DatasetReport:
    input_path: str
    dataset_root: str
    status: str = "PASS"
    complete_lerobot: bool = False
    parquet_files: int = 0
    parquet_rows: int = 0
    sampled_files: int = 0
    info_summary: dict[str, Any] = field(default_factory=dict)
    task_preview: list[str] = field(default_factory=list)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    numeric_summary: dict[str, Any] = field(default_factory=dict)
    episode_summary: dict[str, Any] = field(default_factory=dict)
    file_checks: list[FileCheck] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_DATASETS),
        help="Dataset roots to audit. Defaults to dataV3 dataV2 dataV1.",
    )
    parser.add_argument(
        "--sample-files",
        type=int,
        default=8,
        help="Number of parquet files to read per dataset when not using --full-scan.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=64,
        help="Rows to inspect per sampled parquet file.",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Read every parquet file. This can take a while for large datasets.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full audit report as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 when any ERROR is found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports: list[DatasetReport] = []

    for input_path in args.datasets:
        for root in discover_dataset_roots(input_path):
            reports.append(audit_dataset(input_path, root, args))

    print_human_report(reports)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([report_to_json(report) for report in reports], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nWrote JSON report: {args.json_out}")

    has_errors = any(issue.level == "ERROR" for report in reports for issue in report.issues)
    return 2 if args.strict and has_errors else 0


def discover_dataset_roots(input_path: Path) -> list[Path]:
    path = input_path.expanduser().resolve()
    if (path / "meta" / "info.json").exists():
        return [path]

    nested = sorted({candidate.parent.parent for candidate in path.glob("**/meta/info.json")})
    if nested:
        return nested

    return [path]


def audit_dataset(input_path: Path, root: Path, args: argparse.Namespace) -> DatasetReport:
    report = DatasetReport(input_path=str(input_path), dataset_root=str(root))

    if not root.exists():
        add_issue(report, "ERROR", f"Path does not exist: {root}")
        finalize_status(report)
        return report

    info = load_info(root, report)
    report.complete_lerobot = info is not None
    if info is not None:
        audit_info(info, report)
        audit_tasks(root, report)
        audit_episode_metadata(root, info, report)
    else:
        add_issue(report, "ERROR", "Missing meta/info.json; this is not a complete LeRobot dataset root.")

    data_files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if not data_files:
        add_issue(report, "ERROR", "No data parquet files found under data/chunk-*/*.parquet.")
        finalize_status(report)
        return report

    report.parquet_files = len(data_files)
    report.parquet_rows = count_parquet_rows(data_files, report)

    if info is not None:
        expected_frames = safe_int(info.get("total_frames"))
        if expected_frames is not None and expected_frames != report.parquet_rows:
            add_issue(
                report,
                "ERROR",
                f"info.json total_frames={expected_frames} but parquet row count={report.parquet_rows}.",
            )

    files_to_read = data_files if args.full_scan else pick_sample_files(data_files, args.sample_files)
    report.sampled_files = len(files_to_read)

    numeric_acc = NumericAccumulator()
    seen_episode_counts: dict[int, int] = {}
    seen_episode_last_frame: dict[int, int] = {}
    seen_episode_last_ts: dict[int, float] = {}
    all_columns: set[str] = set()

    for file_path in files_to_read:
        check = audit_data_file(
            file_path=file_path,
            sample_rows=args.sample_rows,
            numeric_acc=numeric_acc,
            seen_episode_counts=seen_episode_counts,
            seen_episode_last_frame=seen_episode_last_frame,
            seen_episode_last_ts=seen_episode_last_ts,
        )
        all_columns.update(check.columns)
        report.file_checks.append(check)
        for issue in check.issues:
            add_issue(report, issue.level, f"{rel(file_path, root)}: {issue.message}")

    report.feature_summary = summarize_features(info, all_columns)
    report.numeric_summary = numeric_acc.summary()
    if seen_episode_counts:
        lengths = list(seen_episode_counts.values())
        report.episode_summary.update(
            {
                "sampled_episodes": len(seen_episode_counts),
                "sampled_episode_min_rows": int(min(lengths)),
                "sampled_episode_max_rows": int(max(lengths)),
            }
        )

    finalize_status(report)
    return report


def load_info(root: Path, report: DatasetReport) -> dict[str, Any] | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - keep audit alive
        add_issue(report, "ERROR", f"Could not parse meta/info.json: {exc}")
        return None
    return info if isinstance(info, dict) else None


def audit_info(info: dict[str, Any], report: DatasetReport) -> None:
    report.info_summary = {
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
    }

    fps = safe_int(info.get("fps"))
    if fps != EXPECTED_FPS:
        add_issue(report, "ERROR", f"Expected fps={EXPECTED_FPS}, got {info.get('fps')!r}.")

    features = info.get("features")
    if not isinstance(features, dict):
        add_issue(report, "ERROR", "info.json missing features mapping.")
        return

    for key in REQUIRED_IMAGE_KEYS:
        feature = features.get(key)
        if not isinstance(feature, dict):
            add_issue(report, "ERROR", f"Missing required image feature: {key}")
            continue
        if feature.get("dtype") != "image":
            add_issue(report, "ERROR", f"{key} dtype should be image, got {feature.get('dtype')!r}.")
        shape = tuple(feature.get("shape") or ())
        if shape != EXPECTED_IMAGE_SHAPE:
            add_issue(report, "ERROR", f"{key} shape should be {EXPECTED_IMAGE_SHAPE}, got {shape}.")

    for key, expected_dim in REQUIRED_VECTOR_DIMS.items():
        feature = features.get(key)
        if not isinstance(feature, dict):
            add_issue(report, "ERROR", f"Missing required vector feature: {key}")
            continue
        shape = tuple(feature.get("shape") or ())
        dim = product(shape)
        if dim != expected_dim:
            add_issue(report, "ERROR", f"{key} dim should be {expected_dim}, got shape={shape}.")
        if feature.get("dtype") not in {"float32", "float64"}:
            add_issue(report, "WARN", f"{key} dtype expected float32/float64, got {feature.get('dtype')!r}.")


def audit_tasks(root: Path, report: DatasetReport) -> None:
    tasks_path = root / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        add_issue(report, "ERROR", "Missing meta/tasks.parquet.")
        return
    try:
        table = pq.read_table(tasks_path)
        df = table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        add_issue(report, "ERROR", f"Could not read meta/tasks.parquet: {exc}")
        return

    if df.empty:
        add_issue(report, "ERROR", "tasks.parquet is empty.")
        return

    preview = df.head(5).copy()
    report.task_preview = [
        f"index={index!r}, row={row}"
        for index, row in preview.to_dict(orient="index").items()
    ]
    has_text_column = any(column in df.columns for column in ("task", "tasks", "description"))
    has_text_index = df.index.name in {"task", "tasks", "description"} and all(
        isinstance(value, str) and value.strip() for value in df.index.tolist()
    )
    if not has_text_column and not has_text_index:
        add_issue(report, "WARN", "tasks.parquet does not expose obvious text task labels.")


def audit_episode_metadata(root: Path, info: dict[str, Any], report: DatasetReport) -> None:
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        add_issue(report, "ERROR", "Missing meta/episodes parquet files.")
        return

    try:
        episode_df = pd.concat((pq.read_table(path).to_pandas() for path in episode_files), ignore_index=True)
    except Exception as exc:  # noqa: BLE001
        add_issue(report, "ERROR", f"Could not read meta/episodes parquet files: {exc}")
        return

    total_episodes = safe_int(info.get("total_episodes"))
    if total_episodes is not None and len(episode_df) != total_episodes:
        add_issue(report, "ERROR", f"info total_episodes={total_episodes} but meta/episodes rows={len(episode_df)}.")

    required = {"episode_index", "dataset_from_index", "dataset_to_index", "length"}
    missing = required - set(episode_df.columns)
    if missing:
        add_issue(report, "ERROR", f"meta/episodes missing columns: {sorted(missing)}")
        return

    indices = episode_df["episode_index"].astype(int).to_numpy()
    expected = np.arange(len(indices), dtype=int)
    if not np.array_equal(indices, expected):
        add_issue(report, "ERROR", "episode_index in meta/episodes is not contiguous from 0.")

    starts = episode_df["dataset_from_index"].astype(int).to_numpy()
    ends = episode_df["dataset_to_index"].astype(int).to_numpy()
    lengths = episode_df["length"].astype(int).to_numpy()
    if np.any(ends <= starts):
        add_issue(report, "ERROR", "Found episode with dataset_to_index <= dataset_from_index.")
    if np.any((ends - starts) != lengths):
        add_issue(report, "ERROR", "Found episode where length != dataset_to_index - dataset_from_index.")
    if len(starts) > 1 and not np.all(starts[1:] == ends[:-1]):
        add_issue(report, "ERROR", "Episode dataset ranges are not contiguous.")
    if np.any(lengths < 30):
        add_issue(report, "WARN", f"{int(np.sum(lengths < 30))} episodes are shorter than chunk_size=30.")

    report.episode_summary.update(
        {
            "metadata_episodes": int(len(episode_df)),
            "metadata_min_length": int(lengths.min()) if len(lengths) else 0,
            "metadata_max_length": int(lengths.max()) if len(lengths) else 0,
            "metadata_mean_length": float(np.mean(lengths)) if len(lengths) else 0.0,
        }
    )


def count_parquet_rows(data_files: list[Path], report: DatasetReport) -> int:
    total = 0
    for file_path in data_files:
        try:
            total += pq.ParquetFile(file_path).metadata.num_rows
        except Exception as exc:  # noqa: BLE001
            add_issue(report, "ERROR", f"Could not read parquet metadata for {file_path}: {exc}")
    return total


def pick_sample_files(data_files: list[Path], sample_files: int) -> list[Path]:
    if sample_files <= 0 or len(data_files) <= sample_files:
        return data_files
    indices = np.linspace(0, len(data_files) - 1, num=sample_files, dtype=int)
    return [data_files[int(idx)] for idx in indices]


def audit_data_file(
    *,
    file_path: Path,
    sample_rows: int,
    numeric_acc: "NumericAccumulator",
    seen_episode_counts: dict[int, int],
    seen_episode_last_frame: dict[int, int],
    seen_episode_last_ts: dict[int, float],
) -> FileCheck:
    check = FileCheck(path=str(file_path))
    try:
        parquet_file = pq.ParquetFile(file_path)
        check.rows = parquet_file.metadata.num_rows
        check.columns = parquet_file.schema_arrow.names
    except Exception as exc:  # noqa: BLE001
        check.issues.append(Issue("ERROR", f"Could not open parquet: {exc}"))
        return check

    missing = sorted(set(REQUIRED_COLUMNS) - set(check.columns))
    if missing:
        check.issues.append(Issue("ERROR", f"Missing required columns: {missing}"))

    columns = [column for column in REQUIRED_COLUMNS if column in check.columns]
    if not columns or check.rows == 0:
        if check.rows == 0:
            check.issues.append(Issue("ERROR", "Parquet file has zero rows."))
        return check

    try:
        table = pq.read_table(file_path, columns=columns)
        df = table.to_pandas()
    except Exception as exc:  # noqa: BLE001
        check.issues.append(Issue("ERROR", f"Could not read required columns: {exc}"))
        return check

    if sample_rows > 0 and len(df) > sample_rows:
        df = df.iloc[:sample_rows].copy()

    audit_index_columns(df, check, seen_episode_counts, seen_episode_last_frame, seen_episode_last_ts)
    audit_vectors(df, check, numeric_acc)
    audit_images(df, check)
    return check


def audit_index_columns(
    df: pd.DataFrame,
    check: FileCheck,
    seen_episode_counts: dict[int, int],
    seen_episode_last_frame: dict[int, int],
    seen_episode_last_ts: dict[int, float],
) -> None:
    for column in ("timestamp", "frame_index", "episode_index", "task_index"):
        if column not in df:
            return
        if df[column].isna().any():
            check.issues.append(Issue("ERROR", f"{column} contains null values."))

    if "timestamp" in df:
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(timestamps).all():
            check.issues.append(Issue("ERROR", "timestamp contains NaN or Inf."))

    if {"episode_index", "frame_index"}.issubset(df.columns):
        for episode, group in df.groupby("episode_index", sort=False):
            episode_idx = int(episode)
            frames = pd.to_numeric(group["frame_index"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(frames).all():
                check.issues.append(Issue("ERROR", f"frame_index has non-finite values in episode {episode_idx}."))
                continue
            frames_int = frames.astype(int)
            if len(frames_int) > 1 and np.any(np.diff(frames_int) <= 0):
                check.issues.append(Issue("ERROR", f"frame_index is not strictly increasing in episode {episode_idx}."))
            if episode_idx in seen_episode_last_frame and frames_int[0] <= seen_episode_last_frame[episode_idx]:
                check.issues.append(Issue("ERROR", f"episode {episode_idx} frame_index went backwards across files."))
            seen_episode_last_frame[episode_idx] = int(frames_int[-1])
            seen_episode_counts[episode_idx] = seen_episode_counts.get(episode_idx, 0) + len(frames_int)

    if {"episode_index", "timestamp"}.issubset(df.columns):
        for episode, group in df.groupby("episode_index", sort=False):
            episode_idx = int(episode)
            timestamps = pd.to_numeric(group["timestamp"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(timestamps).all():
                continue
            if len(timestamps) > 1 and np.any(np.diff(timestamps) < -1e-6):
                check.issues.append(Issue("ERROR", f"timestamp decreases in episode {episode_idx}."))
            if episode_idx in seen_episode_last_ts and timestamps[0] < seen_episode_last_ts[episode_idx] - 1e-6:
                check.issues.append(Issue("ERROR", f"episode {episode_idx} timestamp went backwards across files."))
            seen_episode_last_ts[episode_idx] = float(timestamps[-1])


def audit_vectors(df: pd.DataFrame, check: FileCheck, numeric_acc: "NumericAccumulator") -> None:
    for key, expected_dim in REQUIRED_VECTOR_DIMS.items():
        if key not in df:
            continue
        vectors = []
        bad_dim = 0
        conversion_errors = 0
        for value in df[key].tolist():
            try:
                vector = np.asarray(value, dtype=np.float32).reshape(-1)
            except Exception:  # noqa: BLE001
                conversion_errors += 1
                continue
            if vector.size != expected_dim:
                bad_dim += 1
                continue
            vectors.append(vector)

        if conversion_errors:
            check.issues.append(Issue("ERROR", f"{key}: {conversion_errors} values could not be converted to float."))
        if bad_dim:
            check.issues.append(Issue("ERROR", f"{key}: {bad_dim} rows have wrong dim; expected {expected_dim}."))
        if not vectors:
            check.issues.append(Issue("ERROR", f"{key}: no valid vectors sampled."))
            continue

        array = np.stack(vectors)
        if not np.isfinite(array).all():
            check.issues.append(Issue("ERROR", f"{key}: contains NaN or Inf."))
        numeric_acc.update(key, array)


def audit_images(df: pd.DataFrame, check: FileCheck) -> None:
    for key in REQUIRED_IMAGE_KEYS:
        if key not in df:
            continue
        null_count = int(df[key].isna().sum()) if hasattr(df[key], "isna") else 0
        if null_count:
            check.issues.append(Issue("ERROR", f"{key}: {null_count} null image values sampled."))
        descriptions = [describe_image_value(value) for value in df[key].head(5).tolist()]
        bad = [desc for desc in descriptions if desc.startswith("bad:")]
        if bad:
            check.issues.append(Issue("WARN", f"{key}: suspicious image samples: {bad[:2]}"))


def describe_image_value(value: Any) -> str:
    if value is None:
        return "bad:null"
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return "bytes"
        if value.get("path") is not None:
            return f"path:{value.get('path')}"
        return f"dict:{sorted(value.keys())}"
    if isinstance(value, (bytes, bytearray)):
        return "bytes"
    if isinstance(value, str):
        return f"path:{value}"
    if hasattr(value, "shape"):
        return f"array:{tuple(value.shape)}"
    return type(value).__name__


def summarize_features(info: dict[str, Any] | None, sampled_columns: set[str]) -> dict[str, Any]:
    if info is None or not isinstance(info.get("features"), dict):
        return {"sampled_columns": sorted(sampled_columns)}

    features = info["features"]
    return {
        "required_images": {key: features.get(key) for key in REQUIRED_IMAGE_KEYS},
        "optional_images_present": [key for key in OPTIONAL_IMAGE_KEYS if key in features],
        "vectors": {key: features.get(key) for key in REQUIRED_VECTOR_DIMS},
        "sampled_columns": sorted(sampled_columns),
    }


class NumericAccumulator:
    def __init__(self) -> None:
        self._stats: dict[str, dict[str, Any]] = {}

    def update(self, key: str, array: np.ndarray) -> None:
        stats = self._stats.setdefault(
            key,
            {
                "rows": 0,
                "dim": int(array.shape[-1]),
                "min": np.full(array.shape[-1], np.inf, dtype=np.float64),
                "max": np.full(array.shape[-1], -np.inf, dtype=np.float64),
                "sum": np.zeros(array.shape[-1], dtype=np.float64),
            },
        )
        stats["rows"] += int(array.shape[0])
        stats["min"] = np.minimum(stats["min"], np.nanmin(array, axis=0))
        stats["max"] = np.maximum(stats["max"], np.nanmax(array, axis=0))
        stats["sum"] += np.nansum(array, axis=0)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, stats in self._stats.items():
            rows = max(1, int(stats["rows"]))
            mean = stats["sum"] / rows
            result[key] = {
                "sampled_rows": int(stats["rows"]),
                "dim": int(stats["dim"]),
                "min_first5": round_list(stats["min"][:5]),
                "max_first5": round_list(stats["max"][:5]),
                "mean_first5": round_list(mean[:5]),
            }
        return result


def add_issue(report: DatasetReport, level: str, message: str) -> None:
    report.issues.append(Issue(level=level, message=message))


def finalize_status(report: DatasetReport) -> None:
    levels = {issue.level for issue in report.issues}
    if "ERROR" in levels:
        report.status = "FAIL"
    elif "WARN" in levels:
        report.status = "WARN"
    else:
        report.status = "PASS"


def print_human_report(reports: list[DatasetReport]) -> None:
    for report in reports:
        print(f"\n=== {report.dataset_root} ===")
        print(f"status             : {report.status}")
        print(f"complete_lerobot   : {report.complete_lerobot}")
        print(f"parquet_files      : {report.parquet_files}")
        print(f"parquet_rows       : {report.parquet_rows}")
        print(f"sampled_files      : {report.sampled_files}")
        if report.info_summary:
            print(f"info               : {report.info_summary}")
        if report.episode_summary:
            print(f"episodes           : {report.episode_summary}")
        if report.task_preview:
            print("tasks              :")
            for task in report.task_preview:
                print(f"  - {task}")
        if report.numeric_summary:
            print("numeric            :")
            for key, summary in report.numeric_summary.items():
                print(f"  - {key}: {summary}")
        if report.issues:
            print("issues             :")
            for issue in report.issues:
                print(f"  [{issue.level}] {issue.message}")
        else:
            print("issues             : none")

    print("\n=== Summary ===")
    for report in reports:
        print(f"{report.status:<5} {report.dataset_root}")


def report_to_json(report: DatasetReport) -> dict[str, Any]:
    data = asdict(report)
    return data


def safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def product(shape: tuple[Any, ...]) -> int | None:
    if not shape:
        return None
    try:
        value = 1
        for item in shape:
            value *= int(item)
        return value
    except (TypeError, ValueError):
        return None


def round_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
