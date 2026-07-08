#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Inspect the Pi0.5 training dataloader and dataset metadata.

Examples:
    python3 train/scripts/test_dataloader.py train/scripts/train_lora_4gpu.sh
    python3 train/scripts/test_dataloader.py --config train/config/lora.yaml --num-workers 0
"""

from __future__ import annotations

import argparse
import inspect
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "train" / "config" / "lora.yaml"
COMMON_SRC_ROOT = PROJECT_ROOT / "common" / "src"
TRAIN_SRC_ROOT = PROJECT_ROOT / "train" / "src"
LEROBOT_SRC = WORKSPACE_ROOT / "third_party" / "lerobot" / "src"

for path in (COMMON_SRC_ROOT, TRAIN_SRC_ROOT, LEROBOT_SRC):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print full diagnostic information for the Pi0.5 LeRobot dataloader."
    )
    parser.add_argument(
        "train_script",
        nargs="?",
        type=Path,
        help="Optional training shell script, e.g. train/scripts/train_lora_4gpu.sh.",
    )
    parser.add_argument(
        "--train-script",
        dest="train_script_flag",
        type=Path,
        default=None,
        help="Training shell script to inspect. Same as the positional train_script argument.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training YAML config. Overrides the config parsed from --train-script.",
    )
    parser.add_argument("--dataset-path", type=Path, default=None, help="Override data.dataset_path from config.")
    parser.add_argument("--instruction", type=str, default=None, help="Expected task/instruction string to compare.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size from config.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers from config.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override data.chunk_size from config.")
    parser.add_argument("--image-size", type=int, default=None, help="Override data.image_size from config.")
    parser.add_argument("--state-dim", type=int, default=None, help="Override data.state_dim from config.")
    parser.add_argument("--action-dim", type=int, default=None, help="Override data.action_dim from config.")
    parser.add_argument(
        "--disable-color-jitter",
        action="store_true",
        help="Disable online color jitter for deterministic inspection.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index to inspect in detail.")
    parser.add_argument("--num-batches", type=int, default=1, help="Number of dataloader batches to print.")
    parser.add_argument(
        "--image-patch-size",
        type=int,
        default=14,
        help="Patch size used only for reporting image patch counts, e.g. 224/14=16 patches per side.",
    )
    parser.add_argument(
        "--tactile-patch-size",
        type=int,
        default=16,
        help="Patch size used only for reporting tactile encoder patch counts when supported.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the dataloader.")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Build training normalizers before sampling. This mirrors training and may refresh stats.json.",
    )
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    train_script = args.train_script_flag or args.train_script
    script_context: dict[str, Any] = {}
    if train_script is not None:
        script_context = inspect_train_script(train_script)

    if args.config is not None:
        config_path = args.config.expanduser().resolve()
    elif "config_path" in script_context:
        config_path = Path(script_context["config_path"]).expanduser().resolve()
    else:
        config_path = DEFAULT_CONFIG.resolve()

    return config_path, script_context


def inspect_train_script(train_script: Path) -> dict[str, Any]:
    script_path = train_script.expanduser()
    if not script_path.is_absolute():
        candidates = [
            (Path.cwd() / script_path).resolve(),
            (PROJECT_ROOT / script_path).resolve(),
            (WORKSPACE_ROOT / script_path).resolve(),
        ]
        script_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    else:
        script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Training script does not exist: {script_path}")

    text = script_path.read_text(encoding="utf-8")
    script_dir = script_path.parent
    project_dir = (script_dir / "../..").resolve()
    variables: dict[str, str] = {
        "BASH_SOURCE[0]": str(script_path),
        "SCRIPT_DIR": str(script_dir),
        "PROJECT_DIR": str(project_dir),
    }

    for line in text.splitlines():
        parsed = parse_shell_assignment(line, variables)
        if parsed is None:
            continue
        key, value = parsed
        variables[key] = value

    config_path = variables.get("CONFIG_PATH")
    context: dict[str, Any] = {
        "train_script": script_path,
        "config_path": Path(config_path).expanduser().resolve() if config_path else None,
        "python_bin": variables.get("PYTHON_BIN"),
        "accelerate_bin": variables.get("ACCELERATE_BIN"),
        "cuda_visible_devices": variables.get("CUDA_VISIBLE_DEVICES"),
        "num_processes": variables.get("NUM_PROCESSES"),
        "main_process_port": variables.get("MAIN_PROCESS_PORT"),
    }
    return {key: value for key, value in context.items() if value is not None}


def parse_shell_assignment(line: str, variables: dict[str, str]) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.+)$", stripped)
    if match is None:
        return None
    key, raw_value = match.groups()
    value = raw_value.split("#", 1)[0].strip()
    if "$(" in value:
        return None
    value = value[:-1].rstrip() if value.endswith("\\") else value
    value = strip_shell_quotes(value)
    return key, expand_shell_value(value, variables)


def strip_shell_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def expand_shell_value(value: str, variables: dict[str, str]) -> str:
    def replace_default(match: re.Match[str]) -> str:
        return variables.get(match.group(1), match.group(2))

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]+)\}", replace_default, value)
    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: variables.get(m.group(1), m.group(0)), value)
    return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: variables.get(m.group(1), m.group(0)), value)


def cfg_value(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_kv(key: str, value: Any) -> None:
    print(f"{key:<34} {value}")


def describe_tensor(name: str, tensor: Any) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        print_kv(name, f"{type(tensor).__name__}: {summarize_value(tensor)}")
        return

    summary = f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
    if tensor.numel() > 0 and tensor.is_floating_point():
        summary += (
            f" min={tensor.min().item():.6f}"
            f" max={tensor.max().item():.6f}"
            f" mean={tensor.mean().item():.6f}"
        )
    elif tensor.numel() > 0:
        summary += f" min={tensor.min().item()} max={tensor.max().item()}"
    print_kv(name, summary)


def summarize_value(value: Any, *, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summarize_tasks(tasks: Any, *, limit: int = 8) -> list[str]:
    if tasks is None:
        return []
    if hasattr(tasks, "head"):
        return [summarize_value(row) for row in tasks.head(limit).to_dict(orient="records")]
    if isinstance(tasks, dict):
        return [f"{key}: {summarize_value(value)}" for key, value in list(tasks.items())[:limit]]
    if isinstance(tasks, (list, tuple)):
        return [summarize_value(value) for value in tasks[:limit]]
    return [summarize_value(tasks)]


def feature_shape(feature: Any) -> Any:
    if isinstance(feature, dict):
        return feature.get("shape", "<missing shape>")
    return getattr(feature, "shape", "<missing shape>")


def estimate_episode_lengths(dataset: Any) -> list[int]:
    meta = dataset.lerobot_dataset.meta
    lengths: list[int] = []
    for episode_index in range(getattr(meta, "total_episodes", 0)):
        episode = meta.episodes[episode_index]
        start = dataset._to_int(episode["dataset_from_index"])
        end = dataset._to_int(episode["dataset_to_index"])
        lengths.append(end - start)
    return lengths


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def print_train_script_summary(script_context: dict[str, Any]) -> None:
    if not script_context:
        return
    print_section("Training Script")
    for key in (
        "train_script",
        "config_path",
        "python_bin",
        "accelerate_bin",
        "cuda_visible_devices",
        "num_processes",
        "main_process_port",
    ):
        print_kv(key, script_context.get(key, "<not set>"))


def resolved_runtime_options(config: Any, args: argparse.Namespace, script_context: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config.data
    train_cfg = config.training
    return {
        "dataset_path": (args.dataset_path or data_cfg.resolved_dataset_path).expanduser().resolve(),
        "batch_size": args.batch_size or train_cfg.batch_size,
        "num_workers": cfg_value(data_cfg, "num_workers", 0) if args.num_workers is None else args.num_workers,
        "num_processes": parse_int(script_context.get("num_processes"), 1),
        "chunk_size": args.chunk_size or cfg_value(data_cfg, "chunk_size", 30),
        "image_size": args.image_size or cfg_value(data_cfg, "image_size", 224),
        "state_dim": args.state_dim or cfg_value(data_cfg, "state_dim", 26),
        "action_dim": args.action_dim or cfg_value(data_cfg, "action_dim", 14),
        "cameras": tuple(cfg_value(data_cfg, "cameras", ("top", "left_wrist", "right_wrist"))),
        "use_color_jitter": False if args.disable_color_jitter else cfg_value(data_cfg, "use_color_jitter", True),
        "use_tactile": cfg_value(data_cfg, "use_tactile", False),
    }


def print_config_summary(
    config: Any,
    args: argparse.Namespace,
    config_path: Path,
    script_context: dict[str, Any],
    options: dict[str, Any],
) -> None:
    train_cfg = config.training
    print_section("Config")
    print_kv("config_path", config_path)
    print_kv("run_name", config.logging.run_name)
    print_kv("dataset_path", options["dataset_path"])
    print_kv("instruction_arg", args.instruction or "<not provided>")
    print_kv("batch_size", options["batch_size"])
    print_kv("num_workers", options["num_workers"])
    print_kv("num_processes(script)", options["num_processes"])
    print_kv("gradient_accumulation_steps", train_cfg.gradient_accumulation_steps)
    print_kv("effective_batch_per_process", options["batch_size"] * train_cfg.gradient_accumulation_steps)
    print_kv(
        "effective_batch_global",
        options["batch_size"] * train_cfg.gradient_accumulation_steps * options["num_processes"],
    )
    print_kv("epochs", train_cfg.epochs)
    print_kv("checkpoint_freq_steps", cfg_value(train_cfg, "checkpoint_freq_steps", "<not set>"))
    print_kv("log_freq", cfg_value(config.logging, "log_freq", "<not set>"))
    print_kv("chunk_size/action_horizon", options["chunk_size"])
    print_kv("fps", cfg_value(config.data, "fps", "<not set>"))
    print_kv("image_size", options["image_size"])
    print_kv("use_color_jitter", options["use_color_jitter"])
    print_kv("use_tactile", options["use_tactile"])
    print_kv("state_dim(config)", options["state_dim"])
    print_kv("action_dim(config)", options["action_dim"])


def print_dataset_summary(dataset: Any, options: dict[str, Any], image_patch_size: int) -> None:
    meta = dataset.lerobot_dataset.meta
    episode_lengths = estimate_episode_lengths(dataset)
    strict_chunks = [max(length - dataset.chunk_size + 1, 0) for length in episode_lengths]
    padded_chunks = episode_lengths
    image_size = options["image_size"]
    image_patches_per_side = image_size // image_patch_size if image_size % image_patch_size == 0 else None
    dataloader_batches = math.ceil(len(dataset) / options["batch_size"])

    print_section("Dataset")
    print_kv("frames / dataset samples", len(dataset))
    print_kv("episodes", getattr(meta, "total_episodes", "<unknown>"))
    print_kv("episode_length_min/max", f"{min(episode_lengths)}/{max(episode_lengths)}" if episode_lengths else "n/a")
    print_kv("episode_length_first_10", episode_lengths[:10])
    print_kv("dataloader_batches_total", dataloader_batches)
    print_kv("dataloader_batches_per_process", math.ceil(dataloader_batches / max(options["num_processes"], 1)))
    print_kv("padded_action_chunks", sum(padded_chunks))
    print_kv("strict_non_padded_chunks", sum(strict_chunks))
    print_kv("chunk_size/action_horizon", dataset.chunk_size)
    print_kv("state_dim(dataset)", dataset.state_dim)
    print_kv("action_dim(dataset)", dataset.action_dim)
    print_kv("camera_keys(meta)", list(getattr(meta, "camera_keys", [])))
    print_kv("image_keys(mapped)", dataset.image_keys)
    print_kv("tactile_keys(mapped)", getattr(dataset, "tactile_keys", "<not supported by this codebase>"))
    if image_patches_per_side is None:
        print_kv("image_patch_grid", f"image_size={image_size} is not divisible by {image_patch_size}")
    else:
        print_kv(
            "image_patch_grid",
            f"{image_patches_per_side}x{image_patches_per_side} = {image_patches_per_side ** 2} patches / image",
        )


def print_epoch_step_plan(config: Any, dataset: Any, options: dict[str, Any]) -> None:
    total_batches = math.ceil(len(dataset) / options["batch_size"])
    batches_per_process = math.ceil(total_batches / max(options["num_processes"], 1))
    grad_accum = config.training.gradient_accumulation_steps
    real_steps_per_epoch = math.ceil(batches_per_process / grad_accum)
    if config.training.max_steps_per_epoch is not None:
        real_steps_per_epoch = min(real_steps_per_epoch, config.training.max_steps_per_epoch)
    micro_steps_per_epoch = min(batches_per_process, real_steps_per_epoch * grad_accum)

    print_section("Epoch Step Plan")
    print_kv("step_meaning", "real_step = optimizer.step count")
    print_kv("micro_step_meaning", "micro_step = dataloader iteration count per process")
    print_kv("total_batches_before_accelerate", total_batches)
    print_kv("batches_per_process", batches_per_process)
    print_kv("gradient_accumulation_steps", grad_accum)
    print_kv("real_steps_per_epoch", real_steps_per_epoch)
    print_kv("micro_steps_per_epoch", micro_steps_per_epoch)
    print_kv("micro_per_real", f"{micro_steps_per_epoch / real_steps_per_epoch:.2f}" if real_steps_per_epoch else "n/a")
    print("")
    print("epoch  real_step_start  real_step_end  real_step_delta  micro_step_start  micro_step_end  micro_step_delta")

    real_start = 0
    micro_start = 0
    for epoch in range(config.training.epochs):
        real_end = real_start + real_steps_per_epoch
        micro_end = micro_start + micro_steps_per_epoch
        print(
            f"{epoch:>5}  "
            f"{real_start:>15}  "
            f"{real_end:>13}  "
            f"{real_steps_per_epoch:>15}  "
            f"{micro_start:>16}  "
            f"{micro_end:>14}  "
            f"{micro_steps_per_epoch:>16}"
        )
        real_start = real_end
        micro_start = micro_end

    total_real_steps = real_steps_per_epoch * config.training.epochs
    total_micro_steps = micro_steps_per_epoch * config.training.epochs
    eff_batch_global = options["batch_size"] * options["num_processes"] * grad_accum
    total_frames_consumed = total_real_steps * eff_batch_global

    print("")
    print_section("Totals")
    print_kv("total_frames_in_dataset", len(dataset))
    print_kv("total_real_steps (所有epoch)", total_real_steps)
    print_kv("total_micro_steps (所有epoch)", total_micro_steps)
    print_kv("effective_batch_global", eff_batch_global)
    print_kv("total_frames_consumed", total_frames_consumed)


def print_feature_summary(dataset: Any) -> None:
    print_section("Features")
    for key, feature in sorted(dataset.lerobot_dataset.features.items()):
        print_kv(key, feature_shape(feature))


def print_task_summary(dataset: Any, sample: dict[str, Any], instruction: str | None) -> None:
    meta = dataset.lerobot_dataset.meta
    tasks = getattr(meta, "tasks", None)
    raw_row = dataset.dataset.hf_dataset[0] if len(dataset) else {}

    print_section("Tasks")
    print_kv("instruction_arg", instruction or "<not provided>")
    print_kv("sample_task_resolved", sample.get("task"))
    print_kv("raw_task", raw_row.get("task", "<missing>"))
    print_kv("raw_task_index", raw_row.get("task_index", "<missing>"))
    task_rows = summarize_tasks(tasks)
    print_kv("meta_tasks_type", type(tasks).__name__ if tasks is not None else "<missing>")
    print_kv("meta_tasks_preview_count", len(task_rows))
    for idx, row in enumerate(task_rows):
        print_kv(f"meta_tasks[{idx}]", row)


def print_sample_summary(dataset: Any, sample_index: int) -> dict[str, Any]:
    sample_index = min(max(sample_index, 0), len(dataset) - 1)
    sample = dataset[sample_index]

    print_section("Single Sample")
    print_kv("sample_index", sample_index)
    print_kv("sample_keys", sorted(sample.keys()))
    for key, value in sample.items():
        describe_tensor(key, value)
    return sample


def print_batch_summary(batch: dict[str, Any], batch_index: int) -> None:
    print_section(f"Batch {batch_index}")
    print_kv("batch_keys", sorted(batch.keys()))
    for key, value in batch.items():
        describe_tensor(key, value)
    tasks = batch.get("task")
    if tasks is not None:
        task_counter = Counter(tasks)
        print_kv("task_unique_count", len(task_counter))
        print_kv("task_counts", dict(task_counter))


def print_tactile_patch_summary(config: Any, tactile_patch_size: int) -> None:
    model_use_tactile = cfg_value(config.model, "use_tactile", False)
    tactile_cfg = cfg_value(config.model, "tactile", None)
    if not model_use_tactile or tactile_cfg is None:
        return
    height, width = tactile_cfg.image_size
    print_section("Tactile Encoder Patches")
    print_kv("tactile_encoder", tactile_cfg.encoder_name)
    print_kv("tactile_resize", (height, width))
    print_kv("tactile_num_tokens", tactile_cfg.num_tokens)
    if height % tactile_patch_size or width % tactile_patch_size:
        print_kv("tactile_patch_grid", f"{height}x{width} is not divisible by {tactile_patch_size}")
        return
    h_patches = height // tactile_patch_size
    w_patches = width // tactile_patch_size
    print_kv("tactile_patch_grid", f"{h_patches}x{w_patches} = {h_patches * w_patches} patches / side-pair image")


def dataset_kwargs(dataset_cls: type, options: dict[str, Any], *, normalizers: tuple[Any, Any] | None = None) -> dict[str, Any]:
    signature = inspect.signature(dataset_cls)
    kwargs: dict[str, Any] = {
        "dataset_path": options["dataset_path"],
        "chunk_size": options["chunk_size"],
        "use_color_jitter": options["use_color_jitter"],
        "image_size": options["image_size"],
        "state_dim": options["state_dim"],
        "action_dim": options["action_dim"],
    }
    if "cameras" in signature.parameters:
        kwargs["cameras"] = options["cameras"]
    if normalizers is not None:
        kwargs["state_normalizer"], kwargs["action_normalizer"] = normalizers
    if "use_tactile" in signature.parameters:
        kwargs["use_tactile"] = options["use_tactile"]
    return kwargs


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from pi05.common.config.schema import load_experiment_config
    from pi05.common.data.normalization import build_state_action_normalizers
    from pi05.train.data.dataset import Pi05LeRobotDataset

    config_path, script_context = resolve_config_path(args)
    config = load_experiment_config(config_path)
    options = resolved_runtime_options(config, args, script_context)
    dataset_path = options["dataset_path"]

    print_train_script_summary(script_context)
    print_config_summary(config, args, config_path, script_context, options)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    bootstrap_dataset = Pi05LeRobotDataset(**dataset_kwargs(Pi05LeRobotDataset, options))
    normalizers = build_state_action_normalizers(bootstrap_dataset.dataset) if args.normalize else None
    dataset = Pi05LeRobotDataset(**dataset_kwargs(Pi05LeRobotDataset, options, normalizers=normalizers))

    print_dataset_summary(dataset, options, args.image_patch_size)
    print_epoch_step_plan(config, dataset, options)
    print_feature_summary(dataset)
    sample = print_sample_summary(dataset, args.sample_index)
    print_task_summary(dataset, sample, args.instruction)
    print_tactile_patch_summary(config, args.tactile_patch_size)

    dataloader = DataLoader(
        dataset,
        batch_size=options["batch_size"],
        shuffle=args.shuffle,
        num_workers=options["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    for batch_index, batch in enumerate(dataloader):
        if batch_index >= args.num_batches:
            break
        print_batch_summary(batch, batch_index)


if __name__ == "__main__":
    main()
