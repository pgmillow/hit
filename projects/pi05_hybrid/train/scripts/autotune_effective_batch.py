#!/usr/bin/env python3
"""Auto-tune the largest stable effective batch on multi-GPU.

Search strategy:
1) Binary search max per-GPU micro batch (`training.batch_size`) under bf16.
2) Sweep gradient accumulation candidates to maximize effective batch.
3) Write a tuned YAML config for normal training use.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "train" / "config" / "lora.yaml"
DEFAULT_OUTPUT_CONFIG_PATH = PROJECT_ROOT / "train" / "config" / "lora.autotuned.yaml"
AUTOTUNE_LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "autotune"
OOM_PATTERNS = (
    r"out of memory",
    r"cuda error: out of memory",
    r"cublas_status_alloc_failed",
    r"cuda out of memory",
    r"hip out of memory",
)


@dataclass
class TrialResult:
    ok: bool
    is_oom: bool
    return_code: int
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-tune max stable effective batch for PI05 training.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Base YAML config to clone for probing.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=DEFAULT_OUTPUT_CONFIG_PATH,
        help="Where to write tuned YAML after probing.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
        help="GPU list for probing, e.g. 0,1,2,3.",
    )
    parser.add_argument("--num-gpus", type=int, default=4, help="Accelerate num processes.")
    parser.add_argument(
        "--max-micro-batch",
        type=int,
        default=8,
        help="Upper bound for per-GPU micro batch binary search.",
    )
    parser.add_argument(
        "--grad-accum-candidates",
        type=str,
        default="4,8,12,16,24,32",
        help="Comma-separated gradient accumulation candidates.",
    )
    parser.add_argument(
        "--probe-real-steps",
        type=int,
        default=1,
        help="Real optimizer steps per probe run.",
    )
    parser.add_argument(
        "--main-process-port",
        type=int,
        default=29580,
        help="Start port for accelerate runs.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Timeout for each probe run.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return raw


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)


def _parse_int_list(value: str) -> list[int]:
    nums: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        nums.append(int(chunk))
    nums = sorted(set(nums))
    return [n for n in nums if n > 0]


def _prepare_probe_config(
    base_cfg: dict[str, Any],
    *,
    micro_batch: int,
    grad_accum: int,
    probe_real_steps: int,
) -> dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))

    model_cfg = cfg.setdefault("model", {})
    training_cfg = cfg.setdefault("training", {})
    logging_cfg = cfg.setdefault("logging", {})

    model_cfg["dtype"] = "bfloat16"
    training_cfg["mixed_precision"] = "bf16"
    training_cfg["batch_size"] = int(micro_batch)
    training_cfg["gradient_accumulation_steps"] = int(grad_accum)
    training_cfg["epochs"] = 1
    training_cfg["max_steps_per_epoch"] = int(probe_real_steps)
    training_cfg["checkpoint_freq_steps"] = 10**9
    training_cfg["checkpoint_freq_epochs"] = 10**9

    base_run = str(logging_cfg.get("run_name", "pi05"))
    logging_cfg["run_name"] = f"{base_run}_autotune_mb{micro_batch}_ga{grad_accum}"
    logging_cfg["use_tensorboard"] = False
    logging_cfg["log_freq"] = 1

    return cfg


def _build_probe_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = env.get("PYTHONNOUSERSITE", "1")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)

    py_paths = [
        str(PROJECT_ROOT / "common" / "src"),
        str(PROJECT_ROOT / "train" / "src"),
        str(PROJECT_ROOT / "deploy" / "src"),
    ]
    for candidate in (
        PROJECT_ROOT / "third_party" / "lerobot" / "src",
        PROJECT_ROOT.parent / "third_party" / "lerobot" / "src",
        WORKSPACE_ROOT / "third_party" / "lerobot" / "src",
    ):
        if candidate.exists():
            py_paths.append(str(candidate))
            break
    old_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(py_paths + ([old_path] if old_path else []))
    return env


def _run_probe(
    args: argparse.Namespace,
    *,
    config_path: Path,
    micro_batch: int,
    grad_accum: int,
    port: int,
    env: dict[str, str],
) -> TrialResult:
    AUTOTUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = AUTOTUNE_LOG_DIR / f"probe_mb{micro_batch}_ga{grad_accum}.log"
    cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(args.num_gpus),
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        "--main_process_port",
        str(port),
        "-m",
        "pi05.train.cli.train",
        "--config",
        str(config_path),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    merged = f"{proc.stdout}\n{proc.stderr}"
    log_path.write_text(merged, encoding="utf-8", errors="ignore")
    lowered = merged.lower()
    is_oom = any(re.search(pattern, lowered) for pattern in OOM_PATTERNS)
    return TrialResult(
        ok=(proc.returncode == 0),
        is_oom=is_oom,
        return_code=proc.returncode,
        log_path=log_path,
    )


def _probe_micro_batch_limit(args: argparse.Namespace, base_cfg: dict[str, Any], env: dict[str, str]) -> int:
    lo = 1
    hi = max(1, args.max_micro_batch)
    best = 0
    port = args.main_process_port

    while lo <= hi:
        mid = (lo + hi) // 2
        with tempfile.TemporaryDirectory(prefix="pi05-autotune-") as tmp_dir:
            probe_cfg_path = Path(tmp_dir) / "probe.yaml"
            probe_cfg = _prepare_probe_config(
                base_cfg,
                micro_batch=mid,
                grad_accum=1,
                probe_real_steps=args.probe_real_steps,
            )
            _write_yaml(probe_cfg_path, probe_cfg)
            result = _run_probe(
                args,
                config_path=probe_cfg_path,
                micro_batch=mid,
                grad_accum=1,
                port=port,
                env=env,
            )
        port += 1

        if result.ok:
            print(f"[autotune] micro_batch={mid} ok")
            best = mid
            lo = mid + 1
            continue
        if result.is_oom:
            print(f"[autotune] micro_batch={mid} OOM -> downshift")
            hi = mid - 1
            continue
        raise RuntimeError(
            f"Probe failed for micro_batch={mid}, return_code={result.return_code}. "
            f"See log: {result.log_path}"
        )

    if best <= 0:
        raise RuntimeError("Even micro_batch=1 failed; cannot find stable training setup.")
    return best


def _probe_grad_accum_limit(
    args: argparse.Namespace,
    base_cfg: dict[str, Any],
    env: dict[str, str],
    micro_batch: int,
    candidates: list[int],
) -> int:
    best = 1
    port = args.main_process_port + 100
    for grad_accum in candidates:
        with tempfile.TemporaryDirectory(prefix="pi05-autotune-") as tmp_dir:
            probe_cfg_path = Path(tmp_dir) / "probe.yaml"
            probe_cfg = _prepare_probe_config(
                base_cfg,
                micro_batch=micro_batch,
                grad_accum=grad_accum,
                probe_real_steps=args.probe_real_steps,
            )
            _write_yaml(probe_cfg_path, probe_cfg)
            result = _run_probe(
                args,
                config_path=probe_cfg_path,
                micro_batch=micro_batch,
                grad_accum=grad_accum,
                port=port,
                env=env,
            )
        port += 1

        if result.ok:
            best = grad_accum
            print(f"[autotune] grad_accum={grad_accum} ok")
            continue
        if result.is_oom:
            print(f"[autotune] grad_accum={grad_accum} OOM -> stop accumulation sweep")
            break
        raise RuntimeError(
            f"Probe failed for grad_accum={grad_accum}, return_code={result.return_code}. "
            f"See log: {result.log_path}"
        )
    return best


def _write_final_tuned_config(
    *,
    base_cfg: dict[str, Any],
    output_path: Path,
    micro_batch: int,
    grad_accum: int,
) -> None:
    tuned = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False))
    tuned.setdefault("model", {})["dtype"] = "bfloat16"
    train_cfg = tuned.setdefault("training", {})
    train_cfg["mixed_precision"] = "bf16"
    train_cfg["batch_size"] = int(micro_batch)
    train_cfg["gradient_accumulation_steps"] = int(grad_accum)
    train_cfg["checkpoint_freq_steps"] = int(train_cfg.get("checkpoint_freq_steps", 2000))
    _write_yaml(output_path, tuned)


def main() -> None:
    args = parse_args()
    if shutil.which("accelerate") is None:
        raise RuntimeError("accelerate not found in PATH. Please activate your conda env first.")

    base_config_path = args.base_config.expanduser().resolve()
    output_config_path = args.output_config.expanduser().resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    grad_accum_candidates = _parse_int_list(args.grad_accum_candidates)
    if not grad_accum_candidates:
        raise ValueError("No valid --grad-accum-candidates provided.")

    base_cfg = _load_yaml(base_config_path)
    env = _build_probe_env(args)

    print("[autotune] probing max micro batch with grad_accum=1 ...")
    best_micro_batch = _probe_micro_batch_limit(args, base_cfg, env)

    print("[autotune] probing gradient accumulation candidates ...")
    best_grad_accum = _probe_grad_accum_limit(
        args,
        base_cfg,
        env,
        micro_batch=best_micro_batch,
        candidates=grad_accum_candidates,
    )

    effective_batch = best_micro_batch * best_grad_accum * args.num_gpus
    _write_final_tuned_config(
        base_cfg=base_cfg,
        output_path=output_config_path,
        micro_batch=best_micro_batch,
        grad_accum=best_grad_accum,
    )

    print("")
    print("========== AUTOTUNE RESULT ==========")
    print(f"num_gpus:            {args.num_gpus}")
    print(f"best micro batch:    {best_micro_batch}")
    print(f"best grad accum:     {best_grad_accum}")
    print(f"effective batch:     {effective_batch}")
    print(f"tuned config saved:  {output_config_path}")
    print(f"probe logs dir:      {AUTOTUNE_LOG_DIR}")
    print("=====================================")


if __name__ == "__main__":
    main()
