#!/usr/bin/env python3
"""Smoke test for the Pi0.5 LoRA training stack.

This script validates that the current codebase can:
1. Build a PI0.5 model with LoRA adapters.
2. Create a perfect mock LeRobot-style batch.
3. Run one forward pass and produce a finite loss.
4. Run one backward pass and verify LoRA gradients exist.

It prefers the official LeRobot PI0.5 preprocessor for task tokenization. If the
tokenizer is unavailable or cannot be downloaded, it automatically falls back to
synthetic language tokens so the computation graph can still be validated.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except ModuleNotFoundError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parents[1]


def _bootstrap_import_paths() -> None:
    candidate_paths = [
        WORKSPACE_ROOT / "third_party" / "lerobot" / "src",
        PROJECT_ROOT / "common" / "src",
        PROJECT_ROOT / "train" / "src",
    ]
    for path in candidate_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_bootstrap_import_paths()


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "train" / "config" / "lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forward/backward smoke test for PI0.5 LoRA.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="LoRA YAML config.")
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="",
        help="Optional pretrained model repo or local path. Leave empty to use random init.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size.")
    parser.add_argument("--chunk-size", type=int, default=30, help="Synthetic action chunk size.")
    parser.add_argument("--state-dim", type=int, default=26, help="Synthetic observation.state dimension.")
    parser.add_argument("--action-dim", type=int, default=14, help="Synthetic action dimension.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Target device.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        choices=["auto", "official", "synthetic"],
        default="auto",
        help="How to create language inputs.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping, got {type(config)}")
    return config


def make_mock_batch(
    batch_size: int,
    chunk_size: int,
    state_dim: int,
    action_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    """Create a clean LeRobot-style batch with synthetic tensors."""
    return {
        "observation.images.top": torch.rand(batch_size, 3, 224, 224, device=device, dtype=torch.float32),
        "observation.images.left_wrist": torch.rand(
            batch_size, 3, 224, 224, device=device, dtype=torch.float32
        ),
        "observation.images.right_wrist": torch.rand(
            batch_size, 3, 224, 224, device=device, dtype=torch.float32
        ),
        "observation.state": torch.rand(batch_size, state_dim, device=device, dtype=torch.float32) * 2.0 - 1.0,
        "action": torch.rand(batch_size, chunk_size, action_dim, device=device, dtype=torch.float32) * 2.0 - 1.0,
        "task": ["Pour water into the cup"] * batch_size,
    }


def prepare_model_batch(
    model: torch.nn.Module,
    raw_batch: dict[str, Any],
    tokenizer_mode: str,
) -> tuple[dict[str, Any], str]:
    """Convert the mock batch into the exact PI0.5 training batch expected by the policy."""
    if tokenizer_mode in ("auto", "official"):
        try:
            from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

            preprocessor, _ = make_pi05_pre_post_processors(_get_policy_config(model), dataset_stats=None)
            processed_batch = preprocessor(copy.deepcopy(raw_batch))
            return processed_batch, "official_preprocessor"
        except Exception as exc:
            if tokenizer_mode == "official":
                raise RuntimeError(f"Official tokenizer path failed: {exc}") from exc
            print(f"[smoke] Official tokenizer path unavailable, falling back to synthetic tokens: {exc}")

    fallback_batch = copy.deepcopy(raw_batch)
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    device = fallback_batch["observation.images.top"].device
    batch_size = fallback_batch["observation.images.top"].shape[0]
    seq_len = int(getattr(_get_policy_config(model), "tokenizer_max_length", 200))
    vocab_size = 257152
    fallback_batch[OBS_LANGUAGE_TOKENS] = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        device=device,
        dtype=torch.long,
    )
    fallback_batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
        batch_size,
        seq_len,
        device=device,
        dtype=torch.bool,
    )
    return fallback_batch, "synthetic_tokens"


def _get_policy_config(model: torch.nn.Module) -> Any:
    if hasattr(model, "pi05_policy_config"):
        return model.pi05_policy_config
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "config"):
            return candidate.config
    raise AttributeError("Could not locate PI0.5 policy config on model.")


def find_lora_gradient(model: torch.nn.Module) -> tuple[str, torch.Tensor]:
    """Find one LoRA parameter that received a valid gradient."""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" not in name:
            continue
        if param.grad is not None:
            return name, param.grad
    raise RuntimeError("No LoRA parameter received a gradient.")


def main() -> None:
    args = parse_args()
    _require_torch()
    device = torch.device(args.device)
    config_dict = load_yaml(args.config)

    model_cfg = config_dict.setdefault("model", {})
    model_cfg.setdefault("device", str(device))
    model_cfg.setdefault("dtype", "bfloat16" if device.type == "cuda" else "float32")
    model_cfg.setdefault("gradient_checkpointing", True)
    model_cfg.setdefault("allow_random_init_peft", True)
    model_cfg.setdefault("chunk_size", args.chunk_size)
    model_cfg.setdefault("n_action_steps", args.chunk_size)
    model_cfg.setdefault("state_dim", args.state_dim)
    model_cfg.setdefault("action_dim", args.action_dim)
    model_cfg.setdefault("max_action_dim", args.action_dim)
    if not args.pretrained_path:
        model_cfg.setdefault("paligemma_variant", "gemma_2b")
        model_cfg.setdefault("action_expert_variant", "gemma_300m")

    pretrained_path = args.pretrained_path or None
    from pi05.common.model.builder import build_pi05_with_lora

    model = build_pi05_with_lora(config_dict=config_dict, pretrained_path=pretrained_path)
    model.to(device)
    model.train()

    raw_batch = make_mock_batch(args.batch_size, args.chunk_size, args.state_dim, args.action_dim, device)
    model_batch, batch_mode = prepare_model_batch(model, raw_batch, args.tokenizer_mode)

    print(f"[smoke] batch preparation mode: {batch_mode}")
    print(f"[smoke] top image shape: {tuple(model_batch['observation.images.top'].shape)}")
    print(f"[smoke] left wrist image shape: {tuple(model_batch['observation.images.left_wrist'].shape)}")
    print(f"[smoke] right wrist image shape: {tuple(model_batch['observation.images.right_wrist'].shape)}")
    print(f"[smoke] state shape: {tuple(model_batch['observation.state'].shape)}")
    print(f"[smoke] action shape: {tuple(model_batch['action'].shape)}")

    model.zero_grad(set_to_none=True)
    loss, loss_dict = model(model_batch)
    print(f"[smoke] forward loss: {loss.item():.6f}")
    print(f"[smoke] loss dict: {loss_dict}")

    if not torch.isfinite(loss):
        raise RuntimeError(f"Loss is not finite: {loss.item()}")

    loss.backward()
    grad_name, grad_tensor = find_lora_gradient(model)
    grad_norm = grad_tensor.norm().item()
    print(f"[smoke] gradient found on: {grad_name}")
    print(f"[smoke] gradient norm: {grad_norm:.6f}")

    if not torch.isfinite(torch.tensor(grad_norm)):
        raise RuntimeError(f"Gradient norm is not finite: {grad_norm}")

    print("[smoke] forward/backward smoke test passed.")


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required to run the Pi0.5 smoke test.")


if __name__ == "__main__":
    main()
