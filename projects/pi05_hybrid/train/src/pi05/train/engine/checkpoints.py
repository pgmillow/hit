"""Checkpoint and final adapter export helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import save_file as save_safetensors


def maybe_resume(accelerator: Accelerator, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    accelerator.load_state(str(checkpoint_path))
    accelerator.print(f"Resumed training state from: {checkpoint_path}")


def save_epoch_adapter_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    run_output_dir: Path,
    epoch: int,
) -> Path:
    """Save an epoch-level LoRA adapter without optimizer/scheduler state."""
    adapter_dir = run_output_dir / f"checkpoint_epoch_{epoch}" / "adapter"
    if accelerator.is_main_process:
        _save_adapter(accelerator, model, adapter_dir)
        accelerator.print(f"Saved epoch LoRA adapter to: {adapter_dir}")
    accelerator.wait_for_everyone()
    return adapter_dir


def save_step_adapter_checkpoints(
    accelerator: Accelerator,
    model: torch.nn.Module,
    run_output_dir: Path,
    *,
    real_step: int,
) -> tuple[Path, Path]:
    """Save rolling and latest step checkpoints, overwriting previous ones.

    Keeps only two step-level checkpoint directories at any time:
    - checkpoint_step_rolling/adapter
    - checkpoint_step_latest/adapter
    """
    rolling_adapter_dir = run_output_dir / "checkpoint_step_rolling" / "adapter"
    latest_adapter_dir = run_output_dir / "checkpoint_step_latest" / "adapter"
    if accelerator.is_main_process:
        _save_adapter(accelerator, model, rolling_adapter_dir, overwrite=True)
        _write_step_marker(rolling_adapter_dir.parent, real_step)
        _save_adapter(accelerator, model, latest_adapter_dir, overwrite=True)
        _write_step_marker(latest_adapter_dir.parent, real_step)
        accelerator.print(
            f"Saved step checkpoints at real_step={real_step}: "
            f"rolling={rolling_adapter_dir.parent}, latest={latest_adapter_dir.parent}"
        )
    accelerator.wait_for_everyone()
    return rolling_adapter_dir, latest_adapter_dir


def export_final_adapter(accelerator: Accelerator, model: torch.nn.Module, run_output_dir: Path) -> Path:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter_dir = run_output_dir / "final_adapter"
    if accelerator.is_main_process:
        _save_adapter(accelerator, model, final_adapter_dir)
        accelerator.print(f"Saved LoRA adapter to: {final_adapter_dir}")
    accelerator.wait_for_everyone()
    return final_adapter_dir


def _save_adapter(
    accelerator: Accelerator,
    model: torch.nn.Module,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir)
    tactile_encoder = _find_tactile_encoder(unwrapped_model)
    if tactile_encoder is not None:
        tactile_state = {key: value.detach().cpu() for key, value in tactile_encoder.state_dict().items()}
        save_safetensors(tactile_state, str(output_dir / "tactile_encoder.safetensors"))


def _find_tactile_encoder(model: torch.nn.Module) -> torch.nn.Module | None:
    for module in model.modules():
        tactile_encoder = getattr(module, "tactile_encoder", None)
        if isinstance(tactile_encoder, torch.nn.Module):
            return tactile_encoder
    return None


def _write_step_marker(checkpoint_dir: Path, real_step: int) -> None:
    marker_path = checkpoint_dir / "real_step.txt"
    marker_path.write_text(f"{real_step}\n", encoding="utf-8")
