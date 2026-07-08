from __future__ import annotations

from pathlib import Path

import torch

from pi05.train.engine.checkpoints import save_step_adapter_checkpoints
from safetensors.torch import load_file as load_safetensors


class _DummyAccelerator:
    is_main_process = True

    @staticmethod
    def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model

    @staticmethod
    def wait_for_everyone() -> None:
        return None

    @staticmethod
    def print(*args, **kwargs) -> None:
        return None


class _DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.tactile_encoder = torch.nn.Linear(3, 4)

    def save_pretrained(self, output_dir: Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "adapter.bin")


def test_save_step_adapter_checkpoints_keeps_only_two_dirs(tmp_path: Path) -> None:
    accelerator = _DummyAccelerator()
    model = _DummyModel()
    run_output_dir = tmp_path / "run"

    # First save creates rolling/latest directories.
    save_step_adapter_checkpoints(
        accelerator=accelerator,
        model=model,
        run_output_dir=run_output_dir,
        real_step=2000,
    )

    rolling_dir = run_output_dir / "checkpoint_step_rolling"
    latest_dir = run_output_dir / "checkpoint_step_latest"
    assert (rolling_dir / "adapter" / "adapter.bin").exists()
    assert (latest_dir / "adapter" / "adapter.bin").exists()
    tactile_state = load_safetensors(rolling_dir / "adapter" / "tactile_encoder.safetensors")
    assert tactile_state["weight"].shape == (4, 3)
    assert tactile_state["bias"].shape == (4,)
    assert (rolling_dir / "real_step.txt").read_text(encoding="utf-8").strip() == "2000"
    assert (latest_dir / "real_step.txt").read_text(encoding="utf-8").strip() == "2000"

    # Add stale file, then save again to verify overwrite behavior.
    stale_file = rolling_dir / "adapter" / "stale.tmp"
    stale_file.write_text("old", encoding="utf-8")
    assert stale_file.exists()

    save_step_adapter_checkpoints(
        accelerator=accelerator,
        model=model,
        run_output_dir=run_output_dir,
        real_step=4000,
    )

    assert not stale_file.exists()
    assert (rolling_dir / "real_step.txt").read_text(encoding="utf-8").strip() == "4000"
    assert (latest_dir / "real_step.txt").read_text(encoding="utf-8").strip() == "4000"
    assert (rolling_dir / "adapter" / "adapter.bin").exists()
    assert (latest_dir / "adapter" / "adapter.bin").exists()
