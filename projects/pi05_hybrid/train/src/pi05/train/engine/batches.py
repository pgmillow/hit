"""Batch adapters between the local dataset and LeRobot PI0.5 preprocessor."""

from __future__ import annotations

from typing import Any


def to_lerobot_pi05_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Map local dataset keys to the official PI0.5 processor input schema."""
    model_batch = {
        "observation.state": batch["state"],
        "action": batch["action_chunk"],
        "task": list(batch["task"]),
    }
    for key, value in batch.items():
        if key.startswith("image_"):
            camera = key.removeprefix("image_")
            model_batch[f"observation.images.{camera}"] = value
    if "tactile" in batch:
        model_batch["observation.tactile"] = batch["tactile"]
        model_batch["observation.tactile_mask"] = batch["tactile_mask"]
    return model_batch
