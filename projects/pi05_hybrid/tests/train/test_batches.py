from __future__ import annotations

import torch

from pi05.train.engine.batches import to_lerobot_pi05_batch


def test_batch_adapter_forwards_tactile_window_and_mask() -> None:
    batch = {
        "state": torch.rand(2, 26),
        "action_chunk": torch.rand(2, 30, 14),
        "task": ["a", "b"],
        "image_top": torch.rand(2, 3, 224, 224),
        "tactile": torch.rand(2, 8, 22, 1, 14, 14),
        "tactile_mask": torch.ones(2, 8, dtype=torch.bool),
    }

    converted = to_lerobot_pi05_batch(batch)

    assert converted["observation.tactile"].shape == (2, 8, 22, 1, 14, 14)
    assert converted["observation.tactile_mask"].shape == (2, 8)
    assert "observation.images.tactile" not in converted
