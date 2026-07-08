"""Reproducibility helpers."""

from __future__ import annotations

from accelerate.utils import set_seed as accelerate_set_seed


def set_training_seed(seed: int | None) -> None:
    if seed is None:
        return
    accelerate_set_seed(seed)
