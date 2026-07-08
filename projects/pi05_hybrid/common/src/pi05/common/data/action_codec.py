"""Action encoding helpers for Pi0.5 deployment outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pi05.common.robot.action_spec import ACTION_DIM, BimanualAction, split_bimanual_action


def ensure_action_vector(action: Iterable[float] | np.ndarray) -> np.ndarray:
    """Validate and return one flat 14-D action vector."""
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.size != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} action values, got {vector.size}")
    return vector


def ensure_action_chunk(chunk: Iterable[Iterable[float]] | np.ndarray, *, action_dim: int = ACTION_DIM) -> np.ndarray:
    """Validate and return a 2-D action chunk."""
    array = np.asarray(chunk, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Action chunk must be rank-2, got shape {array.shape}")
    if array.shape[1] != action_dim:
        raise ValueError(f"Expected action dim {action_dim}, got {array.shape[1]}")
    return array


def split_action(action: Iterable[float] | np.ndarray) -> BimanualAction:
    """Return a structured view of a single action vector."""
    return split_bimanual_action(ensure_action_vector(action))
