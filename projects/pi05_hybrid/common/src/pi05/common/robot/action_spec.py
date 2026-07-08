"""Robot action layout shared by training and deployment.

The Pi0.5 policy emits a flat action vector. This module documents the vector
contract and provides small helpers for splitting or assembling that vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


ARM_DOF = 6
HAND_DOF = 1
ACTION_DIM = 14
STATE_DIM = 26
ARM_JOINT_NAMES = tuple(f"joint{i + 1}" for i in range(ARM_DOF))


@dataclass(frozen=True)
class BimanualAction:
    """Structured view of one policy action step."""

    left_arm: np.ndarray
    right_arm: np.ndarray
    left_hand: float
    right_hand: float

    def as_vector(self) -> np.ndarray:
        """Return the canonical 14-D action vector."""
        return np.concatenate(
            [
                np.asarray(self.left_arm, dtype=np.float32).reshape(ARM_DOF),
                np.asarray(self.right_arm, dtype=np.float32).reshape(ARM_DOF),
                np.asarray([self.left_hand, self.right_hand], dtype=np.float32),
            ]
        )


def split_bimanual_action(action: Iterable[float] | np.ndarray) -> BimanualAction:
    """Split a flat 14-D policy action into arm and hand commands."""
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.size != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} action values, got {vector.size}")
    return BimanualAction(
        left_arm=vector[:ARM_DOF].copy(),
        right_arm=vector[ARM_DOF : 2 * ARM_DOF].copy(),
        left_hand=float(vector[2 * ARM_DOF]),
        right_hand=float(vector[2 * ARM_DOF + 1]),
    )


def hand_command_to_trigger(command: float, *, open_value: float = 1000.0, closed_value: float = 300.0) -> float:
    """Convert the dataset hand-command scale to a normalized trigger value."""
    span = max(1e-6, float(open_value) - float(closed_value))
    value = (float(open_value) - float(command)) / span
    return float(np.clip(value, 0.0, 1.0))
