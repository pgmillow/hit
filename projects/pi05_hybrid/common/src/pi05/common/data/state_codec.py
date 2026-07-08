"""State encoding contract for Pi0.5 bimanual deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pi05.common.robot.action_spec import ARM_DOF, STATE_DIM


@dataclass(frozen=True)
class BimanualState:
    """Structured state used to build the 26-D policy observation vector."""

    left_arm_q: np.ndarray
    right_arm_q: np.ndarray
    left_hand_q: float
    right_hand_q: float
    left_ee_pos: np.ndarray
    left_ee_rpy: np.ndarray
    right_ee_pos: np.ndarray
    right_ee_rpy: np.ndarray


def encode_bimanual_state(state: BimanualState) -> np.ndarray:
    """Encode a structured bimanual state into the canonical 26-D vector."""
    vector = np.concatenate(
        [
            _vector(state.left_arm_q, ARM_DOF, "left_arm_q"),
            _vector(state.right_arm_q, ARM_DOF, "right_arm_q"),
            np.asarray([state.left_hand_q], dtype=np.float32),
            np.asarray([state.right_hand_q], dtype=np.float32),
            _vector(state.left_ee_pos, 3, "left_ee_pos"),
            _vector(state.left_ee_rpy, 3, "left_ee_rpy"),
            _vector(state.right_ee_pos, 3, "right_ee_pos"),
            _vector(state.right_ee_rpy, 3, "right_ee_rpy"),
        ]
    ).astype(np.float32, copy=False)
    if vector.size != STATE_DIM:
        raise ValueError(f"Expected encoded state dim {STATE_DIM}, got {vector.size}")
    return vector


def decode_picotele_proprioception(position: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Decode a legacy proprioception vector ordered as [right6, left6]."""
    values = np.asarray(list(position), dtype=np.float32).reshape(-1)
    if values.size < 2 * ARM_DOF:
        raise ValueError(f"Expected at least {2 * ARM_DOF} proprioception values, got {values.size}")
    right = values[:ARM_DOF].copy()
    left = values[ARM_DOF : 2 * ARM_DOF].copy()
    return left, right


def _vector(value: Iterable[float] | np.ndarray, dim: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != dim:
        raise ValueError(f"{name} must contain {dim} values, got {vector.size}")
    return vector
