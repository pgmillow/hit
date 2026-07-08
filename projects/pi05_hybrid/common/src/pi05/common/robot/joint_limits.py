"""Joint limit primitives used by deployment safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pi05.common.robot.action_spec import ARM_DOF


@dataclass(frozen=True)
class JointLimitSpec:
    """Per-arm joint bounds in radians."""

    min_rad: np.ndarray
    max_rad: np.ndarray

    @classmethod
    def from_values(cls, min_rad: Iterable[float], max_rad: Iterable[float]) -> "JointLimitSpec":
        min_vec = np.asarray(list(min_rad), dtype=np.float32).reshape(-1)
        max_vec = np.asarray(list(max_rad), dtype=np.float32).reshape(-1)
        if min_vec.size != ARM_DOF or max_vec.size != ARM_DOF:
            raise ValueError(f"Joint limits must each contain {ARM_DOF} values.")
        if np.any(min_vec > max_vec):
            raise ValueError("Joint limit minimum must be <= maximum for every joint.")
        return cls(min_rad=min_vec, max_rad=max_vec)

    def clamp(self, joints_rad: Iterable[float] | np.ndarray) -> np.ndarray:
        """Clamp a 6-D joint vector into the configured bounds."""
        joints = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if joints.size != ARM_DOF:
            raise ValueError(f"Expected {ARM_DOF} joint values, got {joints.size}")
        return np.clip(joints, self.min_rad, self.max_rad).astype(np.float32, copy=False)

    def contains(self, joints_rad: Iterable[float] | np.ndarray) -> bool:
        """Return whether a 6-D joint vector is inside the configured bounds."""
        joints = np.asarray(joints_rad, dtype=np.float32).reshape(-1)
        if joints.size != ARM_DOF:
            return False
        return bool(np.all(joints >= self.min_rad) and np.all(joints <= self.max_rad))


def broad_joint_limits(limit_rad: float = 6.283185307179586) -> JointLimitSpec:
    """Return permissive limits used when hardware-specific bounds are absent."""
    value = abs(float(limit_rad))
    return JointLimitSpec.from_values([-value] * ARM_DOF, [value] * ARM_DOF)
