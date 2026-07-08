"""Safety checks for Pi0.5 deployment actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pi05.common.data.action_codec import ensure_action_vector, split_action
from pi05.common.robot.action_spec import BimanualAction
from pi05.common.robot.joint_limits import JointLimitSpec
from pi05.deploy.config.schema import SafetyConfig

if TYPE_CHECKING:
    from pi05.deploy.runtime.shared_buffer import ObservationSnapshot


@dataclass(frozen=True)
class SafetyResult:
    """Result returned by the safety guard."""

    action: BimanualAction | None
    accepted: bool
    reason: str | None = None


class SafetyGuard:
    """Validate, clamp, and rate-limit policy actions before publication."""

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._left_limits, self._right_limits = self._build_joint_limits(config)

    def filter_action(
        self,
        action: np.ndarray,
        *,
        observation: ObservationSnapshot | None,
        previous_action: BimanualAction | None,
    ) -> SafetyResult:
        try:
            vector = ensure_action_vector(action)
        except ValueError as exc:
            return SafetyResult(action=None, accepted=False, reason=str(exc))
        if not np.all(np.isfinite(vector)):
            return SafetyResult(action=None, accepted=False, reason="action contains NaN or Inf")

        structured = split_action(vector)
        left = structured.left_arm
        right = structured.right_arm

        if self._left_limits is not None:
            left = self._left_limits.clamp(left)
        if self._right_limits is not None:
            right = self._right_limits.clamp(right)

        if self.config.max_joint_delta_rad > 0.0:
            left_anchor, right_anchor = self._delta_anchor(observation, previous_action)
            if left_anchor is not None:
                left = self._clamp_delta(left, left_anchor)
            if right_anchor is not None:
                right = self._clamp_delta(right, right_anchor)

        return SafetyResult(
            action=BimanualAction(
                left_arm=left.astype(np.float32, copy=False),
                right_arm=right.astype(np.float32, copy=False),
                left_hand=float(np.clip(structured.left_hand, self.config.hand_min, self.config.hand_max)),
                right_hand=float(np.clip(structured.right_hand, self.config.hand_min, self.config.hand_max)),
            ),
            accepted=True,
        )

    def _clamp_delta(self, target: np.ndarray, anchor: np.ndarray) -> np.ndarray:
        limit = float(self.config.max_joint_delta_rad)
        return anchor + np.clip(target - anchor, -limit, limit)

    @staticmethod
    def _delta_anchor(
        observation: ObservationSnapshot | None,
        previous_action: BimanualAction | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if previous_action is not None:
            return previous_action.left_arm, previous_action.right_arm
        if observation is not None:
            return observation.state.left_arm_q, observation.state.right_arm_q
        return None, None

    @staticmethod
    def _build_joint_limits(config: SafetyConfig) -> tuple[JointLimitSpec | None, JointLimitSpec | None]:
        limits = config.joint_limits
        if not limits.enabled:
            return None, None
        return (
            JointLimitSpec.from_values(limits.left_min_rad, limits.left_max_rad),
            JointLimitSpec.from_values(limits.right_min_rad, limits.right_max_rad),
        )
