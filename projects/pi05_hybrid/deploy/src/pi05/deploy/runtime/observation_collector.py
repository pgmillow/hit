"""Observation assembly for Pi0.5 deployment.

ROS callbacks update individual fields. The collector emits a complete
snapshot only when all policy-required fields are available.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import torch

from pi05.common.data.state_codec import BimanualState, decode_picotele_proprioception, encode_bimanual_state
from pi05.deploy.runtime.shared_buffer import ObservationSnapshot


class ObservationCollector:
    """Collect latest images and robot state into policy observations."""

    REQUIRED_IMAGE_KEYS = ("top", "left_wrist", "right_wrist")

    def __init__(self, *, proprioception_order: str = "right_left") -> None:
        self._lock = threading.Lock()
        self._images: dict[str, torch.Tensor] = {}
        self._values: dict[str, Any] = {}
        self._stamps: dict[str, float] = {}
        self._proprioception_order = proprioception_order

    def update_image(self, name: str, image: torch.Tensor) -> None:
        with self._lock:
            self._images[name] = image.detach().clone()
            self._stamps[f"image_{name}"] = time.monotonic()

    def update_proprioception(self, positions: list[float] | tuple[float, ...]) -> None:
        values = np.asarray(positions, dtype=np.float32)
        if self._proprioception_order == "right_left":
            left, right = decode_picotele_proprioception(values)
        elif self._proprioception_order == "left_right":
            if values.size < 12:
                raise ValueError(f"Expected at least 12 proprioception values, got {values.size}")
            left = values[:6].copy()
            right = values[6:12].copy()
        else:
            raise ValueError(f"Unsupported proprioception_order: {self._proprioception_order}")
        with self._lock:
            self._values["left_arm_q"] = left
            self._values["right_arm_q"] = right
            self._stamps["proprioception"] = time.monotonic()

    def update_hand(self, side: str, value: float) -> None:
        with self._lock:
            self._values[f"{side}_hand_q"] = float(value)
            self._stamps[f"{side}_hand"] = time.monotonic()

    def update_vector(self, key: str, values: list[float] | tuple[float, ...] | np.ndarray) -> None:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        with self._lock:
            self._values[key] = vector
            self._stamps[key] = time.monotonic()

    def snapshot(self, *, max_age_s: float | None = None) -> ObservationSnapshot | None:
        now = time.monotonic()
        with self._lock:
            if not self._has_required_locked():
                return None
            if max_age_s is not None and self._has_stale_field_locked(now, max_age_s):
                return None
            images = {key: value.detach().clone() for key, value in self._images.items()}
            values = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in self._values.items()
            }

        state = BimanualState(
            left_arm_q=values["left_arm_q"],
            right_arm_q=values["right_arm_q"],
            left_hand_q=float(values["left_hand_q"]),
            right_hand_q=float(values["right_hand_q"]),
            left_ee_pos=values["left_ee_pos"],
            left_ee_rpy=values["left_ee_rpy"],
            right_ee_pos=values["right_ee_pos"],
            right_ee_rpy=values["right_ee_rpy"],
        )
        return ObservationSnapshot(
            images=images,
            state=state,
            encoded_state=encode_bimanual_state(state),
            captured_at_s=now,
        )

    def missing_fields(self) -> list[str]:
        with self._lock:
            missing = [f"image_{key}" for key in self.REQUIRED_IMAGE_KEYS if key not in self._images]
            for key in self._required_value_keys():
                if key not in self._values:
                    missing.append(key)
            return missing

    def _has_required_locked(self) -> bool:
        return all(key in self._images for key in self.REQUIRED_IMAGE_KEYS) and all(
            key in self._values for key in self._required_value_keys()
        )

    def _has_stale_field_locked(self, now: float, max_age_s: float) -> bool:
        required_stamp_keys = [f"image_{key}" for key in self.REQUIRED_IMAGE_KEYS]
        required_stamp_keys.extend(
            [
                "proprioception",
                "left_hand",
                "right_hand",
                "left_ee_pos",
                "left_ee_rpy",
                "right_ee_pos",
                "right_ee_rpy",
            ]
        )
        return any(now - float(self._stamps.get(key, 0.0)) > max_age_s for key in required_stamp_keys)

    @staticmethod
    def _required_value_keys() -> tuple[str, ...]:
        return (
            "left_arm_q",
            "right_arm_q",
            "left_hand_q",
            "right_hand_q",
            "left_ee_pos",
            "left_ee_rpy",
            "right_ee_pos",
            "right_ee_rpy",
        )
