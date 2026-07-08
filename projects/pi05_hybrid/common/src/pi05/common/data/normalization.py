"""Normalization utilities for Pi0.5 state and action vectors.

This module provides:
- `ActionStateNormalizer`: strict min-max normalization to [-1, 1]
- metadata-aware factory helpers that can bootstrap statistics from a LeRobotDataset
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.io_utils import write_stats


DEFAULT_SAMPLE_COUNT = 2048
VECTOR_KEYS = ("observation.state", "action")


class ActionStateNormalizer:
    """Min-max normalizer for state or action vectors.

    The forward mapping is:
        y = 2 * (x - min) / (max - min) - 1

    For degenerate dimensions where `max == min`, the normalized value is forced
    to `0.0` to avoid division by zero.
    """

    def __init__(
        self,
        min_vals: torch.Tensor | Iterable[float],
        max_vals: torch.Tensor | Iterable[float],
        identity_indices: Sequence[int] | torch.Tensor | np.ndarray | None = None,
    ) -> None:
        self.min_vals = self._as_vector(min_vals, name="min_vals")
        self.vector_dim = int(self.min_vals.numel())
        self.max_vals = self._as_vector(max_vals, name="max_vals", expected_dim=self.vector_dim)
        self.range_vals = self.max_vals - self.min_vals
        self.non_zero_mask = self.range_vals != 0
        self.identity_mask = self._build_identity_mask(identity_indices)

    def normalize(self, data: torch.Tensor | np.ndarray | Iterable[float]) -> torch.Tensor:
        """Normalize data to the closed interval [-1, 1]."""
        tensor = self._as_tensor(data)
        min_vals = self._broadcast(self.min_vals, tensor)
        range_vals = self._broadcast(self.range_vals, tensor)
        non_zero_mask = self._broadcast(self.non_zero_mask, tensor)

        normalized = torch.zeros_like(tensor)
        normalized = torch.where(
            non_zero_mask,
            2.0 * (tensor - min_vals) / range_vals - 1.0,
            normalized,
        )
        identity_mask = self._broadcast(self.identity_mask, tensor)
        normalized = torch.where(identity_mask, tensor, normalized)
        return normalized

    def unnormalize(self, norm_data: torch.Tensor | np.ndarray | Iterable[float]) -> torch.Tensor:
        """Restore normalized data back to the original physical scale."""
        tensor = self._as_tensor(norm_data)
        min_vals = self._broadcast(self.min_vals, tensor)
        range_vals = self._broadcast(self.range_vals, tensor)
        non_zero_mask = self._broadcast(self.non_zero_mask, tensor)

        restored = torch.where(
            non_zero_mask,
            (tensor + 1.0) * 0.5 * range_vals + min_vals,
            min_vals,
        )
        identity_mask = self._broadcast(self.identity_mask, tensor)
        restored = torch.where(identity_mask, tensor, restored)
        return restored

    def __call__(self, data: torch.Tensor | np.ndarray | Iterable[float]) -> torch.Tensor:
        return self.normalize(data)

    @staticmethod
    def _as_vector(
        values: torch.Tensor | Iterable[float],
        name: str,
        expected_dim: int | None = None,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
        if expected_dim is not None and tensor.numel() != expected_dim:
            raise ValueError(f"{name} must contain {expected_dim} values, got {tensor.numel()}")
        return tensor

    def _as_tensor(self, data: torch.Tensor | np.ndarray | Iterable[float]) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            tensor = data.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
        if tensor.shape[-1] != self.vector_dim:
            raise ValueError(f"Expected last dimension to be {self.vector_dim}, got {tensor.shape}")
        return tensor

    def _broadcast(self, vector: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        shape = [1] * max(target.ndim - 1, 0) + [self.vector_dim]
        if vector.dtype == torch.bool:
            return vector.to(device=target.device).view(*shape)
        return vector.to(device=target.device, dtype=target.dtype).view(*shape)

    def _build_identity_mask(
        self,
        identity_indices: Sequence[int] | torch.Tensor | np.ndarray | None,
    ) -> torch.Tensor:
        mask = torch.zeros(self.vector_dim, dtype=torch.bool)
        if identity_indices is None:
            return mask
        indices = torch.as_tensor(identity_indices)
        if indices.dtype == torch.bool:
            indices = indices.flatten()
            if indices.numel() != self.vector_dim:
                raise ValueError(
                    f"identity mask must contain {self.vector_dim} values, got {indices.numel()}"
                )
            return indices.to(dtype=torch.bool)
        for idx in indices.flatten().tolist():
            idx = int(idx)
            if idx < 0 or idx >= self.vector_dim:
                raise ValueError(f"identity index {idx} is outside vector dim {self.vector_dim}")
            mask[idx] = True
        return mask


def build_normalizer_from_lerobot(
    dataset: LeRobotDataset,
    key: str,
    max_samples: int = DEFAULT_SAMPLE_COUNT,
    identity_indices: Sequence[int] | torch.Tensor | np.ndarray | None = None,
    refresh_invalid_stats: bool = True,
) -> ActionStateNormalizer:
    """Create a normalizer for one vector feature from LeRobot metadata or sampled data.

    Args:
        dataset: Loaded LeRobot dataset.
        key: Feature name, typically `observation.state` or `action`.
        max_samples: Maximum number of frames to scan when metadata stats are absent.
    """
    if refresh_invalid_stats:
        ensure_vector_stats(dataset, keys=(key,))

    stats_normalizer = _build_from_stats(dataset, key, identity_indices=identity_indices)
    if stats_normalizer is not None:
        return stats_normalizer

    if len(dataset) == 0:
        raise ValueError("Cannot build a normalizer from an empty dataset.")

    sample_count = min(len(dataset), max_samples)
    sample_indices = np.linspace(0, len(dataset) - 1, num=sample_count, dtype=int)

    min_vals = None
    max_vals = None
    for idx in sample_indices.tolist():
        value = torch.as_tensor(dataset.hf_dataset[idx][key], dtype=torch.float32).flatten()
        if min_vals is None:
            min_vals = value.clone()
            max_vals = value.clone()
            continue
        if value.numel() != min_vals.numel():
            raise ValueError(
                f"Feature '{key}' has inconsistent dimensions: {value.numel()} vs {min_vals.numel()}."
            )
        min_vals = torch.minimum(min_vals, value)
        max_vals = torch.maximum(max_vals, value)

    assert min_vals is not None
    assert max_vals is not None
    return ActionStateNormalizer(min_vals=min_vals, max_vals=max_vals, identity_indices=identity_indices)


def build_state_action_normalizers(
    dataset: LeRobotDataset,
    max_samples: int = DEFAULT_SAMPLE_COUNT,
    identity_indices: Mapping[str, Sequence[int] | torch.Tensor | np.ndarray] | None = None,
) -> tuple[ActionStateNormalizer, ActionStateNormalizer]:
    """Convenience helper returning both state and action normalizers."""
    ensure_vector_stats(dataset, keys=VECTOR_KEYS)
    identity_indices = identity_indices or {}
    state_normalizer = build_normalizer_from_lerobot(
        dataset=dataset,
        key="observation.state",
        max_samples=max_samples,
        identity_indices=identity_indices.get("observation.state"),
        refresh_invalid_stats=False,
    )
    action_normalizer = build_normalizer_from_lerobot(
        dataset=dataset,
        key="action",
        max_samples=max_samples,
        identity_indices=identity_indices.get("action"),
        refresh_invalid_stats=False,
    )
    return state_normalizer, action_normalizer


def ensure_vector_stats(dataset: LeRobotDataset, keys: Sequence[str] = VECTOR_KEYS) -> None:
    """Ensure stats.json has metadata-compatible vector stats for the requested keys."""
    invalid_keys = [key for key in keys if _stats_need_refresh(dataset, key)]
    if not invalid_keys:
        return

    fresh_stats = _scan_vector_stats(dataset, invalid_keys)
    merged_stats = dict(getattr(dataset.meta, "stats", None) or {})
    merged_stats.update(fresh_stats)
    write_stats(merged_stats, dataset.root)
    dataset.meta.stats = merged_stats


def _build_from_stats(
    dataset: LeRobotDataset,
    key: str,
    identity_indices: Sequence[int] | torch.Tensor | np.ndarray | None = None,
) -> ActionStateNormalizer | None:
    stats = getattr(dataset.meta, "stats", None)
    if not isinstance(stats, dict):
        return None

    feature_stats = stats.get(key)
    if not isinstance(feature_stats, dict):
        return None

    min_vals = feature_stats.get("min")
    max_vals = feature_stats.get("max")
    if min_vals is None or max_vals is None:
        return None

    return ActionStateNormalizer(min_vals=min_vals, max_vals=max_vals, identity_indices=identity_indices)


def _stats_need_refresh(dataset: LeRobotDataset, key: str) -> bool:
    expected_dim = _feature_dim(dataset, key)
    stats = getattr(dataset.meta, "stats", None)
    if not isinstance(stats, dict):
        return True
    feature_stats = stats.get(key)
    if not isinstance(feature_stats, dict):
        return True
    required_stats = ("min", "max", "mean", "std")
    for stat_name in required_stats:
        values = feature_stats.get(stat_name)
        if values is None:
            return True
        if torch.as_tensor(values, dtype=torch.float32).flatten().numel() != expected_dim:
            return True
    return False


def _scan_vector_stats(dataset: LeRobotDataset, keys: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    if len(dataset) == 0:
        raise ValueError("Cannot compute stats from an empty dataset.")

    accumulators = {key: _VectorStatsAccumulator(_feature_dim(dataset, key), key) for key in keys}
    for idx in range(len(dataset)):
        row = dataset.hf_dataset[idx]
        for key, accumulator in accumulators.items():
            accumulator.update(row[key])
    return {key: accumulator.finalize() for key, accumulator in accumulators.items()}


def _feature_dim(dataset: LeRobotDataset, key: str) -> int:
    features = dataset.features
    if key not in features:
        raise ValueError(f"Dataset is missing required feature '{key}'.")
    shape = tuple(features[key].get("shape", ()))
    if not shape:
        raise ValueError(f"Dataset feature '{key}' must have a non-empty shape.")
    return int(np.prod(shape))


class _VectorStatsAccumulator:
    def __init__(self, vector_dim: int, key: str) -> None:
        self.vector_dim = int(vector_dim)
        self.key = key
        self.count = 0
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None
        self.sum_vals = np.zeros(self.vector_dim, dtype=np.float64)
        self.sum_sq_vals = np.zeros(self.vector_dim, dtype=np.float64)

    def update(self, value: torch.Tensor | np.ndarray | Iterable[float]) -> None:
        array = np.asarray(value, dtype=np.float32).reshape(-1, self.vector_dim)
        if array.shape[-1] != self.vector_dim:
            raise ValueError(f"Feature '{self.key}' has shape {array.shape}, expected last dim {self.vector_dim}.")

        batch_min = array.min(axis=0)
        batch_max = array.max(axis=0)
        self.min_vals = batch_min if self.min_vals is None else np.minimum(self.min_vals, batch_min)
        self.max_vals = batch_max if self.max_vals is None else np.maximum(self.max_vals, batch_max)
        self.sum_vals += array.sum(axis=0, dtype=np.float64)
        self.sum_sq_vals += np.square(array, dtype=np.float64).sum(axis=0, dtype=np.float64)
        self.count += int(array.shape[0])

    def finalize(self) -> dict[str, np.ndarray]:
        if self.count == 0 or self.min_vals is None or self.max_vals is None:
            raise ValueError(f"No values observed while computing stats for '{self.key}'.")
        mean = self.sum_vals / self.count
        variance = np.maximum(self.sum_sq_vals / self.count - mean**2, 0.0)
        return {
            "min": self.min_vals.astype(np.float32),
            "max": self.max_vals.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(variance).astype(np.float32),
            "count": np.asarray([self.count], dtype=np.int64),
        }
