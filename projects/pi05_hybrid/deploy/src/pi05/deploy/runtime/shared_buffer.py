"""Thread-safe buffers shared by deployment producer and consumer loops."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
from typing import Generic, Mapping, TypeVar, TYPE_CHECKING

import numpy as np

from pi05.common.data.state_codec import BimanualState

if TYPE_CHECKING:
    import torch


T = TypeVar("T")


@dataclass(frozen=True)
class ObservationSnapshot:
    """Complete policy observation captured at one monotonic timestamp."""

    images: Mapping[str, "torch.Tensor"]
    state: BimanualState
    encoded_state: np.ndarray
    captured_at_s: float


@dataclass
class ActionChunk:
    """One action chunk produced by asynchronous policy inference."""

    actions: np.ndarray
    obs_time: float
    infer_start_time: float
    ready_time: float
    action_dt: float
    request_id: int
    cursor: int = 0

    def __post_init__(self) -> None:
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.actions.ndim != 2:
            raise ValueError(f"ActionChunk.actions must be rank-2, got shape {self.actions.shape}")
        if self.action_dt <= 0.0:
            raise ValueError("ActionChunk.action_dt must be positive")

    @property
    def chunk_size(self) -> int:
        return int(self.actions.shape[0])

    def aligned_index(self, now: float) -> int:
        """Return the time-aligned action index for ``now`` clamped to the chunk."""
        raw_idx = int((float(now) - float(self.obs_time)) / float(self.action_dt))
        return int(np.clip(raw_idx, 0, max(0, self.chunk_size - 1)))


@dataclass(frozen=True)
class InferenceRequest:
    """Latest observation submitted by the control loop for background inference."""

    observation: ObservationSnapshot
    obs_time: float
    request_id: int
    trigger_step: int


class LatestQueue(Generic[T]):
    """Small bounded queue that keeps only the newest item."""

    def __init__(self, maxsize: int = 1) -> None:
        if int(maxsize) < 1:
            raise ValueError("LatestQueue.maxsize must be >= 1")
        self._items: deque[T] = deque(maxlen=int(maxsize))
        self._lock = threading.Lock()

    def put_latest(self, item: T) -> None:
        """Insert an item, dropping older items when the queue is full."""
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self._items.popleft()
            self._items.append(item)

    def get_latest_or_none(self) -> T | None:
        """Return the newest item and clear all older queued items."""
        with self._lock:
            if not self._items:
                return None
            latest = self._items.pop()
            self._items.clear()
            return latest

    def empty(self) -> bool:
        with self._lock:
            return not self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass
class RuntimeMetrics:
    """Small set of live counters useful for deployment monitoring."""

    inference_count: int = 0
    inference_error_count: int = 0
    inference_request_count: int = 0
    chunk_result_count: int = 0
    discarded_chunk_count: int = 0
    chunk_switch_count: int = 0
    fallback_count: int = 0
    dropped_observation_count: int = 0
    published_action_count: int = 0
    held_action_count: int = 0
    rejected_action_count: int = 0
    last_inference_latency_s: float = 0.0
    ema_inference_latency_s: float = 0.0
    last_action_age_s: float = 0.0
    last_error: str | None = None
    updated_at_s: float = field(default_factory=time.monotonic)

    def record_latency(self, latency_s: float) -> None:
        self.inference_count += 1
        self.last_inference_latency_s = max(0.0, float(latency_s))
        if self.ema_inference_latency_s <= 0.0:
            self.ema_inference_latency_s = self.last_inference_latency_s
        else:
            self.ema_inference_latency_s = 0.8 * self.ema_inference_latency_s + 0.2 * self.last_inference_latency_s
        self.updated_at_s = time.monotonic()

    def as_dict(self) -> dict[str, float | int | str | None]:
        return {
            "inference_count": self.inference_count,
            "inference_error_count": self.inference_error_count,
            "inference_request_count": self.inference_request_count,
            "chunk_result_count": self.chunk_result_count,
            "discarded_chunk_count": self.discarded_chunk_count,
            "chunk_switch_count": self.chunk_switch_count,
            "fallback_count": self.fallback_count,
            "dropped_observation_count": self.dropped_observation_count,
            "published_action_count": self.published_action_count,
            "held_action_count": self.held_action_count,
            "rejected_action_count": self.rejected_action_count,
            "last_inference_latency_s": self.last_inference_latency_s,
            "ema_inference_latency_s": self.ema_inference_latency_s,
            "last_action_age_s": self.last_action_age_s,
            "last_error": self.last_error,
            "updated_at_s": self.updated_at_s,
        }


class SharedBuffer:
    """Shared observation, inference request, result, and metrics state."""

    def __init__(
        self,
        *,
        max_inference_requests: int = 1,
        max_pending_chunks: int = 1,
    ) -> None:
        self._lock = threading.Lock()
        self._latest_observation: ObservationSnapshot | None = None
        self.inference_request_queue: LatestQueue[InferenceRequest] = LatestQueue(max_inference_requests)
        self.chunk_result_queue: LatestQueue[ActionChunk] = LatestQueue(max_pending_chunks)
        self.metrics = RuntimeMetrics()

    def set_observation(self, observation: ObservationSnapshot) -> None:
        with self._lock:
            if self._latest_observation is not None:
                self.metrics.dropped_observation_count += 1
            self._latest_observation = observation

    def latest_observation(self, *, max_age_s: float | None = None) -> ObservationSnapshot | None:
        now = time.monotonic()
        with self._lock:
            observation = self._latest_observation
        if observation is None:
            return None
        if max_age_s is not None and now - observation.captured_at_s > max_age_s:
            return None
        return observation

    def record_inference_latency(self, latency_s: float) -> None:
        with self._lock:
            self.metrics.record_latency(latency_s)

    def record_inference_request(self) -> None:
        with self._lock:
            self.metrics.inference_request_count += 1
            self.metrics.updated_at_s = time.monotonic()

    def record_chunk_result(self) -> None:
        with self._lock:
            self.metrics.chunk_result_count += 1
            self.metrics.updated_at_s = time.monotonic()

    def record_discarded_chunk(self, reason: str) -> None:
        with self._lock:
            self.metrics.discarded_chunk_count += 1
            self.metrics.last_error = reason
            self.metrics.updated_at_s = time.monotonic()

    def record_chunk_switch(self) -> None:
        with self._lock:
            self.metrics.chunk_switch_count += 1
            self.metrics.updated_at_s = time.monotonic()

    def record_fallback(self, reason: str) -> None:
        with self._lock:
            self.metrics.fallback_count += 1
            self.metrics.last_error = reason
            self.metrics.updated_at_s = time.monotonic()

    def record_inference_error(self, message: str) -> None:
        with self._lock:
            self.metrics.inference_error_count += 1
            self.metrics.last_error = message
            self.metrics.updated_at_s = time.monotonic()

    def record_published_action(self) -> None:
        with self._lock:
            self.metrics.published_action_count += 1
            self.metrics.updated_at_s = time.monotonic()

    def record_held_action(self) -> None:
        with self._lock:
            self.metrics.held_action_count += 1
            self.metrics.updated_at_s = time.monotonic()

    def record_rejected_action(self, reason: str) -> None:
        with self._lock:
            self.metrics.rejected_action_count += 1
            self.metrics.last_error = reason
            self.metrics.updated_at_s = time.monotonic()

    def metrics_snapshot(self) -> dict[str, float | int | str | None]:
        with self._lock:
            return self.metrics.as_dict()
