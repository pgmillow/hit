"""High-rate rolling action consumer for Pi0.5 deployment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from pi05.common.robot.action_spec import BimanualAction
from pi05.deploy.runtime.safety_guard import SafetyGuard
from pi05.deploy.runtime.shared_buffer import ActionChunk, InferenceRequest, LatestQueue, ObservationSnapshot, SharedBuffer


LogFn = Callable[[str], None]
ObservationProvider = Callable[[], ObservationSnapshot | None]


@dataclass(frozen=True)
class ControlCommand:
    """One safe command produced by the control loop."""

    action: BimanualAction
    held: bool = False
    fallback: bool = False


def smoothstep_alpha(step: int, blend_steps: int) -> float:
    """Return smoothstep alpha for a 1-based blend step."""
    if blend_steps <= 0:
        return 1.0
    s = float(np.clip(float(step) / float(blend_steps), 0.0, 1.0))
    return float(3.0 * s * s - 2.0 * s * s * s)


def is_action_chunk_usable(
    chunk: ActionChunk,
    *,
    now: float,
    action_dim: int,
    max_action_age_s: float,
) -> tuple[bool, str | None]:
    """Validate chunk freshness, shape, alignment, and numeric values."""
    if chunk.actions.ndim != 2:
        return False, f"invalid rank {chunk.actions.ndim}"
    if chunk.actions.shape[1] != int(action_dim):
        return False, f"invalid action_dim {chunk.actions.shape[1]}"
    if not np.all(np.isfinite(chunk.actions)):
        return False, "chunk contains NaN or Inf"
    age_s = float(now) - float(chunk.obs_time)
    if age_s > float(max_action_age_s):
        return False, f"chunk too old age_ms={age_s * 1000.0:.1f}"
    if chunk.aligned_index(now) >= max(0, chunk.chunk_size - 2):
        return False, "aligned index too close to chunk end"
    return True, None


class ControlLoop:
    """Consume active chunks, prefetch pending chunks, and blend at boundaries."""

    def __init__(
        self,
        *,
        shared_buffer: SharedBuffer,
        request_queue: LatestQueue[InferenceRequest],
        result_queue: LatestQueue[ActionChunk],
        observation_provider: ObservationProvider,
        safety_guard: SafetyGuard,
        control_hz: float,
        execute_horizon: int,
        prefetch_steps: int,
        blend_steps: int,
        action_dim: int,
        max_action_age_s: float,
        fallback_policy: str,
        stale_observation_timeout_s: float = 0.5,
        log_info: LogFn | None = None,
        log_warning: LogFn | None = None,
    ) -> None:
        self.shared_buffer = shared_buffer
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.observation_provider = observation_provider
        self.safety_guard = safety_guard
        self.control_hz = float(control_hz)
        self.execute_horizon = int(execute_horizon)
        self.prefetch_steps = int(prefetch_steps)
        self.blend_steps = int(blend_steps)
        self.action_dim = int(action_dim)
        self.max_action_age_s = float(max_action_age_s)
        self.fallback_policy = fallback_policy
        self.stale_observation_timeout_s = float(stale_observation_timeout_s)
        self.log_info = log_info or (lambda message: None)
        self.log_warning = log_warning or (lambda message: None)

        self.active_chunk: ActionChunk | None = None
        self.pending_chunk: ActionChunk | None = None
        self.active_cursor = 0
        self.last_command: BimanualAction | None = None
        self.request_pending = False
        self.blend_active = False
        self.blend_counter = 0
        self.blend_start_command: BimanualAction | None = None
        self.next_chunk: ActionChunk | None = None
        self.next_cursor = 0
        self.request_id = 0
        self._last_fallback_log_s = 0.0
        self._last_summary_log_s = 0.0

    def tick(self) -> ControlCommand | None:
        """Return the next safe command without waiting for model inference."""
        now = time.monotonic()
        self._collect_result(now)

        if self.active_chunk is None:
            self._activate_pending(now, immediate=True)

        self._maybe_submit_request(now)

        raw_action = self._next_raw_action(now)
        if raw_action is None:
            return self._fallback("no active action available")

        observation = self.shared_buffer.latest_observation(max_age_s=self.stale_observation_timeout_s)
        result = self.safety_guard.filter_action(
            raw_action,
            observation=observation,
            previous_action=self.last_command,
        )
        if not result.accepted or result.action is None:
            self.shared_buffer.record_rejected_action(result.reason or "action rejected")
            return self._fallback(result.reason or "action rejected")

        self.last_command = result.action
        self.shared_buffer.record_published_action()
        self._log_summary(now)
        return ControlCommand(action=result.action)

    def status_snapshot(self) -> dict[str, int | bool | None]:
        return {
            "active_request_id": None if self.active_chunk is None else self.active_chunk.request_id,
            "pending_request_id": None if self.pending_chunk is None else self.pending_chunk.request_id,
            "active_cursor": self.active_cursor,
            "request_pending": self.request_pending,
            "blend_active": self.blend_active,
            "request_queue_len": len(self.request_queue),
            "result_queue_len": len(self.result_queue),
        }

    def _collect_result(self, now: float) -> None:
        chunk = self.result_queue.get_latest_or_none()
        if chunk is None:
            return
        self.request_pending = False
        ok, reason = is_action_chunk_usable(
            chunk,
            now=now,
            action_dim=self.action_dim,
            max_action_age_s=self.max_action_age_s,
        )
        if not ok:
            message = f"discarding chunk request_id={chunk.request_id}: {reason}"
            self.shared_buffer.record_discarded_chunk(message)
            self.log_warning(message)
            return
        self.pending_chunk = chunk

    def _maybe_submit_request(self, now: float) -> None:
        if self.request_pending:
            return
        if self.pending_chunk is not None or self.blend_active:
            return
        if self.active_chunk is not None:
            trigger_cursor = max(0, self.execute_horizon - self.prefetch_steps)
            if self.active_cursor < trigger_cursor:
                return
        observation = self.observation_provider()
        if observation is None:
            self._log_fallback(now, "observation unavailable for inference request")
            return
        self.request_id += 1
        request = InferenceRequest(
            observation=observation,
            obs_time=observation.captured_at_s,
            request_id=self.request_id,
            trigger_step=self.active_cursor,
        )
        self.request_queue.put_latest(request)
        self.request_pending = True
        self.shared_buffer.record_inference_request()
        self.log_info(
            "submitted inference request "
            f"request_id={request.request_id} active_cursor={self.active_cursor} obs_time={request.obs_time:.6f}"
        )

    def _next_raw_action(self, now: float) -> np.ndarray | None:
        if self.blend_active:
            return self._blend_next_action()

        if self.active_chunk is None:
            return None

        if self.active_cursor >= self.execute_horizon and self.pending_chunk is not None:
            self._start_blend_or_switch(now)
            if self.blend_active:
                return self._blend_next_action()

        if self.active_cursor < self.active_chunk.chunk_size:
            action = self.active_chunk.actions[self.active_cursor]
            self.active_cursor += 1
            return action

        if self.pending_chunk is not None:
            self._start_blend_or_switch(now)
            if self.blend_active:
                return self._blend_next_action()
            if self.active_chunk is not None and self.active_cursor < self.active_chunk.chunk_size:
                action = self.active_chunk.actions[self.active_cursor]
                self.active_cursor += 1
                return action
        return None

    def _activate_pending(self, now: float, *, immediate: bool = False) -> None:
        if self.pending_chunk is None:
            return
        ok, reason = is_action_chunk_usable(
            self.pending_chunk,
            now=now,
            action_dim=self.action_dim,
            max_action_age_s=self.max_action_age_s,
        )
        if not ok:
            message = f"discarding pending chunk request_id={self.pending_chunk.request_id}: {reason}"
            self.shared_buffer.record_discarded_chunk(message)
            self.log_warning(message)
            self.pending_chunk = None
            return

        aligned = self.pending_chunk.aligned_index(now)
        old_cursor = self.active_cursor
        chunk_age_ms = (now - self.pending_chunk.obs_time) * 1000.0
        self.active_chunk = self.pending_chunk
        self.active_cursor = aligned
        self.pending_chunk = None
        self.blend_active = False
        self.shared_buffer.record_chunk_switch()
        self.log_info(
            "activated action chunk "
            f"old_cursor={old_cursor} new_aligned_idx={aligned} chunk_age_ms={chunk_age_ms:.1f} "
            f"blend_steps={0 if immediate else self.blend_steps}"
        )

    def _start_blend_or_switch(self, now: float) -> None:
        if self.pending_chunk is None:
            return
        ok, reason = is_action_chunk_usable(
            self.pending_chunk,
            now=now,
            action_dim=self.action_dim,
            max_action_age_s=self.max_action_age_s,
        )
        if not ok:
            message = f"discarding pending chunk request_id={self.pending_chunk.request_id}: {reason}"
            self.shared_buffer.record_discarded_chunk(message)
            self.log_warning(message)
            self.pending_chunk = None
            return

        aligned = self.pending_chunk.aligned_index(now)
        old_cursor = self.active_cursor
        chunk_age_ms = (now - self.pending_chunk.obs_time) * 1000.0
        if self.blend_steps <= 0 or self.last_command is None:
            self._activate_pending(now, immediate=True)
            return

        self.blend_active = True
        self.blend_counter = 0
        self.blend_start_command = self.last_command
        self.next_chunk = self.pending_chunk
        self.next_cursor = aligned
        self.pending_chunk = None
        self.shared_buffer.record_chunk_switch()
        self.log_info(
            "starting chunk blend "
            f"old_cursor={old_cursor} new_aligned_idx={aligned} chunk_age_ms={chunk_age_ms:.1f} "
            f"blend_steps={self.blend_steps}"
        )

    def _blend_next_action(self) -> np.ndarray | None:
        if self.next_chunk is None or self.blend_start_command is None:
            self.blend_active = False
            return None
        if self.next_cursor >= self.next_chunk.chunk_size:
            self.blend_active = False
            self.active_chunk = self.next_chunk
            self.active_cursor = self.next_cursor
            self.next_chunk = None
            return None

        self.blend_counter += 1
        alpha = smoothstep_alpha(self.blend_counter, self.blend_steps)
        new_action = self.next_chunk.actions[self.next_cursor]
        self.next_cursor += 1
        blended = (1.0 - alpha) * self.blend_start_command.as_vector() + alpha * new_action

        if self.blend_counter >= self.blend_steps:
            self.active_chunk = self.next_chunk
            self.active_cursor = self.next_cursor
            self.next_chunk = None
            self.blend_active = False
            self.blend_counter = 0
            self.blend_start_command = None
        return blended.astype(np.float32, copy=False)

    def _fallback(self, reason: str) -> ControlCommand | None:
        now = time.monotonic()
        self.shared_buffer.record_fallback(reason)
        self._log_fallback(now, f"{self.fallback_policy}: {reason}")

        if self.fallback_policy == "safe_stop":
            return None
        if self.fallback_policy in {"hold_last_action", "continue_old_chunk"} and self.last_command is not None:
            safe_result = self.safety_guard.filter_action(
                self.last_command.as_vector(),
                observation=self.shared_buffer.latest_observation(max_age_s=self.stale_observation_timeout_s),
                previous_action=self.last_command,
            )
            if safe_result.accepted and safe_result.action is not None:
                self.last_command = safe_result.action
                self.shared_buffer.record_held_action()
                return ControlCommand(action=safe_result.action, held=True, fallback=True)
        return None

    def _log_fallback(self, now: float, reason: str) -> None:
        if now - self._last_fallback_log_s >= 1.0:
            self.log_warning(f"fallback triggered fallback_policy={self.fallback_policy} reason={reason}")
            self._last_fallback_log_s = now

    def _log_summary(self, now: float) -> None:
        if now - self._last_summary_log_s < 1.0:
            return
        self.log_info(
            "control status "
            f"active_cursor={self.active_cursor} request_pending={self.request_pending} "
            f"blend_active={self.blend_active}"
        )
        self._last_summary_log_s = now
