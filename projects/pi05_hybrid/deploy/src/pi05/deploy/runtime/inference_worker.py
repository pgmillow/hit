"""Request-driven background inference worker for Pi0.5 deployment."""

from __future__ import annotations

from collections.abc import Callable
import threading
import time

from pi05.deploy.runtime.shared_buffer import ActionChunk, InferenceRequest, LatestQueue, SharedBuffer


LogFn = Callable[[str], None]


class InferenceWorker(threading.Thread):
    """Consume inference requests and publish action chunks to a result queue."""

    def __init__(
        self,
        *,
        policy_runtime: object,
        request_queue: LatestQueue[InferenceRequest],
        result_queue: LatestQueue[ActionChunk],
        shared_buffer: SharedBuffer,
        inference_hz: float,
        control_hz: float,
        log_info: LogFn | None = None,
        log_debug: LogFn | None = None,
        log_warning: LogFn | None = None,
    ) -> None:
        super().__init__(daemon=True, name="pi05_inference_worker")
        self.policy_runtime = policy_runtime
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.shared_buffer = shared_buffer
        self.period_s = 1.0 / max(1e-6, float(inference_hz))
        self.action_dt = 1.0 / max(1e-6, float(control_hz))
        self.log_info = log_info or (lambda message: None)
        self.log_debug = log_debug or (lambda message: None)
        self.log_warning = log_warning or (lambda message: None)
        self._stop_event = threading.Event()
        self._last_infer_start_s = 0.0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            request = self.request_queue.get_latest_or_none()
            if request is None:
                self._stop_event.wait(0.001)
                continue

            now = time.monotonic()
            remaining = self.period_s - (now - self._last_infer_start_s)
            if remaining > 0.0:
                self._stop_event.wait(remaining)
                if self._stop_event.is_set():
                    return

            self._run_request(request)

    def _run_request(self, request: InferenceRequest) -> None:
        infer_start = time.monotonic()
        self._last_infer_start_s = infer_start
        try:
            actions = self.policy_runtime.predict_action_chunk(request.observation)
        except Exception as exc:  # pragma: no cover - exercised on robot.
            message = f"policy inference failed request_id={request.request_id}: {exc}"
            self.shared_buffer.record_inference_error(message)
            self.log_warning(message)
            return

        ready_time = time.monotonic()
        chunk = ActionChunk(
            actions=actions,
            obs_time=request.obs_time,
            infer_start_time=infer_start,
            ready_time=ready_time,
            action_dt=self.action_dt,
            request_id=request.request_id,
        )
        latency_s = ready_time - infer_start
        self.shared_buffer.record_inference_latency(latency_s)
        self.shared_buffer.record_chunk_result()
        self.result_queue.put_latest(chunk)
        self.log_debug(
            "policy inference complete "
            f"request_id={request.request_id} inference_time_ms={latency_s * 1000.0:.1f} "
            f"action_chunk_shape={tuple(chunk.actions.shape)}"
        )
