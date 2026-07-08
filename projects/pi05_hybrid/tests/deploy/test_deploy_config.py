from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest

from pi05.common.data.action_codec import ensure_action_chunk, split_action
from pi05.common.data.state_codec import BimanualState, encode_bimanual_state
from pi05.deploy.config import DeployConfigError, load_deploy_config
from pi05.deploy.runtime.control_loop import ControlLoop, is_action_chunk_usable, smoothstep_alpha
from pi05.deploy.runtime.safety_guard import SafetyGuard
from pi05.deploy.runtime.shared_buffer import ActionChunk, LatestQueue, SharedBuffer


def test_load_deploy_config_resolves_bundle_path(tmp_path: Path) -> None:
    config_path = tmp_path / "deploy.yaml"
    bundle_dir = tmp_path / "outputs" / "exports" / "run"
    bundle_dir.mkdir(parents=True)
    config_path.write_text(
        """
bundle:
  bundle_dir: outputs/exports/run
runtime:
  mode: shadow-run
  inference_hz: 10
  control_hz: 30
  chunk_size: 30
  execute_horizon: 30
topics:
  namespace: /pi05_vla
bridge:
  enabled: false
""",
        encoding="utf-8",
    )

    config = load_deploy_config(config_path)

    assert config.bundle.resolved_bundle_dir == bundle_dir.resolve()
    assert config.runtime.mode == "shadow-run"
    assert config.runtime.prefetch_steps == 5
    assert config.runtime.publishes_command_topics is True
    assert config.topics.command.left_arm_joint_target == "/pi05_vla/command/left_arm/joint_target"
    assert config.topics.bridge_output.left_arm_joint_target == "/vla/left_arm/safe_joint_target"
    assert config.topics.mux.output_left_arm_joint_target == "/mux/left_arm/safe_joint_target"


def test_state_and_action_codecs_match_pi05_dims() -> None:
    state = BimanualState(
        left_arm_q=np.zeros(6, dtype=np.float32),
        right_arm_q=np.ones(6, dtype=np.float32),
        left_hand_q=1000.0,
        right_hand_q=300.0,
        left_ee_pos=np.zeros(3, dtype=np.float32),
        left_ee_rpy=np.zeros(3, dtype=np.float32),
        right_ee_pos=np.ones(3, dtype=np.float32),
        right_ee_rpy=np.ones(3, dtype=np.float32),
    )
    encoded = encode_bimanual_state(state)
    action_chunk = ensure_action_chunk(np.zeros((30, 14), dtype=np.float32))
    action = split_action(action_chunk[0])

    assert encoded.shape == (26,)
    assert action_chunk.shape == (30, 14)
    assert action.left_arm.shape == (6,)
    assert action.right_arm.shape == (6,)


def test_safety_guard_rejects_nan_and_clamps_delta() -> None:
    config_path = Path(__file__).parents[2] / "deploy" / "config" / "deploy.yaml"
    deploy_config = load_deploy_config(config_path)
    guard = SafetyGuard(deploy_config.safety)
    previous = split_action(np.zeros(14, dtype=np.float32))

    bad = np.zeros(14, dtype=np.float32)
    bad[0] = np.nan
    rejected = guard.filter_action(bad, observation=None, previous_action=previous)
    assert rejected.accepted is False

    large = np.ones(14, dtype=np.float32)
    accepted = guard.filter_action(large, observation=None, previous_action=previous)
    assert accepted.accepted is True
    assert accepted.action is not None
    assert np.all(np.abs(accepted.action.left_arm) <= deploy_config.safety.max_joint_delta_rad + 1e-6)


def test_latest_queue_keeps_only_latest() -> None:
    queue = LatestQueue[int](maxsize=1)
    queue.put_latest(1)
    queue.put_latest(2)

    assert queue.get_latest_or_none() == 2
    assert queue.empty()


def test_action_chunk_aligned_index_clamps() -> None:
    chunk = ActionChunk(
        actions=np.zeros((5, 14), dtype=np.float32),
        obs_time=10.0,
        infer_start_time=10.1,
        ready_time=10.2,
        action_dt=0.1,
        request_id=1,
    )

    assert chunk.aligned_index(9.9) == 0
    assert chunk.aligned_index(10.21) == 2
    assert chunk.aligned_index(99.0) == 4


def test_deploy_schema_rejects_invalid_horizons(tmp_path: Path) -> None:
    path = tmp_path / "deploy.yaml"
    path.write_text(
        """
bundle:
  bundle_dir: outputs/exports/run
runtime:
  chunk_size: 10
  execute_horizon: 11
""",
        encoding="utf-8",
    )
    with pytest.raises(DeployConfigError, match="execute_horizon"):
        load_deploy_config(path)

    path.write_text(
        """
bundle:
  bundle_dir: outputs/exports/run
runtime:
  chunk_size: 30
  execute_horizon: 10
  prefetch_steps: 11
""",
        encoding="utf-8",
    )
    with pytest.raises(DeployConfigError, match="prefetch_steps"):
        load_deploy_config(path)


def test_bridge_forward_commands_aliases_legacy_publish_flag(tmp_path: Path) -> None:
    path = tmp_path / "deploy.yaml"
    path.write_text(
        """
bundle:
  bundle_dir: outputs/exports/run
bridge:
  enabled: true
  publish_to_picotele: true
""",
        encoding="utf-8",
    )

    config = load_deploy_config(path)

    assert config.bridge.publish_to_picotele is True
    assert config.bridge.forward_commands is True
    assert config.bridge.forwards_commands is True


def test_mux_config_loads_topics_and_mode(tmp_path: Path) -> None:
    path = tmp_path / "deploy.yaml"
    path.write_text(
        """
bundle:
  bundle_dir: outputs/exports/run
topics:
  mux:
    vla_enable: /test/enable_vla
mux:
  enabled: true
  default_mode: vla
  vla_command_timeout_s: 0.25
""",
        encoding="utf-8",
    )

    config = load_deploy_config(path)

    assert config.mux.enabled is True
    assert config.mux.default_mode == "vla"
    assert config.mux.vla_command_timeout_s == pytest.approx(0.25)
    assert config.topics.mux.vla_enable == "/test/enable_vla"


def test_smoothstep_alpha_is_monotonic_and_ends_at_one() -> None:
    values = [smoothstep_alpha(step, 3) for step in (1, 2, 3)]

    assert values == sorted(values)
    assert values[-1] == pytest.approx(1.0)
    assert values[0] == pytest.approx(0.259259, rel=1e-4)


def test_action_chunk_usability_rejects_old_or_nan_chunks() -> None:
    now = 100.0
    good = ActionChunk(
        actions=np.zeros((30, 14), dtype=np.float32),
        obs_time=99.9,
        infer_start_time=99.95,
        ready_time=100.0,
        action_dt=1.0 / 30.0,
        request_id=1,
    )
    ok, reason = is_action_chunk_usable(good, now=now, action_dim=14, max_action_age_s=0.45)
    assert ok is True
    assert reason is None

    old = ActionChunk(
        actions=np.zeros((30, 14), dtype=np.float32),
        obs_time=99.0,
        infer_start_time=99.1,
        ready_time=99.2,
        action_dt=1.0 / 30.0,
        request_id=2,
    )
    ok, reason = is_action_chunk_usable(old, now=now, action_dim=14, max_action_age_s=0.45)
    assert ok is False
    assert "old" in str(reason)

    nan = ActionChunk(
        actions=np.zeros((30, 14), dtype=np.float32),
        obs_time=99.9,
        infer_start_time=99.95,
        ready_time=100.0,
        action_dt=1.0 / 30.0,
        request_id=3,
    )
    nan.actions[0, 0] = np.nan
    ok, reason = is_action_chunk_usable(nan, now=now, action_dim=14, max_action_age_s=0.45)
    assert ok is False
    assert "NaN" in str(reason)


def test_control_loop_does_not_block_when_pending_chunk_is_missing() -> None:
    deploy_config = load_deploy_config(Path(__file__).parents[2] / "deploy" / "config" / "deploy.yaml")
    shared = SharedBuffer(max_inference_requests=1, max_pending_chunks=1)
    guard = SafetyGuard(deploy_config.safety)
    loop = ControlLoop(
        shared_buffer=shared,
        request_queue=shared.inference_request_queue,
        result_queue=shared.chunk_result_queue,
        observation_provider=lambda: None,
        safety_guard=guard,
        control_hz=30,
        execute_horizon=2,
        prefetch_steps=1,
        blend_steps=1,
        action_dim=14,
        max_action_age_s=10.0,
        fallback_policy="hold_last_action",
    )
    loop.active_chunk = ActionChunk(
        actions=np.zeros((4, 14), dtype=np.float32),
        obs_time=time.monotonic(),
        infer_start_time=time.monotonic(),
        ready_time=time.monotonic(),
        action_dt=1.0 / 30.0,
        request_id=1,
    )

    started = time.perf_counter()
    first = loop.tick()
    second = loop.tick()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.02
    assert first is not None
    assert second is not None
