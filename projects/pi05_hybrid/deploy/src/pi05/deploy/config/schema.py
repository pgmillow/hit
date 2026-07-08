"""Typed YAML configuration for Pi0.5 deployment.

Deployment config is intentionally separate from training config. Real robot
deployment consumes only an exported bundle plus runtime/ROS settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from pi05.common.ros.topics import Pi05CommandTopics, Pi05ObservationTopics


class DeployConfigError(ValueError):
    """Raised when deploy YAML is missing required fields or has invalid values."""


@dataclass(frozen=True)
class BundleConfig:
    """Runtime model bundle location."""

    bundle_dir: Path

    @property
    def resolved_bundle_dir(self) -> Path:
        return self.bundle_dir.expanduser().resolve()


@dataclass(frozen=True)
class RuntimeConfig:
    """Policy runtime scheduling and device settings."""

    mode: str = "dry-run"
    device: str = "cuda:0"
    secondary_device: str | None = None
    multi_gpu_strategy: str = "none"
    dtype: str = "bfloat16"
    inference_hz: float = 10.0
    control_hz: float = 30.0
    chunk_size: int = 30
    execute_horizon: int = 10
    prefetch_steps: int = 5
    blend_steps: int = 3
    action_dim: int = 14
    state_dim: int = 26
    max_action_age_sec: float = 0.45
    max_inference_requests: int = 1
    max_pending_chunks: int = 1
    fallback_policy: str = "hold_last_action"
    max_delta_per_step: float = 0.03
    warmup_steps: int = 2
    compile_model: bool = True
    compile_mode: str = "reduce-overhead"
    publish_metrics_hz: float = 1.0
    task: str = "bimanual manipulation"

    @property
    def publishes_command_topics(self) -> bool:
        return self.mode in {"shadow-run", "safe-run"}

    def __post_init__(self) -> None:
        if self.control_hz <= 0.0:
            raise DeployConfigError("runtime.control_hz must be positive.")
        if self.inference_hz <= 0.0:
            raise DeployConfigError("runtime.inference_hz must be positive.")
        if self.prefetch_steps < 0:
            raise DeployConfigError("runtime.prefetch_steps must be >= 0.")
        if self.blend_steps < 0:
            raise DeployConfigError("runtime.blend_steps must be >= 0.")
        if self.execute_horizon > self.chunk_size:
            raise DeployConfigError(
                f"runtime.execute_horizon ({self.execute_horizon}) must be <= runtime.chunk_size ({self.chunk_size})."
            )
        if self.prefetch_steps > self.execute_horizon:
            raise DeployConfigError(
                f"runtime.prefetch_steps ({self.prefetch_steps}) must be <= runtime.execute_horizon ({self.execute_horizon})."
            )
        if self.max_action_age_sec <= 0.0:
            raise DeployConfigError("runtime.max_action_age_sec must be positive.")
        if self.max_inference_requests < 1:
            raise DeployConfigError("runtime.max_inference_requests must be >= 1.")
        if self.max_pending_chunks < 1:
            raise DeployConfigError("runtime.max_pending_chunks must be >= 1.")
        if self.fallback_policy not in {"hold_last_action", "continue_old_chunk", "safe_stop"}:
            raise DeployConfigError(
                "runtime.fallback_policy must be one of: hold_last_action, continue_old_chunk, safe_stop."
            )


@dataclass(frozen=True)
class ObservationTopicsConfig:
    """ROS input topics consumed by the observation collector."""

    top_image: str
    left_wrist_image: str
    right_wrist_image: str
    top_image_raw: str
    left_wrist_image_raw: str
    right_wrist_image_raw: str
    proprioception: str
    left_hand_state: str
    right_hand_state: str
    left_ee_position: str
    left_ee_rpy: str
    right_ee_position: str
    right_ee_rpy: str
    proprioception_order: str = "right_left"


@dataclass(frozen=True)
class CommandTopicsConfig:
    """ROS command and telemetry topics produced by deployment."""

    left_arm_joint_target: str
    right_arm_joint_target: str
    left_hand_target: str
    right_hand_target: str
    status: str
    metrics: str


@dataclass(frozen=True)
class BridgeTopicsConfig:
    """Optional adapter topics for an existing execution stack."""

    left_arm_joint_target: str = "/vla/left_arm/safe_joint_target"
    right_arm_joint_target: str = "/vla/right_arm/safe_joint_target"
    left_hand_trigger: str = "/vla/left_hand/trigger"
    right_hand_trigger: str = "/vla/right_hand/trigger"
    left_deadman: str = "/vla/left_arm/deadman"
    right_deadman: str = "/vla/right_arm/deadman"


@dataclass(frozen=True)
class MuxTopicsConfig:
    """Topic set used by the teleop/VLA command multiplexer."""

    teleop_left_arm_joint_target: str = "/teleop/left_arm/safe_joint_target"
    teleop_right_arm_joint_target: str = "/teleop/right_arm/safe_joint_target"
    teleop_left_hand_trigger: str = "/xr/pico/left/trigger"
    teleop_right_hand_trigger: str = "/xr/pico/right/trigger"
    teleop_left_deadman: str = "/xr/pico/left/grip"
    teleop_right_deadman: str = "/xr/pico/right/grip"
    vla_left_arm_joint_target: str = "/vla/left_arm/safe_joint_target"
    vla_right_arm_joint_target: str = "/vla/right_arm/safe_joint_target"
    vla_left_hand_trigger: str = "/vla/left_hand/trigger"
    vla_right_hand_trigger: str = "/vla/right_hand/trigger"
    output_left_arm_joint_target: str = "/mux/left_arm/safe_joint_target"
    output_right_arm_joint_target: str = "/mux/right_arm/safe_joint_target"
    output_left_hand_trigger: str = "/mux/left_hand/trigger"
    output_right_hand_trigger: str = "/mux/right_hand/trigger"
    output_left_deadman: str = "/mux/left_arm/deadman"
    output_right_deadman: str = "/mux/right_arm/deadman"
    vla_enable: str = "/mux/enable_vla"
    status: str = "/mux/status"


@dataclass(frozen=True)
class TopicsConfig:
    """All ROS topic names used by deployment."""

    namespace: str
    observation: ObservationTopicsConfig
    command: CommandTopicsConfig
    bridge_output: BridgeTopicsConfig
    mux: MuxTopicsConfig


@dataclass(frozen=True)
class JointLimitsConfig:
    """Optional arm joint limits in radians."""

    enabled: bool = False
    left_min_rad: tuple[float, ...] = field(default_factory=tuple)
    left_max_rad: tuple[float, ...] = field(default_factory=tuple)
    right_min_rad: tuple[float, ...] = field(default_factory=tuple)
    right_max_rad: tuple[float, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SafetyConfig:
    """Runtime safety checks applied after policy inference."""

    max_joint_delta_rad: float = 0.08
    stale_observation_timeout_s: float = 0.5
    command_timeout_s: float = 0.45
    clamp_normalized_action: bool = True
    hold_last_action: bool = True
    hand_min: float = 300.0
    hand_max: float = 1000.0
    joint_limits: JointLimitsConfig = field(default_factory=JointLimitsConfig)


@dataclass(frozen=True)
class BridgeConfig:
    """Whether and how the bridge forwards Pi0.5 command topics."""

    enabled: bool = False
    publish_to_picotele: bool = False
    forward_commands: bool = False
    speed_scale: float = 0.25
    publish_deadman: bool = False

    @property
    def forwards_commands(self) -> bool:
        return self.enabled and (self.forward_commands or self.publish_to_picotele)


@dataclass(frozen=True)
class MuxConfig:
    """Runtime behavior for the teleop/VLA command multiplexer."""

    enabled: bool = False
    default_mode: str = "teleop"
    publish_hz: float = 60.0
    vla_command_timeout_s: float = 0.45
    manual_takeover_deadman_threshold: float = 0.3
    vla_deadman_value: float = 1.0
    status_publish_hz: float = 2.0


@dataclass(frozen=True)
class ImageConfig:
    """Deployment image preprocessing settings."""

    image_size: int = 224
    resize_mode: str = "resize_pad"
    transport: str = "raw"


@dataclass(frozen=True)
class DeployConfig:
    """Complete Pi0.5 deployment configuration."""

    bundle: BundleConfig
    runtime: RuntimeConfig
    image: ImageConfig
    topics: TopicsConfig
    safety: SafetyConfig
    bridge: BridgeConfig
    mux: MuxConfig
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path) -> "DeployConfig":
        return _deploy_from_mapping(cls, raw, base_dir=base_dir)


def load_deploy_config(path: str | Path) -> DeployConfig:
    """Load a deployment config from YAML."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise DeployConfigError(f"Deploy config root must be a mapping, got {type(raw).__name__}")
    return DeployConfig.from_mapping(raw, base_dir=config_path.parent)


def _deploy_from_mapping(cls: type[DeployConfig], raw: Mapping[str, Any], *, base_dir: Path) -> DeployConfig:
    root = _mapping(raw, "<root>")
    namespace = _str(_mapping(root.get("topics", {}), "topics"), "namespace", default="/pi05_vla")
    default_obs = Pi05ObservationTopics.with_namespace(namespace)
    default_cmd = Pi05CommandTopics.with_namespace(namespace)

    bundle_raw = _required_mapping(root, "bundle")
    runtime_raw = _mapping(root.get("runtime", {}), "runtime")
    image_raw = _mapping(root.get("image", {}), "image")
    topics_raw = _mapping(root.get("topics", {}), "topics")
    safety_raw = _mapping(root.get("safety", {}), "safety")
    bridge_raw = _mapping(root.get("bridge", {}), "bridge")
    mux_raw = _mapping(root.get("mux", {}), "mux")

    return cls(
        bundle=BundleConfig(bundle_dir=_path(bundle_raw, "bundle_dir", base_dir=base_dir)),
        runtime=RuntimeConfig(
            mode=_choice(runtime_raw, "mode", {"dry-run", "shadow-run", "safe-run"}, default="dry-run"),
            device=_str(runtime_raw, "device", default="cuda:0"),
            secondary_device=_optional_str(runtime_raw, "secondary_device"),
            multi_gpu_strategy=_choice(runtime_raw, "multi_gpu_strategy", {"none", "auto"}, default="none"),
            dtype=_choice(runtime_raw, "dtype", {"float32", "float16", "bfloat16"}, default="bfloat16"),
            inference_hz=_positive_float(runtime_raw, "inference_hz", default=10.0),
            control_hz=_positive_float(runtime_raw, "control_hz", default=30.0),
            chunk_size=_positive_int(runtime_raw, "chunk_size", default=30),
            execute_horizon=_positive_int(runtime_raw, "execute_horizon", default=10),
            prefetch_steps=_non_negative_int(runtime_raw, "prefetch_steps", default=5),
            blend_steps=_non_negative_int(runtime_raw, "blend_steps", default=_int_value(runtime_raw, "chunk_blend_steps", default=3)),
            action_dim=_positive_int(runtime_raw, "action_dim", default=14),
            state_dim=_positive_int(runtime_raw, "state_dim", default=26),
            max_action_age_sec=_positive_float(runtime_raw, "max_action_age_sec", default=0.45),
            max_inference_requests=_positive_int(runtime_raw, "max_inference_requests", default=1),
            max_pending_chunks=_positive_int(runtime_raw, "max_pending_chunks", default=1),
            fallback_policy=_choice(
                runtime_raw,
                "fallback_policy",
                {"hold_last_action", "continue_old_chunk", "safe_stop"},
                default="hold_last_action",
            ),
            max_delta_per_step=_positive_float(runtime_raw, "max_delta_per_step", default=0.03),
            warmup_steps=_non_negative_int(runtime_raw, "warmup_steps", default=2),
            compile_model=_bool(runtime_raw, "compile_model", default=True),
            compile_mode=_str(runtime_raw, "compile_mode", default="reduce-overhead"),
            publish_metrics_hz=_positive_float(runtime_raw, "publish_metrics_hz", default=1.0),
            task=_str(runtime_raw, "task", default="bimanual manipulation"),
        ),
        image=ImageConfig(
            image_size=_positive_int(image_raw, "image_size", default=224),
            resize_mode=_choice(image_raw, "resize_mode", {"resize_pad", "resize_crop"}, default="resize_pad"),
            transport=_choice(image_raw, "transport", {"raw", "compressed", "both"}, default="raw"),
        ),
        topics=TopicsConfig(
            namespace=namespace,
            observation=_observation_topics(topics_raw, default_obs),
            command=_command_topics(topics_raw, default_cmd),
            bridge_output=_bridge_topics(topics_raw),
            mux=_mux_topics(topics_raw),
        ),
        safety=_safety_config(safety_raw, runtime_raw=runtime_raw),
        bridge=BridgeConfig(
            enabled=_bool(bridge_raw, "enabled", default=False),
            publish_to_picotele=_bool(bridge_raw, "publish_to_picotele", default=False),
            forward_commands=_bool(
                bridge_raw,
                "forward_commands",
                default=_bool(bridge_raw, "publish_to_picotele", default=False),
            ),
            speed_scale=_float(bridge_raw, "speed_scale", default=0.25),
            publish_deadman=_bool(bridge_raw, "publish_deadman", default=False),
        ),
        mux=_mux_config(mux_raw),
        raw=dict(root),
    )

def _observation_topics(topics_raw: Mapping[str, Any], defaults: Pi05ObservationTopics) -> ObservationTopicsConfig:
    raw = _mapping(topics_raw.get("observation", {}), "topics.observation")
    return ObservationTopicsConfig(
        top_image=_str(raw, "top_image", default=defaults.top_image),
        left_wrist_image=_str(raw, "left_wrist_image", default=defaults.left_wrist_image),
        right_wrist_image=_str(raw, "right_wrist_image", default=defaults.right_wrist_image),
        top_image_raw=_str(raw, "top_image_raw", default=defaults.top_image.removesuffix("/compressed")),
        left_wrist_image_raw=_str(raw, "left_wrist_image_raw", default=defaults.left_wrist_image.removesuffix("/compressed")),
        right_wrist_image_raw=_str(raw, "right_wrist_image_raw", default=defaults.right_wrist_image.removesuffix("/compressed")),
        proprioception=_str(raw, "proprioception", default=defaults.proprioception),
        left_hand_state=_str(raw, "left_hand_state", default=defaults.left_hand_state),
        right_hand_state=_str(raw, "right_hand_state", default=defaults.right_hand_state),
        left_ee_position=_str(raw, "left_ee_position", default=defaults.left_ee_position),
        left_ee_rpy=_str(raw, "left_ee_rpy", default=defaults.left_ee_rpy),
        right_ee_position=_str(raw, "right_ee_position", default=defaults.right_ee_position),
        right_ee_rpy=_str(raw, "right_ee_rpy", default=defaults.right_ee_rpy),
        proprioception_order=_choice(raw, "proprioception_order", {"right_left", "left_right"}, default="right_left"),
    )


def _command_topics(topics_raw: Mapping[str, Any], defaults: Pi05CommandTopics) -> CommandTopicsConfig:
    raw = _mapping(topics_raw.get("command", {}), "topics.command")
    return CommandTopicsConfig(
        left_arm_joint_target=_str(raw, "left_arm_joint_target", default=defaults.left_arm_joint_target),
        right_arm_joint_target=_str(raw, "right_arm_joint_target", default=defaults.right_arm_joint_target),
        left_hand_target=_str(raw, "left_hand_target", default=defaults.left_hand_target),
        right_hand_target=_str(raw, "right_hand_target", default=defaults.right_hand_target),
        status=_str(raw, "status", default=defaults.status),
        metrics=_str(raw, "metrics", default=defaults.metrics),
    )


def _bridge_topics(topics_raw: Mapping[str, Any]) -> BridgeTopicsConfig:
    raw = _mapping(topics_raw.get("bridge_output", {}), "topics.bridge_output")
    return BridgeTopicsConfig(
        left_arm_joint_target=_str(raw, "left_arm_joint_target", default="/vla/left_arm/safe_joint_target"),
        right_arm_joint_target=_str(raw, "right_arm_joint_target", default="/vla/right_arm/safe_joint_target"),
        left_hand_trigger=_str(raw, "left_hand_trigger", default="/vla/left_hand/trigger"),
        right_hand_trigger=_str(raw, "right_hand_trigger", default="/vla/right_hand/trigger"),
        left_deadman=_str(raw, "left_deadman", default="/vla/left_arm/deadman"),
        right_deadman=_str(raw, "right_deadman", default="/vla/right_arm/deadman"),
    )


def _mux_topics(topics_raw: Mapping[str, Any]) -> MuxTopicsConfig:
    raw = _mapping(topics_raw.get("mux", {}), "topics.mux")
    return MuxTopicsConfig(
        teleop_left_arm_joint_target=_str(
            raw,
            "teleop_left_arm_joint_target",
            default="/teleop/left_arm/safe_joint_target",
        ),
        teleop_right_arm_joint_target=_str(
            raw,
            "teleop_right_arm_joint_target",
            default="/teleop/right_arm/safe_joint_target",
        ),
        teleop_left_hand_trigger=_str(raw, "teleop_left_hand_trigger", default="/xr/pico/left/trigger"),
        teleop_right_hand_trigger=_str(raw, "teleop_right_hand_trigger", default="/xr/pico/right/trigger"),
        teleop_left_deadman=_str(raw, "teleop_left_deadman", default="/xr/pico/left/grip"),
        teleop_right_deadman=_str(raw, "teleop_right_deadman", default="/xr/pico/right/grip"),
        vla_left_arm_joint_target=_str(raw, "vla_left_arm_joint_target", default="/vla/left_arm/safe_joint_target"),
        vla_right_arm_joint_target=_str(raw, "vla_right_arm_joint_target", default="/vla/right_arm/safe_joint_target"),
        vla_left_hand_trigger=_str(raw, "vla_left_hand_trigger", default="/vla/left_hand/trigger"),
        vla_right_hand_trigger=_str(raw, "vla_right_hand_trigger", default="/vla/right_hand/trigger"),
        output_left_arm_joint_target=_str(
            raw,
            "output_left_arm_joint_target",
            default="/mux/left_arm/safe_joint_target",
        ),
        output_right_arm_joint_target=_str(
            raw,
            "output_right_arm_joint_target",
            default="/mux/right_arm/safe_joint_target",
        ),
        output_left_hand_trigger=_str(raw, "output_left_hand_trigger", default="/mux/left_hand/trigger"),
        output_right_hand_trigger=_str(raw, "output_right_hand_trigger", default="/mux/right_hand/trigger"),
        output_left_deadman=_str(raw, "output_left_deadman", default="/mux/left_arm/deadman"),
        output_right_deadman=_str(raw, "output_right_deadman", default="/mux/right_arm/deadman"),
        vla_enable=_str(raw, "vla_enable", default="/mux/enable_vla"),
        status=_str(raw, "status", default="/mux/status"),
    )


def _safety_config(raw: Mapping[str, Any], *, runtime_raw: Mapping[str, Any]) -> SafetyConfig:
    limits_raw = _mapping(raw.get("joint_limits", {}), "safety.joint_limits")
    return SafetyConfig(
        max_joint_delta_rad=_float(
            raw,
            "max_joint_delta_rad",
            default=_float(runtime_raw, "max_delta_per_step", default=0.03),
        ),
        stale_observation_timeout_s=_positive_float(raw, "stale_observation_timeout_s", default=0.5),
        command_timeout_s=_positive_float(raw, "command_timeout_s", default=0.45),
        clamp_normalized_action=_bool(raw, "clamp_normalized_action", default=True),
        hold_last_action=_bool(raw, "hold_last_action", default=True),
        hand_min=_float(raw, "hand_min", default=300.0),
        hand_max=_float(raw, "hand_max", default=1000.0),
        joint_limits=JointLimitsConfig(
            enabled=_bool(limits_raw, "enabled", default=False),
            left_min_rad=tuple(_float_list(limits_raw, "left_min_rad", default=[])),
            left_max_rad=tuple(_float_list(limits_raw, "left_max_rad", default=[])),
            right_min_rad=tuple(_float_list(limits_raw, "right_min_rad", default=[])),
            right_max_rad=tuple(_float_list(limits_raw, "right_max_rad", default=[])),
        ),
    )


def _mux_config(raw: Mapping[str, Any]) -> MuxConfig:
    return MuxConfig(
        enabled=_bool(raw, "enabled", default=False),
        default_mode=_choice(raw, "default_mode", {"teleop", "vla"}, default="teleop"),
        publish_hz=_positive_float(raw, "publish_hz", default=60.0),
        vla_command_timeout_s=_positive_float(raw, "vla_command_timeout_s", default=0.45),
        manual_takeover_deadman_threshold=_float(raw, "manual_takeover_deadman_threshold", default=0.3),
        vla_deadman_value=_float(raw, "vla_deadman_value", default=1.0),
        status_publish_hz=_positive_float(raw, "status_publish_hz", default=2.0),
    )


def _required_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in raw:
        raise DeployConfigError(f"Missing required deploy config section: {key}")
    return _mapping(raw[key], key)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeployConfigError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def _path(raw: Mapping[str, Any], key: str, *, base_dir: Path) -> Path:
    value = _str(raw, key)
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path)


def _str(raw: Mapping[str, Any], key: str, default: str | None = None) -> str:
    if key not in raw:
        if default is None:
            raise DeployConfigError(f"Missing required deploy config value: {key}")
        return default
    value = raw[key]
    if not isinstance(value, str):
        raise DeployConfigError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    value = raw.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise DeployConfigError(f"{key} must be a string or null, got {type(value).__name__}")
    return value


def _choice(raw: Mapping[str, Any], key: str, choices: set[str], default: str) -> str:
    value = _str(raw, key, default=default)
    if value not in choices:
        raise DeployConfigError(f"{key} must be one of {sorted(choices)}, got {value!r}")
    return value


def _bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    raise DeployConfigError(f"{key} must be a boolean, got {value!r}")


def _positive_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = int(raw.get(key, default))
    if value <= 0:
        raise DeployConfigError(f"{key} must be positive, got {value}")
    return value


def _non_negative_int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = int(raw.get(key, default))
    if value < 0:
        raise DeployConfigError(f"{key} must be non-negative, got {value}")
    return value


def _int_value(raw: Mapping[str, Any], key: str, default: int) -> int:
    return int(raw.get(key, default))


def _float(raw: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(raw.get(key, default))
    except (TypeError, ValueError) as exc:
        raise DeployConfigError(f"{key} must be a float") from exc


def _positive_float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = _float(raw, key, default)
    if value <= 0.0:
        raise DeployConfigError(f"{key} must be positive, got {value}")
    return value


def _float_list(raw: Mapping[str, Any], key: str, default: list[float]) -> list[float]:
    value = raw.get(key, default)
    if not isinstance(value, list):
        raise DeployConfigError(f"{key} must be a list of floats")
    return [float(item) for item in value]
