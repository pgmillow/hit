"""Small typed config layer for the YAML training configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from pi05.common.utils.paths import PROJECT_ROOT


class ConfigError(ValueError):
    """Raised when a YAML config is missing required fields or has invalid types."""


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: str | tuple[str, ...]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LoraConfig":
        return cls(
            rank=_positive_int(raw, "rank"),
            alpha=_positive_int(raw, "alpha"),
            dropout=_float(raw, "dropout", min_value=0.0, max_value=1.0),
            target_modules=_str_or_str_list(raw, "target_modules"),
        )


@dataclass(frozen=True)
class TactileConfig:
    enabled: bool = False
    history_steps: int = 8
    history_stride: int = 2
    patch_count: int = 22
    patch_size: int = 14
    input_channels: int = 1
    hidden_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    zero_init_gate: bool = True
    tactile_dropout_prob: float = 0.1

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TactileConfig":
        hidden_dim = _positive_int(raw, "hidden_dim", default=256)
        attention_heads = _positive_int(raw, "attention_heads", default=8)
        if hidden_dim % attention_heads != 0:
            raise ConfigError(
                f"tactile.hidden_dim ({hidden_dim}) must be divisible by "
                f"tactile.attention_heads ({attention_heads})."
            )
        patch_count = _positive_int(raw, "patch_count", default=22)
        patch_size = _positive_int(raw, "patch_size", default=14)
        input_channels = _positive_int(raw, "input_channels", default=1)
        if patch_count != 22 or patch_size != 14 or input_channels != 1:
            raise ConfigError(
                "The current tactile dataset layout requires patch_count=22, "
                "patch_size=14, and input_channels=1."
            )
        return cls(
            enabled=_bool(raw, "enabled", default=False),
            history_steps=_positive_int(raw, "history_steps", default=8),
            history_stride=_positive_int(raw, "history_stride", default=2),
            patch_count=patch_count,
            patch_size=patch_size,
            input_channels=input_channels,
            hidden_dim=hidden_dim,
            transformer_layers=_positive_int(raw, "transformer_layers", default=4),
            attention_heads=attention_heads,
            mlp_ratio=_positive_int(raw, "mlp_ratio", default=4),
            dropout=_float({**{"dropout": 0.1}, **raw}, "dropout", min_value=0.0, max_value=1.0),
            zero_init_gate=_bool(raw, "zero_init_gate", default=True),
            tactile_dropout_prob=_float(
                {**{"tactile_dropout_prob": 0.1}, **raw},
                "tactile_dropout_prob",
                min_value=0.0,
                max_value=1.0,
            ),
        )


@dataclass(frozen=True)
class DataConfig:
    dataset_path: Path
    fps: int = 60
    num_workers: int = 4
    chunk_size: int = 30
    use_color_jitter: bool = True
    image_size: int = 224
    state_dim: int = 26
    action_dim: int = 14
    cameras: tuple[str, ...] = ("top", "left_wrist", "right_wrist")
    features: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "DataConfig":
        return cls(
            dataset_path=_path(raw, "dataset_path", base_dir=base_dir),
            fps=_positive_int(raw, "fps", default=60),
            num_workers=_non_negative_int(raw, "num_workers", default=4),
            chunk_size=_positive_int(raw, "chunk_size", default=30),
            use_color_jitter=_bool(raw, "use_color_jitter", default=True),
            image_size=_positive_int(raw, "image_size", default=224),
            state_dim=_positive_int(raw, "state_dim", default=26),
            action_dim=_positive_int(raw, "action_dim", default=14),
            cameras=tuple(_str_list(raw, "cameras", default=["top", "left_wrist", "right_wrist"])),
            features=_mapping(raw, "features", default={}),
        )

    @property
    def resolved_dataset_path(self) -> Path:
        return self.dataset_path.expanduser().resolve()


@dataclass(frozen=True)
class ModelConfig:
    pretrained_path: str | Path
    device: str = "auto"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    train_expert_only: bool = False
    allow_random_init_peft: bool = False
    chunk_size: int = 30
    n_action_steps: int = 30
    state_dim: int = 26
    action_dim: int = 14
    max_action_dim: int = 14
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    attention_implementation: str = "sdpa"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "ModelConfig":
        chunk_size = _positive_int(raw, "chunk_size", default=30)
        return cls(
            pretrained_path=_str_or_path(raw, "pretrained_path", default="lerobot/pi05_base", base_dir=base_dir),
            device=_str(raw, "device", default="auto"),
            dtype=_str(raw, "dtype", default="bfloat16"),
            gradient_checkpointing=_bool(raw, "gradient_checkpointing", default=True),
            train_expert_only=_bool(raw, "train_expert_only", default=False),
            allow_random_init_peft=_bool(raw, "allow_random_init_peft", default=False),
            chunk_size=chunk_size,
            n_action_steps=_positive_int(raw, "n_action_steps", default=chunk_size),
            state_dim=_positive_int(raw, "state_dim", default=26),
            action_dim=_positive_int(raw, "action_dim", default=14),
            max_action_dim=_positive_int(raw, "max_action_dim", default=14),
            paligemma_variant=_str(raw, "paligemma_variant", default="gemma_2b"),
            action_expert_variant=_str(raw, "action_expert_variant", default="gemma_300m"),
            attention_implementation=_attention_implementation(raw, "attention_implementation", default="sdpa"),
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    lr: float
    epochs: int
    gradient_accumulation_steps: int
    warmup_steps: int
    checkpoint_freq_epochs: int
    checkpoint_freq_steps: int = 2000
    max_steps_per_epoch: int | None = None
    seed: int | None = None
    grad_clip_norm: float | None = None
    resume_from_checkpoint: Path | None = None
    init_adapter_from: Path | None = None
    mixed_precision: str | None = None
    lr_control_file: Path | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "TrainingConfig":
        grad_clip_norm = _optional_float(raw, "grad_clip_norm", min_value=0.0)
        mixed_precision = _optional_str(raw, "mixed_precision")
        if mixed_precision not in (None, "no", "fp16", "bf16", "fp8"):
            raise ConfigError(
                "training.mixed_precision must be one of: no, fp16, bf16, fp8"
            )
        return cls(
            batch_size=_positive_int(raw, "batch_size"),
            lr=_float(raw, "lr", min_value=0.0),
            epochs=_positive_int(raw, "epochs"),
            gradient_accumulation_steps=_positive_int(raw, "gradient_accumulation_steps"),
            warmup_steps=_non_negative_int(raw, "warmup_steps"),
            checkpoint_freq_epochs=_positive_int(raw, "checkpoint_freq_epochs"),
            checkpoint_freq_steps=_positive_int(raw, "checkpoint_freq_steps", default=2000),
            max_steps_per_epoch=_optional_int(raw, "max_steps_per_epoch"),
            seed=_optional_int(raw, "seed"),
            grad_clip_norm=grad_clip_norm,
            resume_from_checkpoint=_optional_path(raw, "resume_from_checkpoint", base_dir=base_dir),
            init_adapter_from=_optional_path(raw, "init_adapter_from", base_dir=base_dir),
            mixed_precision=mixed_precision,
            lr_control_file=_optional_path(raw, "lr_control_file", base_dir=base_dir),
        )


@dataclass(frozen=True)
class LoggingConfig:
    project_name: str
    run_name: str
    output_dir: Path
    export_dir: Path | None = None
    use_tensorboard: bool = True
    tensorboard_dir: Path | None = None
    tensorboard_auto_launch: bool = True
    tensorboard_host: str = "localhost"
    tensorboard_port: int = 6006
    log_freq: int = 200

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "LoggingConfig":
        output_dir = _path(raw, "output_dir", base_dir=base_dir)
        default_export_dir = output_dir.parent / "exports" if output_dir.name == "checkpoints" else output_dir / "exports"
        return cls(
            project_name=_str(raw, "project_name"),
            run_name=_str(raw, "run_name"),
            output_dir=output_dir,
            export_dir=_optional_path(raw, "export_dir", base_dir=base_dir) or default_export_dir,
            use_tensorboard=_bool(raw, "use_tensorboard", default=True),
            tensorboard_dir=_optional_path(raw, "tensorboard_dir", base_dir=base_dir) or output_dir / "tensorboard",
            tensorboard_auto_launch=_bool(raw, "tensorboard_auto_launch", default=True),
            tensorboard_host=_str(raw, "tensorboard_host", default="localhost"),
            tensorboard_port=_port(raw, "tensorboard_port", default=6006),
            log_freq=_positive_int(raw, "log_freq", default=200),
        )

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir.expanduser().resolve()

    @property
    def run_output_dir(self) -> Path:
        return self.resolved_output_dir / self.run_name

    @property
    def resolved_export_dir(self) -> Path:
        assert self.export_dir is not None
        return self.export_dir.expanduser().resolve()

    @property
    def run_export_dir(self) -> Path:
        return self.resolved_export_dir / self.run_name

    @property
    def resolved_tensorboard_dir(self) -> Path:
        assert self.tensorboard_dir is not None
        return self.tensorboard_dir.expanduser().resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    lora: LoraConfig
    tactile: TactileConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, base_dir: Path | None = None) -> "ExperimentConfig":
        root = _mapping_value(raw, "<root>")
        return cls(
            lora=LoraConfig.from_mapping(_required_section(root, "lora")),
            tactile=TactileConfig.from_mapping(_mapping(root, "tactile", default={})),
            data=DataConfig.from_mapping(_required_section(root, "data"), base_dir=base_dir),
            model=ModelConfig.from_mapping(_required_section(root, "model"), base_dir=base_dir),
            training=TrainingConfig.from_mapping(_required_section(root, "training"), base_dir=base_dir),
            logging=LoggingConfig.from_mapping(_required_section(root, "logging"), base_dir=base_dir),
            raw=dict(root),
        )

    def to_tracker_config(self) -> dict[str, Any]:
        tracker_cfg: dict[str, Any] = {}
        for section, value in self.raw.items():
            if not isinstance(value, Mapping):
                tracker_cfg[section] = _serialize_tracker_value(value)
                continue
            for key, sub_value in value.items():
                tracker_cfg[f"{section}.{key}"] = _serialize_tracker_value(sub_value)
        return tracker_cfg

    def run_summary(self) -> dict[str, Any]:
        return {
            "run": self.logging.run_name,
            "dataset": str(self.data.resolved_dataset_path),
            "pretrained": str(self.model.pretrained_path),
            "train_expert_only": self.model.train_expert_only,
            "epochs": self.training.epochs,
            "batch_size": self.training.batch_size,
            "grad_accumulation": self.training.gradient_accumulation_steps,
            "checkpoint_freq_steps": self.training.checkpoint_freq_steps,
            "max_steps_per_epoch": self.training.max_steps_per_epoch,
            "lr_control_file": (
                str(self.training.lr_control_file) if self.training.lr_control_file is not None else None
            ),
            "chunk_size": self.data.chunk_size,
            "tactile_enabled": self.tactile.enabled,
            "tactile_history_steps": self.tactile.history_steps,
            "tactile_history_stride": self.tactile.history_stride,
            "tactile_hidden_dim": self.tactile.hidden_dim,
            "state_dim": self.data.state_dim,
            "action_dim": self.data.action_dim,
            "output_dir": str(self.logging.run_output_dir),
            "export_dir": str(self.logging.run_export_dir),
            "log_freq": self.logging.log_freq,
        }


def load_experiment_config(path: Path) -> ExperimentConfig:
    config_path = path.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config root must be a YAML mapping, got {type(raw).__name__}.")
    return ExperimentConfig.from_mapping(raw, base_dir=config_path.parent)


def _required_section(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in raw:
        raise ConfigError(f"Missing required config section: {key}")
    value = raw[key]
    if not isinstance(value, Mapping):
        raise ConfigError(f"Config section '{key}' must be a mapping, got {type(value).__name__}.")
    return value


def _mapping_value(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{key} must be a mapping, got {type(value).__name__}.")
    return value


def _mapping(raw: Mapping[str, Any], key: str, default: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if key not in raw:
        if default is not None:
            return default
        raise ConfigError(f"Missing required field: {key}")
    return _mapping_value(raw[key], key)


def _str(raw: Mapping[str, Any], key: str, default: str | None = None) -> str:
    if key not in raw:
        if default is not None:
            return default
        raise ConfigError(f"Missing required field: {key}")
    value = raw[key]
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string, got {type(value).__name__}.")
    value = os.path.expandvars(value.strip())
    if not value:
        raise ConfigError(f"{key} must not be empty.")
    return value


def _attention_implementation(raw: Mapping[str, Any], key: str, default: str = "sdpa") -> str:
    value = _str(raw, key, default=default)
    allowed = {"eager", "sdpa", "flash_attention_2"}
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {sorted(allowed)}, got {value!r}.")
    return value


def _optional_str(raw: Mapping[str, Any], key: str) -> str | None:
    if key not in raw or raw[key] in (None, ""):
        return None
    return _str(raw, key)


def _str_or_str_list(raw: Mapping[str, Any], key: str) -> str | tuple[str, ...]:
    if key not in raw:
        raise ConfigError(f"Missing required field: {key}")
    value = raw[key]
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ConfigError(f"{key} must not be empty.")
        return normalized
    return tuple(_str_list(raw, key))


def _str_or_path(
    raw: Mapping[str, Any],
    key: str,
    default: str | None = None,
    *,
    base_dir: Path | None = None,
) -> str | Path:
    value = _str(raw, key, default=default)
    if value.startswith(("/", "~", "./", "../")):
        return _resolve_path_value(value, base_dir=base_dir)
    return value


def _path(raw: Mapping[str, Any], key: str, *, base_dir: Path | None = None) -> Path:
    return _resolve_path_value(_str(raw, key), base_dir=base_dir)


def _optional_path(raw: Mapping[str, Any], key: str, *, base_dir: Path | None = None) -> Path | None:
    value = _optional_str(raw, key)
    return _resolve_path_value(value, base_dir=base_dir) if value is not None else None


def _resolve_path_value(value: str, *, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = base_dir or PROJECT_ROOT
    return (anchor / path).resolve()


def _bool(raw: Mapping[str, Any], key: str, default: bool | None = None) -> bool:
    if key not in raw:
        if default is not None:
            return default
        raise ConfigError(f"Missing required field: {key}")
    value = raw[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    raise ConfigError(f"{key} must be a boolean, got {value!r}.")


def _positive_int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = _int(raw, key, default=default)
    if value <= 0:
        raise ConfigError(f"{key} must be > 0, got {value}.")
    return value


def _non_negative_int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = _int(raw, key, default=default)
    if value < 0:
        raise ConfigError(f"{key} must be >= 0, got {value}.")
    return value


def _optional_int(raw: Mapping[str, Any], key: str) -> int | None:
    if key not in raw or raw[key] is None:
        return None
    return _int(raw, key)


def _int(raw: Mapping[str, Any], key: str, default: int | None = None) -> int:
    if key not in raw:
        if default is not None:
            return default
        raise ConfigError(f"Missing required field: {key}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer, got {value!r}.")
    return value


def _float(
    raw: Mapping[str, Any],
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    if key not in raw:
        raise ConfigError(f"Missing required field: {key}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number, got {value!r}.")
    float_value = float(value)
    if min_value is not None and float_value < min_value:
        raise ConfigError(f"{key} must be >= {min_value}, got {float_value}.")
    if max_value is not None and float_value > max_value:
        raise ConfigError(f"{key} must be <= {max_value}, got {float_value}.")
    return float_value


def _optional_float(
    raw: Mapping[str, Any],
    key: str,
    *,
    min_value: float | None = None,
) -> float | None:
    if key not in raw or raw[key] is None:
        return None
    return _float(raw, key, min_value=min_value)


def _str_list(raw: Mapping[str, Any], key: str, default: list[str] | None = None) -> list[str]:
    if key not in raw:
        if default is not None:
            return list(default)
        raise ConfigError(f"Missing required field: {key}")
    value = raw[key]
    if not isinstance(value, list):
        raise ConfigError(f"{key} must be a list of strings, got {type(value).__name__}.")
    items: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{key}[{idx}] must be a non-empty string.")
        items.append(item.strip())
    if not items:
        raise ConfigError(f"{key} must not be empty.")
    return items


def _port(raw: Mapping[str, Any], key: str, default: int) -> int:
    port = _positive_int(raw, key, default=default)
    if port > 65535:
        raise ConfigError(f"{key} must be <= 65535, got {port}.")
    return port


def _serialize_tracker_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if value is None:
        return "None"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set, dict)):
        return yaml.safe_dump(value, sort_keys=False).strip()
    return str(value)
