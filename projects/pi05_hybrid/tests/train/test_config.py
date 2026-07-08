from __future__ import annotations

from pathlib import Path

import pytest

from pi05.common.config.schema import ConfigError, ExperimentConfig, load_experiment_config


def _minimal_config() -> dict:
    return {
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "data": {
            "dataset_path": "/tmp/dataset",
            "chunk_size": 25,
        },
        "model": {
            "pretrained_path": "lerobot/pi05_base",
            "chunk_size": 25,
            "n_action_steps": 25,
            "max_action_dim": 14,
        },
        "training": {
            "batch_size": 2,
            "lr": 1.0e-4,
            "epochs": 1,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 0,
            "checkpoint_freq_epochs": 1,
        },
        "logging": {
            "project_name": "pi05",
            "run_name": "test",
            "output_dir": "/tmp/pi05_outputs/checkpoints",
            "export_dir": "/tmp/pi05_outputs/exports",
        },
    }


def test_config_accepts_existing_yaml_shape() -> None:
    config = ExperimentConfig.from_mapping(_minimal_config())
    assert config.data.chunk_size == 25
    assert config.model.action_dim == 14
    assert config.logging.tensorboard_port == 6006
    assert config.logging.log_freq == 200
    assert config.training.checkpoint_freq_steps == 2000
    assert config.tactile.enabled is False
    assert config.tactile.patch_count == 22
    assert config.logging.run_export_dir == Path("/tmp/pi05_outputs/exports/test")


def test_config_accepts_22_token_tactile_encoder() -> None:
    raw = _minimal_config()
    raw["tactile"] = {
        "enabled": True,
        "history_steps": 8,
        "history_stride": 2,
        "hidden_dim": 256,
        "attention_heads": 8,
    }

    config = ExperimentConfig.from_mapping(raw)

    assert config.tactile.enabled is True
    assert config.tactile.history_steps == 8
    assert config.tactile.patch_count == 22


def test_config_accepts_train_expert_only_with_regex_targets() -> None:
    raw = _minimal_config()
    raw["lora"]["target_modules"] = (
        r"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))"
    )
    raw["model"]["train_expert_only"] = True

    config = ExperimentConfig.from_mapping(raw)
    assert config.model.train_expert_only is True
    assert isinstance(config.lora.target_modules, str)


def test_config_rejects_ambiguous_string_bool() -> None:
    raw = _minimal_config()
    raw["data"]["use_color_jitter"] = "definitely"
    with pytest.raises(ConfigError, match="use_color_jitter"):
        ExperimentConfig.from_mapping(raw)


def test_load_experiment_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
lora:
  rank: 4
  alpha: 8
  dropout: 0.0
  target_modules: [q_proj]
data:
  dataset_path: /tmp/dataset
model:
  pretrained_path: lerobot/pi05_base
  train_expert_only: true
training:
  batch_size: 1
  lr: 0.0001
  epochs: 1
  gradient_accumulation_steps: 1
  warmup_steps: 0
  checkpoint_freq_epochs: 1
logging:
  project_name: pi05
  run_name: test
  output_dir: /tmp/out/checkpoints
  export_dir: /tmp/out/exports
  log_freq: 50
""",
        encoding="utf-8",
    )
    loaded = load_experiment_config(path)
    assert loaded.lora.rank == 4
    assert loaded.model.train_expert_only is True
    assert loaded.logging.log_freq == 50


def test_load_experiment_config_resolves_relative_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "data" / "processed" / "lerobot_data"
    dataset_dir.mkdir(parents=True)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    path.write_text(
        """
lora:
  rank: 4
  alpha: 8
  dropout: 0.0
  target_modules: [q_proj]
data:
  dataset_path: ../data/processed/lerobot_data
model:
  pretrained_path: lerobot/pi05_base
training:
  batch_size: 1
  lr: 0.0001
  epochs: 1
  gradient_accumulation_steps: 1
  warmup_steps: 0
  checkpoint_freq_epochs: 1
logging:
  project_name: pi05
  run_name: test
  output_dir: ../outputs/checkpoints
  export_dir: ../outputs/exports
""",
        encoding="utf-8",
    )

    loaded = load_experiment_config(path)
    assert loaded.data.resolved_dataset_path == dataset_dir.resolve()
    assert loaded.logging.run_output_dir == (tmp_path / "outputs" / "checkpoints" / "test").resolve()
