"""Deploy bundle export/load helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pi05.common.config.schema import ExperimentConfig
from pi05.common.data.normalization import ActionStateNormalizer, build_state_action_normalizers


MANIFEST_NAME = "manifest.json"
NORMALIZERS_NAME = "normalizers.json"
EXPERIMENT_CONFIG_NAME = "experiment_config.yaml"
BUNDLE_SCHEMA_VERSION = 1


def export_deploy_bundle(
    config: ExperimentConfig,
    *,
    adapter_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Export the minimum runtime payload needed by deployment."""
    adapter_dir = adapter_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory does not exist: {adapter_dir}")

    _prepare_output_dir(output_dir, overwrite=overwrite)
    adapter_target_dir = output_dir / "adapter"
    shutil.copytree(adapter_dir, adapter_target_dir, dirs_exist_ok=overwrite)

    dataset = LeRobotDataset(
        repo_id=config.data.resolved_dataset_path.name,
        root=config.data.resolved_dataset_path,
    )
    state_normalizer, action_normalizer = build_state_action_normalizers(dataset)

    _write_yaml(output_dir / EXPERIMENT_CONFIG_NAME, dict(config.raw))
    _write_json(output_dir / NORMALIZERS_NAME, _normalizer_payload(state_normalizer, action_normalizer))
    _write_json(output_dir / MANIFEST_NAME, _manifest_payload(config))
    return output_dir


def load_bundle_manifest(bundle_dir: str | Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    manifest_path = bundle_dir / MANIFEST_NAME
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_bundle_normalizers(bundle_dir: str | Path) -> tuple[ActionStateNormalizer, ActionStateNormalizer]:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    normalizer_path = bundle_dir / NORMALIZERS_NAME
    with normalizer_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    state_payload = payload["state"]
    action_payload = payload["action"]
    state_normalizer = ActionStateNormalizer(
        min_vals=state_payload["min"],
        max_vals=state_payload["max"],
        identity_indices=state_payload.get("identity_indices"),
    )
    action_normalizer = ActionStateNormalizer(
        min_vals=action_payload["min"],
        max_vals=action_payload["max"],
        identity_indices=action_payload.get("identity_indices"),
    )
    return state_normalizer, action_normalizer


def resolve_bundle_adapter_dir(bundle_dir: str | Path) -> Path:
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    adapter_dir = bundle_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Bundle adapter directory does not exist: {adapter_dir}")
    return adapter_dir


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        has_content = any(output_dir.iterdir())
        if has_content and not overwrite:
            raise FileExistsError(
                f"Bundle output directory already exists and is not empty: {output_dir}. "
                "Pass overwrite=True to replace it."
            )
        if has_content and overwrite:
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _manifest_payload(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "project_name": config.logging.project_name,
            "run_name": config.logging.run_name,
        },
        "model": {
            "pretrained_path": str(config.model.pretrained_path),
            "dtype": config.model.dtype,
            "chunk_size": config.model.chunk_size,
            "n_action_steps": config.model.n_action_steps,
            "state_dim": config.model.state_dim,
            "action_dim": config.model.action_dim,
            "max_action_dim": config.model.max_action_dim,
        },
        "observation": {
            "fps": config.data.fps,
            "image_size": config.data.image_size,
            "cameras": list(config.data.cameras),
            "features": dict(config.data.features),
        },
        "artifacts": {
            "adapter_dir": "adapter",
            "normalizers_path": NORMALIZERS_NAME,
            "experiment_config_path": EXPERIMENT_CONFIG_NAME,
        },
    }


def _normalizer_payload(
    state_normalizer: ActionStateNormalizer,
    action_normalizer: ActionStateNormalizer,
) -> dict[str, Any]:
    return {
        "state": _single_normalizer_payload(state_normalizer),
        "action": _single_normalizer_payload(action_normalizer),
    }


def _single_normalizer_payload(normalizer: ActionStateNormalizer) -> dict[str, Any]:
    return {
        "min": normalizer.min_vals.tolist(),
        "max": normalizer.max_vals.tolist(),
        "identity_indices": _identity_indices(normalizer),
    }


def _identity_indices(normalizer: ActionStateNormalizer) -> list[int]:
    return [int(idx) for idx, is_identity in enumerate(normalizer.identity_mask.tolist()) if is_identity]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=False)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
