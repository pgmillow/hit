"""Path helpers for running the project from source or editable installs."""

from __future__ import annotations

import sys
from pathlib import Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (
            (parent / "pyproject.toml").exists()
            and (parent / "common" / "src").exists()
            and (parent / "train" / "src").exists()
            and (parent / "deploy" / "src").exists()
        ):
            return parent
    raise RuntimeError("Could not locate the project root from common/utils/paths.py")


def _find_workspace_root(project_root: Path) -> Path:
    for candidate in (project_root, *project_root.parents):
        if (candidate / "third_party" / "lerobot" / "src").exists():
            return candidate
    return project_root


PROJECT_ROOT = _find_project_root()
COMMON_SRC_ROOT = PROJECT_ROOT / "common" / "src"
TRAIN_SRC_ROOT = PROJECT_ROOT / "train" / "src"
DEPLOY_SRC_ROOT = PROJECT_ROOT / "deploy" / "src"
PROJECT_SRC_ROOTS = (
    COMMON_SRC_ROOT,
    TRAIN_SRC_ROOT,
    DEPLOY_SRC_ROOT,
)
WORKSPACE_ROOT = _find_workspace_root(PROJECT_ROOT)
THIRD_PARTY_LEROBOT_SRC = WORKSPACE_ROOT / "third_party" / "lerobot" / "src"


def bootstrap_project_paths(include_project_src: bool = True) -> None:
    """Make local source checkout imports explicit for non-installed entrypoints.

    Package modules should import by package name and avoid path mutation. This
    function is only for thin scripts such as ``train.py`` and tools under
    ``scripts/`` where the package might not be installed in editable mode.
    """
    candidates = []
    if include_project_src:
        candidates.extend(PROJECT_SRC_ROOTS)
    candidates.append(THIRD_PARTY_LEROBOT_SRC)

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def default_train_config_path() -> Path:
    return PROJECT_ROOT / "train" / "config" / "lora.yaml"


def default_data_config_path() -> Path:
    return PROJECT_ROOT / "train" / "config" / "data.yaml"


def default_deploy_config_path() -> Path:
    return PROJECT_ROOT / "deploy" / "config" / "deploy.yaml"
