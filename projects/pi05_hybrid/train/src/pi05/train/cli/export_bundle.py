"""CLI for exporting a deployment-ready PI05 bundle from training artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from pi05.common.config.schema import load_experiment_config
from pi05.common.utils.paths import bootstrap_project_paths, default_train_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a deployment bundle for a trained PI05 adapter.")
    parser.add_argument("--config", type=Path, default=default_train_config_path(), help="YAML config path.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run directory containing final_adapter.")
    parser.add_argument("--adapter-dir", type=Path, default=None, help="Explicit adapter directory to package.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Deploy bundle output directory.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing bundle directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    bootstrap_project_paths(include_project_src=False)

    from pi05.common.runtime.bundle import export_deploy_bundle

    adapter_dir, output_dir = _resolve_paths(args, config=config)
    bundle_dir = export_deploy_bundle(
        config,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        overwrite=bool(args.overwrite),
    )
    print(f"[export] deploy bundle saved to: {bundle_dir}")


def _resolve_paths(args: argparse.Namespace, *, config) -> tuple[Path, Path]:
    run_dir = args.run_dir.expanduser().resolve() if args.run_dir is not None else None
    adapter_dir = args.adapter_dir.expanduser().resolve() if args.adapter_dir is not None else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else None

    if run_dir is None and adapter_dir is None:
        run_dir = config.logging.run_output_dir
    if adapter_dir is None:
        assert run_dir is not None
        adapter_dir = run_dir / "final_adapter"
    if output_dir is None:
        if run_dir is not None:
            output_dir = config.logging.run_export_dir
        else:
            output_dir = adapter_dir.parent / "deploy_bundle"
    return adapter_dir, output_dir


if __name__ == "__main__":
    main()
