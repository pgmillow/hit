"""ROS 2 deployment entry point for the Pi0.5 VLA runtime."""

from __future__ import annotations

from pi05.common.utils.paths import bootstrap_project_paths


def main() -> None:
    bootstrap_project_paths(include_project_src=False)
    from pi05.deploy.ros_nodes.pi05_vla_deploy_node import main as run_policy_node

    run_policy_node()


__all__ = ["main"]


if __name__ == "__main__":
    main()
