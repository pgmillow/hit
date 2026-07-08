"""ROS 2 bridge entry point for Pi0.5 command topics."""

from __future__ import annotations

from pi05.common.utils.paths import bootstrap_project_paths


def main() -> None:
    bootstrap_project_paths(include_project_src=False)
    from pi05.deploy.ros_nodes.pi05_bridge_node import main as run_bridge_node

    run_bridge_node()


__all__ = ["main"]


if __name__ == "__main__":
    main()
