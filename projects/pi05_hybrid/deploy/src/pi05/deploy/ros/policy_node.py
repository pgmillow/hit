"""Compatibility wrapper for the new Pi0.5 VLA deployment node."""

from __future__ import annotations

from pi05.deploy.ros_nodes.pi05_vla_deploy_node import main


__all__ = ["main"]


if __name__ == "__main__":
    main()
