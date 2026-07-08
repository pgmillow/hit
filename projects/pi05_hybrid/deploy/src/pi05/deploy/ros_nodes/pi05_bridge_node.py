"""ROS 2 bridge from Pi0.5 command topics to an optional execution stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from pi05.common.robot.action_spec import ARM_DOF, ARM_JOINT_NAMES, hand_command_to_trigger
from pi05.deploy.config import load_deploy_config
from pi05.deploy.config.schema import DeployConfig


class Pi05BridgeNode(Node):
    """Adapt /pi05_vla/command topics to configured VLA candidate topics."""

    def __init__(self, config: DeployConfig) -> None:
        super().__init__("pi05_bridge_node")
        self.config = config
        self.left_last: np.ndarray | None = None
        self.right_last: np.ndarray | None = None
        self.enabled = bool(config.bridge.forwards_commands)

        command_topics = config.topics.command
        self.create_subscription(JointState, command_topics.left_arm_joint_target, lambda msg: self._arm_cb("left", msg), 10)
        self.create_subscription(JointState, command_topics.right_arm_joint_target, lambda msg: self._arm_cb("right", msg), 10)
        self.create_subscription(Float64, command_topics.left_hand_target, lambda msg: self._hand_cb("left", msg), 10)
        self.create_subscription(Float64, command_topics.right_hand_target, lambda msg: self._hand_cb("right", msg), 10)

        bridge_topics = config.topics.bridge_output
        self.left_arm_pub = self.create_publisher(JointState, bridge_topics.left_arm_joint_target, 10)
        self.right_arm_pub = self.create_publisher(JointState, bridge_topics.right_arm_joint_target, 10)
        self.left_hand_pub = self.create_publisher(Float64, bridge_topics.left_hand_trigger, 10)
        self.right_hand_pub = self.create_publisher(Float64, bridge_topics.right_hand_trigger, 10)
        self.left_deadman_pub = self.create_publisher(Float64, bridge_topics.left_deadman, 10)
        self.right_deadman_pub = self.create_publisher(Float64, bridge_topics.right_deadman, 10)
        if config.bridge.publish_deadman:
            self.deadman_timer = self.create_timer(1.0 / config.runtime.control_hz, self._publish_deadman)

        state = "enabled" if self.enabled else "disabled"
        self.get_logger().warning(
            f"Pi0.5 bridge started in {state} forwarding mode; "
            f"arm_out=({bridge_topics.left_arm_joint_target}, {bridge_topics.right_arm_joint_target}) "
            f"hand_out=({bridge_topics.left_hand_trigger}, {bridge_topics.right_hand_trigger})"
        )

    def _arm_cb(self, side: str, msg: JointState) -> None:
        if len(msg.position) < ARM_DOF:
            return
        target = np.asarray(msg.position[:ARM_DOF], dtype=np.float32)
        target = self._filter_joint_target(side, target)
        if target is None or not self.enabled:
            return
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = f"pi05_vla_{side}_arm_joint_target"
        out.name = list(ARM_JOINT_NAMES)
        out.position = [float(value) for value in target]
        out.velocity = [float(self.config.bridge.speed_scale), 0.0, 1.0]
        if side == "left":
            self.left_arm_pub.publish(out)
        else:
            self.right_arm_pub.publish(out)

    def _hand_cb(self, side: str, msg: Float64) -> None:
        trigger = hand_command_to_trigger(
            msg.data,
            open_value=self.config.safety.hand_max,
            closed_value=self.config.safety.hand_min,
        )
        if not self.enabled:
            return
        publisher = self.left_hand_pub if side == "left" else self.right_hand_pub
        publisher.publish(Float64(data=trigger))

    def _filter_joint_target(self, side: str, target: np.ndarray) -> np.ndarray | None:
        if not np.all(np.isfinite(target)):
            self.get_logger().warning(f"rejecting {side} arm command containing NaN or Inf")
            return None
        previous = self.left_last if side == "left" else self.right_last
        if previous is not None and self.config.safety.max_joint_delta_rad > 0.0:
            limit = float(self.config.safety.max_joint_delta_rad)
            target = previous + np.clip(target - previous, -limit, limit)
        if side == "left":
            self.left_last = target.astype(np.float32, copy=True)
        else:
            self.right_last = target.astype(np.float32, copy=True)
        return target

    def _publish_deadman(self) -> None:
        if not self.enabled:
            return
        value = Float64(data=1.0)
        self.left_deadman_pub.publish(value)
        self.right_deadman_pub.publish(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge Pi0.5 command topics to a downstream controller.")
    parser.add_argument("--config", type=Path, required=True, help="Path to deploy/config/deploy.yaml.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_deploy_config(args.config)
    rclpy.init()
    node = Pi05BridgeNode(config)
    try:
        rclpy.spin(node)
    finally:
        if config.bridge.publish_deadman and node.enabled:
            zero = Float64(data=0.0)
            node.left_deadman_pub.publish(zero)
            node.right_deadman_pub.publish(zero)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
