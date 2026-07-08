"""ROS 2 command multiplexer for teleop and Pi0.5 VLA control.

The mux is intentionally placed above the existing picotele execution nodes.
It arbitrates command ownership only; low-level smoothing, RM CANFD output,
and hand register writes remain inside picotele.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

from pi05.deploy.config import load_deploy_config
from pi05.deploy.config.schema import DeployConfig


TELEOP_MODE = "teleop"
VLA_MODE = "vla"


class CommandMuxNode(Node):
    """Forward either teleop or VLA candidate commands to picotele."""

    def __init__(self, config: DeployConfig) -> None:
        super().__init__("pi05_command_mux_node")
        self.config = config
        self.mux = config.mux
        self.topics = config.topics.mux
        self.enabled = bool(self.mux.enabled)
        self.mode = self.mux.default_mode if self.enabled else TELEOP_MODE
        self.vla_requested = self.mode == VLA_MODE
        self.last_reason = "startup"
        self._last_status_publish_s = 0.0
        self._last_vla_arm_s = {"left": 0.0, "right": 0.0}
        self._last_teleop_deadman = {"left": 0.0, "right": 0.0}

        self.left_arm_pub = self.create_publisher(JointState, self.topics.output_left_arm_joint_target, 10)
        self.right_arm_pub = self.create_publisher(JointState, self.topics.output_right_arm_joint_target, 10)
        self.left_hand_pub = self.create_publisher(Float64, self.topics.output_left_hand_trigger, 10)
        self.right_hand_pub = self.create_publisher(Float64, self.topics.output_right_hand_trigger, 10)
        self.left_deadman_pub = self.create_publisher(Float64, self.topics.output_left_deadman, 10)
        self.right_deadman_pub = self.create_publisher(Float64, self.topics.output_right_deadman, 10)
        self.status_pub = self.create_publisher(String, self.topics.status, 10)

        self._create_subscriptions()
        self.timer = self.create_timer(1.0 / self.mux.publish_hz, self._tick)
        state = "enabled" if self.enabled else "disabled"
        self.get_logger().warning(
            f"command mux {state}: mode={self.mode} "
            f"teleop_arm=({self.topics.teleop_left_arm_joint_target}, {self.topics.teleop_right_arm_joint_target}) "
            f"vla_arm=({self.topics.vla_left_arm_joint_target}, {self.topics.vla_right_arm_joint_target}) "
            f"out_arm=({self.topics.output_left_arm_joint_target}, {self.topics.output_right_arm_joint_target})"
        )

    def _create_subscriptions(self) -> None:
        self.create_subscription(
            JointState,
            self.topics.teleop_left_arm_joint_target,
            lambda msg: self._arm_cb("teleop", "left", msg),
            10,
        )
        self.create_subscription(
            JointState,
            self.topics.teleop_right_arm_joint_target,
            lambda msg: self._arm_cb("teleop", "right", msg),
            10,
        )
        self.create_subscription(
            JointState,
            self.topics.vla_left_arm_joint_target,
            lambda msg: self._arm_cb("vla", "left", msg),
            10,
        )
        self.create_subscription(
            JointState,
            self.topics.vla_right_arm_joint_target,
            lambda msg: self._arm_cb("vla", "right", msg),
            10,
        )
        self.create_subscription(
            Float64,
            self.topics.teleop_left_hand_trigger,
            lambda msg: self._hand_cb("teleop", "left", msg),
            10,
        )
        self.create_subscription(
            Float64,
            self.topics.teleop_right_hand_trigger,
            lambda msg: self._hand_cb("teleop", "right", msg),
            10,
        )
        self.create_subscription(
            Float64,
            self.topics.vla_left_hand_trigger,
            lambda msg: self._hand_cb("vla", "left", msg),
            10,
        )
        self.create_subscription(
            Float64,
            self.topics.vla_right_hand_trigger,
            lambda msg: self._hand_cb("vla", "right", msg),
            10,
        )
        self.create_subscription(
            Float64,
            self.topics.teleop_left_deadman,
            lambda msg: self._teleop_deadman_cb("left", msg),
            10,
        )
        self.create_subscription(
            Float64,
            self.topics.teleop_right_deadman,
            lambda msg: self._teleop_deadman_cb("right", msg),
            10,
        )
        self.create_subscription(Bool, self.topics.vla_enable, self._vla_enable_cb, 10)

    def _arm_cb(self, source: str, side: str, msg: JointState) -> None:
        if len(msg.position) < 6:
            self.get_logger().warning(f"ignoring {source} {side} arm target with <6 joints")
            return
        now = time.monotonic()
        if source == VLA_MODE:
            self._last_vla_arm_s[side] = now
            self._maybe_enter_vla("fresh VLA arm target")
        if not self.enabled:
            return
        if self.mode == source:
            self._arm_publisher(side).publish(msg)

    def _hand_cb(self, source: str, side: str, msg: Float64) -> None:
        if not self.enabled:
            return
        if self.mode == source:
            self._hand_publisher(side).publish(msg)

    def _teleop_deadman_cb(self, side: str, msg: Float64) -> None:
        value = float(msg.data)
        self._last_teleop_deadman[side] = value
        if (
            self.enabled
            and self.mode == VLA_MODE
            and value >= float(self.mux.manual_takeover_deadman_threshold)
        ):
            self.vla_requested = False
            self._switch_mode(TELEOP_MODE, f"{side} deadman manual takeover")
        if self.enabled and self.mode == TELEOP_MODE:
            self._deadman_publisher(side).publish(msg)

    def _vla_enable_cb(self, msg: Bool) -> None:
        if not self.enabled:
            return
        self.vla_requested = bool(msg.data)
        if self.vla_requested:
            self._maybe_enter_vla("VLA enable requested")
        else:
            self._switch_mode(TELEOP_MODE, "VLA disabled")

    def _tick(self) -> None:
        if self.enabled and self.mode == VLA_MODE:
            if not self.vla_requested:
                self._switch_mode(TELEOP_MODE, "VLA no longer requested")
            elif not self._vla_arm_targets_fresh(time.monotonic()):
                self.vla_requested = False
                self._switch_mode(TELEOP_MODE, "VLA arm target timeout")
            else:
                self._publish_vla_deadman()
        self._publish_status_if_due()

    def _maybe_enter_vla(self, reason: str) -> None:
        if not self.enabled or not self.vla_requested:
            return
        if self._vla_arm_targets_fresh(time.monotonic()):
            self._switch_mode(VLA_MODE, reason)

    def _vla_arm_targets_fresh(self, now: float) -> bool:
        timeout_s = float(self.mux.vla_command_timeout_s)
        return all(
            stamp > 0.0 and now - stamp <= timeout_s
            for stamp in self._last_vla_arm_s.values()
        )

    def _switch_mode(self, mode: str, reason: str) -> None:
        if mode == self.mode:
            self.last_reason = reason
            return
        previous = self.mode
        self.mode = mode
        self.last_reason = reason
        self.get_logger().warning(f"command mux mode switch: {previous} -> {mode}; reason={reason}")
        if mode == VLA_MODE:
            self._publish_vla_deadman()
            return
        self._publish_teleop_deadman_or_stop()

    def _publish_vla_deadman(self) -> None:
        value = Float64(data=float(self.mux.vla_deadman_value))
        self.left_deadman_pub.publish(value)
        self.right_deadman_pub.publish(value)

    def _publish_teleop_deadman_or_stop(self) -> None:
        self.left_deadman_pub.publish(Float64(data=float(self._last_teleop_deadman["left"])))
        self.right_deadman_pub.publish(Float64(data=float(self._last_teleop_deadman["right"])))

    def publish_safe_stop(self) -> None:
        zero = Float64(data=0.0)
        self.left_deadman_pub.publish(zero)
        self.right_deadman_pub.publish(zero)

    def _publish_status_if_due(self) -> None:
        now = time.monotonic()
        interval_s = 1.0 / max(1.0e-6, float(self.mux.status_publish_hz))
        if now - self._last_status_publish_s < interval_s:
            return
        self._last_status_publish_s = now
        ages = {
            side: (None if stamp <= 0.0 else now - stamp)
            for side, stamp in self._last_vla_arm_s.items()
        }
        payload = {
            "enabled": self.enabled,
            "mode": self.mode,
            "vla_requested": self.vla_requested,
            "reason": self.last_reason,
            "vla_arm_age_s": ages,
        }
        self.status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _arm_publisher(self, side: str):
        return self.left_arm_pub if side == "left" else self.right_arm_pub

    def _hand_publisher(self, side: str):
        return self.left_hand_pub if side == "left" else self.right_hand_pub

    def _deadman_publisher(self, side: str):
        return self.left_deadman_pub if side == "left" else self.right_deadman_pub


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pi0.5 teleop/VLA command mux.")
    parser.add_argument("--config", type=Path, required=True, help="Path to deploy/config/deploy.yaml.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_deploy_config(args.config)
    rclpy.init()
    node = CommandMuxNode(config)
    try:
        rclpy.spin(node)
    finally:
        node.publish_safe_stop()
        time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
