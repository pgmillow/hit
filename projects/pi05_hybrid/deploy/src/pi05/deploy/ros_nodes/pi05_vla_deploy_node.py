"""ROS 2 node that runs Pi0.5 VLA inference and publishes Pi0.5 command topics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float64, String

from pi05.common.data.image_preprocess import ImagePreprocessConfig, preprocess_rgb_image
from pi05.deploy.config import load_deploy_config
from pi05.deploy.config.schema import DeployConfig
from pi05.deploy.models import load_policy_runtime
from pi05.deploy.runtime.control_loop import ControlLoop
from pi05.deploy.runtime.inference_worker import InferenceWorker
from pi05.deploy.runtime.observation_collector import ObservationCollector
from pi05.deploy.runtime.safety_guard import SafetyGuard
from pi05.deploy.runtime.shared_buffer import SharedBuffer


class Pi05VlaDeployNode(Node):
    """Collect observations, run policy inference, and publish /pi05_vla/command topics."""

    def __init__(self, config: DeployConfig) -> None:
        super().__init__("pi05_vla_deploy_node")
        self.config = config
        self.image_config = ImagePreprocessConfig(
            image_size=config.image.image_size,
            mode=config.image.resize_mode,  # type: ignore[arg-type]
        )
        self.collector = ObservationCollector(
            proprioception_order=config.topics.observation.proprioception_order
        )
        self.shared_buffer = SharedBuffer(
            max_inference_requests=config.runtime.max_inference_requests,
            max_pending_chunks=config.runtime.max_pending_chunks,
        )
        self.safety_guard = SafetyGuard(config.safety)
        self.control_loop = ControlLoop(
            shared_buffer=self.shared_buffer,
            request_queue=self.shared_buffer.inference_request_queue,
            result_queue=self.shared_buffer.chunk_result_queue,
            observation_provider=lambda: self.shared_buffer.latest_observation(
                max_age_s=config.safety.stale_observation_timeout_s
            ),
            safety_guard=self.safety_guard,
            control_hz=config.runtime.control_hz,
            execute_horizon=config.runtime.execute_horizon,
            prefetch_steps=config.runtime.prefetch_steps,
            blend_steps=config.runtime.blend_steps,
            action_dim=config.runtime.action_dim,
            max_action_age_s=config.runtime.max_action_age_sec,
            fallback_policy=config.runtime.fallback_policy,
            stale_observation_timeout_s=config.safety.stale_observation_timeout_s,
            log_info=self.get_logger().info,
            log_warning=self.get_logger().warning,
        )
        self._last_missing_log_s = 0.0

        self.get_logger().info(f"loading Pi0.5 policy bundle: {config.bundle.resolved_bundle_dir}")
        policy_runtime = load_policy_runtime(config)
        self.get_logger().info(f"torch.compile predict_action_chunk enabled={policy_runtime.compile_enabled}")
        self.inference_worker = InferenceWorker(
            policy_runtime=policy_runtime,
            request_queue=self.shared_buffer.inference_request_queue,
            result_queue=self.shared_buffer.chunk_result_queue,
            shared_buffer=self.shared_buffer,
            inference_hz=config.runtime.inference_hz,
            control_hz=config.runtime.control_hz,
            log_info=self.get_logger().info,
            log_debug=self.get_logger().debug,
            log_warning=self.get_logger().warning,
        )

        self._create_subscriptions()
        self._create_publishers()
        self.control_timer = self.create_timer(1.0 / config.runtime.control_hz, self._control_tick)
        self.metrics_timer = self.create_timer(1.0 / config.runtime.publish_metrics_hz, self._publish_metrics)
        self.inference_worker.start()
        self.get_logger().warning(
            "Pi0.5 deployment started "
            f"mode={config.runtime.mode} infer_hz={config.runtime.inference_hz:.1f} "
            f"control_hz={config.runtime.control_hz:.1f}"
        )

    def shutdown(self) -> None:
        self.inference_worker.stop()
        self.inference_worker.join(timeout=2.0)

    def _create_subscriptions(self) -> None:
        topics = self.config.topics.observation
        if self.config.image.transport in {"compressed", "both"}:
            self.create_subscription(CompressedImage, topics.top_image, lambda msg: self._image_cb("top", msg), 10)
            self.create_subscription(CompressedImage, topics.left_wrist_image, lambda msg: self._image_cb("left_wrist", msg), 10)
            self.create_subscription(CompressedImage, topics.right_wrist_image, lambda msg: self._image_cb("right_wrist", msg), 10)
        if self.config.image.transport in {"raw", "both"}:
            self.create_subscription(Image, topics.top_image_raw, lambda msg: self._image_cb("top", msg), 10)
            self.create_subscription(Image, topics.left_wrist_image_raw, lambda msg: self._image_cb("left_wrist", msg), 10)
            self.create_subscription(Image, topics.right_wrist_image_raw, lambda msg: self._image_cb("right_wrist", msg), 10)
        self.create_subscription(JointState, topics.proprioception, self._proprio_cb, 10)
        self.create_subscription(JointState, topics.left_hand_state, lambda msg: self._hand_cb("left", msg), 10)
        self.create_subscription(JointState, topics.right_hand_state, lambda msg: self._hand_cb("right", msg), 10)
        self.create_subscription(Point, topics.left_ee_position, lambda msg: self._point_cb("left_ee_pos", msg), 10)
        self.create_subscription(Vector3, topics.left_ee_rpy, lambda msg: self._vec3_cb("left_ee_rpy", msg), 10)
        self.create_subscription(Point, topics.right_ee_position, lambda msg: self._point_cb("right_ee_pos", msg), 10)
        self.create_subscription(Vector3, topics.right_ee_rpy, lambda msg: self._vec3_cb("right_ee_rpy", msg), 10)

    def _create_publishers(self) -> None:
        topics = self.config.topics.command
        self.left_arm_pub = self.create_publisher(JointState, topics.left_arm_joint_target, 10)
        self.right_arm_pub = self.create_publisher(JointState, topics.right_arm_joint_target, 10)
        self.left_hand_pub = self.create_publisher(Float64, topics.left_hand_target, 10)
        self.right_hand_pub = self.create_publisher(Float64, topics.right_hand_target, 10)
        self.status_pub = self.create_publisher(String, topics.status, 10)
        self.metrics_pub = self.create_publisher(String, topics.metrics, 10)

    def _image_cb(self, name: str, msg: CompressedImage | Image) -> None:
        try:
            rgb = _decode_image(msg)
            image = preprocess_rgb_image(rgb, self.image_config)
            self.collector.update_image(name, image)
            self._publish_observation_if_ready()
        except Exception as exc:
            self.get_logger().warning(f"failed to process {name} image: {exc}")

    def _proprio_cb(self, msg: JointState) -> None:
        try:
            self.collector.update_proprioception(list(msg.position))
            self._publish_observation_if_ready()
        except Exception as exc:
            self.get_logger().warning(f"failed to process proprioception: {exc}")

    def _hand_cb(self, side: str, msg: JointState) -> None:
        if not msg.position:
            return
        self.collector.update_hand(side, float(msg.position[0]))
        self._publish_observation_if_ready()

    def _point_cb(self, key: str, msg: Point) -> None:
        self.collector.update_vector(key, [msg.x, msg.y, msg.z])
        self._publish_observation_if_ready()

    def _vec3_cb(self, key: str, msg: Vector3) -> None:
        self.collector.update_vector(key, [msg.x, msg.y, msg.z])
        self._publish_observation_if_ready()

    def _publish_observation_if_ready(self) -> None:
        snapshot = self.collector.snapshot(max_age_s=self.config.safety.stale_observation_timeout_s)
        if snapshot is not None:
            self.shared_buffer.set_observation(snapshot)
            return
        now = time.monotonic()
        if now - self._last_missing_log_s > 2.0:
            missing = ", ".join(self.collector.missing_fields())
            if missing:
                self.get_logger().info(f"waiting for observation fields: {missing}")
            self._last_missing_log_s = now

    def _control_tick(self) -> None:
        command = self.control_loop.tick()
        if command is None:
            return
        if not self.config.runtime.publishes_command_topics:
            self.get_logger().info(
                "dry-run command "
                f"left={np.round(command.action.left_arm, 3).tolist()} "
                f"right={np.round(command.action.right_arm, 3).tolist()} "
                f"hands=({command.action.left_hand:.1f}, {command.action.right_hand:.1f})"
            )
            return
        self.left_arm_pub.publish(_joint_msg(command.action.left_arm))
        self.right_arm_pub.publish(_joint_msg(command.action.right_arm))
        self.left_hand_pub.publish(Float64(data=float(command.action.left_hand)))
        self.right_hand_pub.publish(Float64(data=float(command.action.right_hand)))

    def _publish_metrics(self) -> None:
        payload = self.shared_buffer.metrics_snapshot()
        payload["mode"] = self.config.runtime.mode
        payload.update(self.control_loop.status_snapshot())
        self.metrics_pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        self.status_pub.publish(String(data=f"mode={self.config.runtime.mode} metrics={payload}"))


def _decode_image(msg: CompressedImage | Image) -> np.ndarray:
    import cv2

    if isinstance(msg, CompressedImage):
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        image = array.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if msg.encoding == "bgr8" else image
    if msg.encoding in ("rgba8", "bgra8"):
        image = array.reshape(msg.height, msg.width, 4)
        code = cv2.COLOR_BGRA2RGB if msg.encoding == "bgra8" else cv2.COLOR_RGBA2RGB
        return cv2.cvtColor(image, code)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def _joint_msg(joints_rad: np.ndarray) -> JointState:
    from pi05.common.robot.action_spec import ARM_JOINT_NAMES

    msg = JointState()
    msg.name = list(ARM_JOINT_NAMES)
    msg.position = [float(value) for value in joints_rad[: len(ARM_JOINT_NAMES)]]
    return msg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Pi0.5 VLA deployment node.")
    parser.add_argument("--config", type=Path, required=True, help="Path to deploy/config/deploy.yaml.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_deploy_config(args.config)
    rclpy.init()
    node = Pi05VlaDeployNode(config)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
