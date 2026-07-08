"""ROS topic naming helpers for the Pi0.5 deployment stack."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_NAMESPACE = "/pi05_vla"


def join_topic(namespace: str, *parts: str) -> str:
    """Join ROS topic fragments while keeping a single leading slash."""
    cleaned = [str(namespace).strip("/")]
    cleaned.extend(str(part).strip("/") for part in parts if str(part).strip("/"))
    return "/" + "/".join(part for part in cleaned if part)


@dataclass(frozen=True)
class Pi05CommandTopics:
    """Topic set produced by the Pi0.5 deployment command loop."""

    left_arm_joint_target: str
    right_arm_joint_target: str
    left_hand_target: str
    right_hand_target: str
    status: str
    metrics: str

    @classmethod
    def with_namespace(cls, namespace: str = DEFAULT_NAMESPACE) -> "Pi05CommandTopics":
        return cls(
            left_arm_joint_target=join_topic(namespace, "command", "left_arm", "joint_target"),
            right_arm_joint_target=join_topic(namespace, "command", "right_arm", "joint_target"),
            left_hand_target=join_topic(namespace, "command", "left_hand", "target"),
            right_hand_target=join_topic(namespace, "command", "right_hand", "target"),
            status=join_topic(namespace, "status"),
            metrics=join_topic(namespace, "metrics"),
        )


@dataclass(frozen=True)
class Pi05ObservationTopics:
    """Default topic set consumed by the Pi0.5 observation collector."""

    top_image: str
    left_wrist_image: str
    right_wrist_image: str
    proprioception: str
    left_hand_state: str
    right_hand_state: str
    left_ee_position: str
    left_ee_rpy: str
    right_ee_position: str
    right_ee_rpy: str

    @classmethod
    def with_namespace(cls, namespace: str = DEFAULT_NAMESPACE) -> "Pi05ObservationTopics":
        return cls(
            top_image=join_topic(namespace, "observation", "image", "top", "compressed"),
            left_wrist_image=join_topic(namespace, "observation", "image", "left_wrist", "compressed"),
            right_wrist_image=join_topic(namespace, "observation", "image", "right_wrist", "compressed"),
            proprioception=join_topic(namespace, "observation", "proprioception"),
            left_hand_state=join_topic(namespace, "observation", "left_hand", "joint_state"),
            right_hand_state=join_topic(namespace, "observation", "right_hand", "joint_state"),
            left_ee_position=join_topic(namespace, "observation", "left_arm", "ee_position"),
            left_ee_rpy=join_topic(namespace, "observation", "left_arm", "ee_rpy"),
            right_ee_position=join_topic(namespace, "observation", "right_arm", "ee_position"),
            right_ee_rpy=join_topic(namespace, "observation", "right_arm", "ee_rpy"),
        )
