#!/usr/bin/env python3
"""
将 Octopus 录制的 .mcap（ROS2）转换为 LeRobot v3 数据集格式，
用于 LeRobot pi0.5（pi05）训练。

输出字段：
- RGB 图像（HWC, uint8，始终启用）：
    observation.images.top
    observation.images.left_wrist
    observation.images.right_wrist
- VTLA 触觉伪图像（HWC, uint8，通过 --include-tactile 启用）：
    observation.images.left_tactile  （可选，VTLA）
    observation.images.right_tactile （可选，VTLA）
- observation.state: float32 (26,) = [left_arm_qpos6, right_arm_qpos6, left_hand_qpos1, right_hand_qpos1,
                                      left_ee_position3, left_ee_rpy3, right_ee_position3, right_ee_rpy3]
- action: float32 (14,) = [left_arm_cmd_pos6, right_arm_cmd_pos6, left_hand_cmd_pos1, right_hand_cmd_pos1]

视觉 schema 固定为离线预处理后的 224x224x3 图像。
触觉数据可转换为两张 224x224x3 伪图像，每只手一张。

触觉转换：
- 输入 topic：
    /inspire/{left,right}_hand/tactile_{12,13,22,23,32,33,42,43,52,54,61}
- 输入编码：
    sensor_msgs/Image mono16.
- 输出特征：
    observation.images.left_tactile
    observation.images.right_tactile
- 拼接布局：
    第一行：12 | 22 | 32 | 42 | 52
    第二行：13 | 23 | 33 | 43 | 54
    底部行：61 横向拉伸铺满整行。
- 通道含义：
    R = 扣除 baseline 后的接触强度
    G = 正向时间差分
    B = 负向时间差分
- baseline：
    从第一帧 top 相机 anchor 时间开始，取前 --tactile-baseline-seconds 秒触觉值的中位数。

时间同步行为：
- 以 top 相机时间戳作为 episode 时间轴 anchor。
- RGB 腕部图像和触觉图像都使用最近邻采样，并使用严格 50 ms gate。
  如果任一必需图像/触觉流距离 anchor 超过 50 ms，则丢弃整个 frame。
- --numeric-interp-mode 同时影响数值流（机械臂、手、末端位姿）和触觉图像的采样策略。
  strict 模式：触觉图像使用 50 ms 最近邻 gate，超出则丢帧。
  continuous 模式：触觉图像使用无门限最近邻采样，不因时间戳偏差丢帧。
  注意：腕部 RGB 相机始终使用 50 ms gate，不受此参数影响。

重要 schema 规则：
    在 RGB-only 转换和 --include-tactile 转换之间切换时，请使用新的 --out 目录。
    脚本会拒绝把 VTLA 五路图像数据 append 到已有 RGB-only 数据集，也会拒绝反向混用。

示例，RGB-only 单文件：
    python mcap_to_lerobot_v3.py \
      --mcap /path/to/demo.mcap \
      --out  /path/to/out_dataset \
      --repo-id local/octopus_pi05 \
      --task "bimanual manipulation with dexterous hand" \
      --fps 60

示例，VTLA 单文件：
    python mcap_to_lerobot_v3.py \
      --mcap /path/to/demo.mcap \
      --out /path/to/out_dataset_vtla \
      --repo-id local/octopus_pi05_vtla \
      --task "bimanual manipulation with tactile feedback" \
      --fps 60 \
      --numeric-interp-mode continuous \
      --include-tactile

示例，VTLA 目录模式：
    python mcap_to_lerobot_v3.py \
      --mcap-dir /path/to/mcap_dir \
      --out /path/to/out_dataset_vtla \
      --repo-id local/octopus_pi05_vtla \
      --task "bimanual manipulation with dexterous hand" \
      --fps 60 \
      --numeric-interp-mode continuous \
      --include-tactile

可选触觉尺度控制：
    默认每个 episode 从 99.5 分位数自动估计 pressure/delta scale。
    如果希望训练/部署可视化使用固定尺度，可传入：
      --tactile-pressure-scale <value>
      --tactile-delta-scale <value>

Append 行为：
    如果 --out 已经是有效 LeRobot 数据集，脚本会继续追加写入而不是覆盖；
    但前提是已有数据集的 image feature schema 与本次请求的转换模式完全一致。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = WORKSPACE_ROOT / "third_party" / "lerobot" / "src"
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

cv2 = None
np = None
Image = None
read_ros2_messages = None
LeRobotDataset = None
DEFAULT_FEATURES = None


def _ensure_runtime_dependencies() -> None:
    global cv2, np, Image, read_ros2_messages, LeRobotDataset, DEFAULT_FEATURES
    if cv2 is not None:
        return

    import cv2 as cv2_module
    import numpy as np_module
    from PIL import Image as image_module
    from mcap_ros2.reader import read_ros2_messages as read_ros2_messages_func
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as lerobot_dataset_cls
    from lerobot.datasets.utils import DEFAULT_FEATURES as default_features

    cv2 = cv2_module
    np = np_module
    Image = image_module
    read_ros2_messages = read_ros2_messages_func
    LeRobotDataset = lerobot_dataset_cls
    DEFAULT_FEATURES = default_features


# -----------------------------
# ROS topics (your recording config, compressed image topics)
# -----------------------------
TOPIC_RIGHT_IMG = "/realsense/right_hand/color/image_rect_raw/compressed"
TOPIC_LEFT_IMG = "/realsense/left_hand/color/image_rect_raw/compressed"
TOPIC_TOP_IMG = "/realsense/top/color/image_raw/compressed"

TOPIC_VLA_ARM_STATE = "/vla_teleop/proprioception"
TOPIC_LEFT_EE_POSITION = "/left_arm/ee_position"
TOPIC_LEFT_EE_RPY = "/left_arm/ee_rpy"
TOPIC_RIGHT_EE_POSITION = "/right_arm/ee_position"
TOPIC_RIGHT_EE_RPY = "/right_arm/ee_rpy"

TOPIC_LEFT_HAND_JOINT = "/inspire/left_hand/joint_states"
TOPIC_RIGHT_HAND_JOINT = "/inspire/right_hand/joint_states"

TOPIC_VLA_ARM_CMD = "/vla_teleop/action_label"

TOPIC_RIGHT_HAND_CMD = "/inspire/right_hand/joint_cmd"
TOPIC_LEFT_HAND_CMD = "/inspire/left_hand/joint_cmd"

# 50 ms hard gate: if any critical stream is farther than this from the TOP-camera anchor,
# drop the whole frame instead of manufacturing an observation from stale data.
EXTRAPOLATION_TOLERANCE_S = 0.050
ARM_DOF = 6
HAND_DOF = 1
EE_POSE_DOF = 6
# 当前灵巧手仅映射 1 个关节（如中指）来控制整体张合；
# 未来升级四代手或 6 DoF 独立控制时，只需将此值改为 6。
# 动作空间仍然锁定为 14 维：左右臂各 6 维关节位控 + 左右手各 1 维张合。
ACTION_DIM = 2 * ARM_DOF + 2 * HAND_DOF
# 末端位姿由 /{left,right}_arm/ee_position 的 xyz 与 /{left,right}_arm/ee_rpy 的 rpy 拼接。
# action 维度不随 observation.state 扩展变化。
STATE_DIM = ACTION_DIM + 2 * EE_POSE_DOF
EXPECTED_IMAGE_SHAPE = (224, 224, 3)
OBS_TOP_KEY = "observation.images.top"
OBS_LEFT_WRIST_KEY = "observation.images.left_wrist"
OBS_RIGHT_WRIST_KEY = "observation.images.right_wrist"
OBS_LEFT_TACTILE_KEY = "observation.images.left_tactile"
OBS_RIGHT_TACTILE_KEY = "observation.images.right_tactile"

TACTILE_IDS = ("12", "13", "22", "23", "32", "33", "42", "43", "52", "54", "61")
TACTILE_LAYOUT_VERSION = "inspire_hand_v1"
TACTILE_TOPIC_BY_SIDE = {
    "left": {taxel_id: f"/inspire/left_hand/tactile_{taxel_id}" for taxel_id in TACTILE_IDS},
    "right": {taxel_id: f"/inspire/right_hand/tactile_{taxel_id}" for taxel_id in TACTILE_IDS},
}
TACTILE_TOPIC_TO_SIDE_ID = {
    topic: (side, taxel_id)
    for side, topics in TACTILE_TOPIC_BY_SIDE.items()
    for taxel_id, topic in topics.items()
}
TACTILE_LAYOUT = {
    "12": (0, 80, 0, 44),
    "13": (80, 160, 0, 44),
    "22": (0, 80, 44, 89),
    "23": (80, 160, 44, 89),
    "32": (0, 80, 89, 134),
    "33": (80, 160, 89, 134),
    "42": (0, 80, 134, 179),
    "43": (80, 160, 134, 179),
    "52": (0, 80, 179, 224),
    "54": (80, 160, 179, 224),
    "61": (160, 224, 0, 224),
}
TACTILE_BASELINE_SECONDS = 1.0
TACTILE_SCALE_PERCENTILE = 99.5


# -----------------------------
# Helpers: ROS Image decode
# -----------------------------
def ros_image_to_numpy_rgb(msg) -> np.ndarray:
    _ensure_runtime_dependencies()
    """
    Convert sensor_msgs/Image or sensor_msgs/CompressedImage to HxWx3 uint8 RGB numpy array.
    Supports: rgb8, bgr8, rgba8, bgra8, mono8.
    """
    enc = getattr(msg, "encoding", None)
    if enc:
        enc = enc.lower()
        h, w = int(msg.height), int(msg.width)

        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            if enc == "bgr8":
                arr = arr[..., ::-1]  # BGR -> RGB
            return arr

        if enc in ("rgba8", "bgra8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
            if enc == "bgra8":
                arr = arr[..., [2, 1, 0, 3]]  # BGRA -> RGBA
            return arr[..., :3]  # drop alpha -> RGB

        if enc == "mono8":
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 1)
            return np.repeat(arr, 3, axis=2)

        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    data = bytes(msg.data)
    if not data:
        raise ValueError("Compressed image data is empty")
    try:
        img = Image.open(BytesIO(data))
        img = img.convert("RGB")
    except Exception as exc:
        fmt = getattr(msg, "format", None)
        raise ValueError(f"Failed to decode compressed image: format={fmt}") from exc
    return np.asarray(img, dtype=np.uint8)


def preprocess_to_vla_shape(img: np.ndarray) -> np.ndarray:
    _ensure_runtime_dependencies()
    """
    将原始 RGB 图像按等比例缩放后填充到 224x224x3。
    不做中心裁剪，尽量保留完整视野。
    """
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected raw RGB image shape (H, W, 3), got {img.shape}")

    target_h, target_w, _ = EXPECTED_IMAGE_SHAPE
    src_h, src_w = img.shape[:2]

    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid image size: {(src_h, src_w)}")

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized
    return canvas

# -----------------------------
# Build features
# -----------------------------
def build_features_pi05(*, include_tactile: bool = False) -> dict:
    _ensure_runtime_dependencies()
    """
    Use DEFAULT_FEATURES as base, then expose a position-only schema:
    - observation.state: [left_arm_qpos6, right_arm_qpos6, left_hand_qpos(HAND_DOF), right_hand_qpos(HAND_DOF),
                          left_ee_position3, left_ee_rpy3, right_ee_position3, right_ee_rpy3]
    - action:            [left_arm_cmd6, right_arm_cmd6, left_hand_cmd(HAND_DOF), right_hand_cmd(HAND_DOF)]
    - images:            fixed preprocessed HWC 224x224x3
    """
    h, w, c = EXPECTED_IMAGE_SHAPE

    features = dict(DEFAULT_FEATURES)

    # Images: HWC
    img_spec = {
        "dtype": "image",
        "shape": (h, w, c),
        "names": ["height", "width", "channel"],
    }
    features[OBS_TOP_KEY] = dict(img_spec)
    features[OBS_LEFT_WRIST_KEY] = dict(img_spec)
    features[OBS_RIGHT_WRIST_KEY] = dict(img_spec)
    if include_tactile:
        features[OBS_LEFT_TACTILE_KEY] = dict(img_spec)
        features[OBS_RIGHT_TACTILE_KEY] = dict(img_spec)

    state_names = (
        [f"left_arm_qpos{i}" for i in range(ARM_DOF)]
        + [f"right_arm_qpos{i}" for i in range(ARM_DOF)]
        + [f"left_hand_qpos{i}" for i in range(HAND_DOF)]
        + [f"right_hand_qpos{i}" for i in range(HAND_DOF)]
        + _ee_pose_names("left")
        + _ee_pose_names("right")
    )
    action_names = (
        [f"left_arm_cmd_pos{i}" for i in range(ARM_DOF)]
        + [f"right_arm_cmd_pos{i}" for i in range(ARM_DOF)]
        + [f"left_hand_cmd_pos{i}" for i in range(HAND_DOF)]
        + [f"right_hand_cmd_pos{i}" for i in range(HAND_DOF)]
    )

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": state_names,
    }
    features["action"] = {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": action_names,
    }

    return features


def _ee_pose_names(side: str) -> list[str]:
    if EE_POSE_DOF == 7:
        suffixes = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    elif EE_POSE_DOF == 6:
        suffixes = ["x", "y", "z", "roll", "pitch", "yaw"]
    else:
        suffixes = [f"dim{i}" for i in range(EE_POSE_DOF)]
    return [f"{side}_ee_pose_{suffix}" for suffix in suffixes]


def open_or_resume_dataset(
    out_root: Path,
    repo_id: str,
    fps: int,
    features: dict,
    robot_type: str,
) -> LeRobotDataset:
    """Create a new dataset or resume appending to an existing one.

    If ``out_root`` already looks like a LeRobot dataset root, resume writing so
    new episodes are appended to the existing metadata and parquet/video/image
    indices. Otherwise create a new dataset from scratch.
    """
    info_path = out_root / "meta" / "info.json"
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            existing_info = json.load(f)
        existing_fps = int(existing_info.get("fps", -1))
        existing_features = existing_info.get("features", {})
        existing_state_shape = existing_features.get("observation.state", {}).get("shape")
        existing_action_shape = existing_features.get("action", {}).get("shape")
        if existing_fps != int(fps):
            raise RuntimeError(
                f"Existing dataset fps={existing_fps} does not match requested fps={fps}. "
                "请为 60Hz 新 schema 使用新的输出目录，避免把不同时间基准的数据混在一起。"
            )
        if list(existing_state_shape or []) != [STATE_DIM] or list(existing_action_shape or []) != [ACTION_DIM]:
            raise RuntimeError(
                "Existing dataset feature schema does not match the current converter "
                f"(state={existing_state_shape}, action={existing_action_shape}; "
                f"expected state={[STATE_DIM]}, action={[ACTION_DIM]}). "
                "请重新生成数据集，避免 14D state 与新 EE-pose state 混用。"
            )
        expected_image_keys = sorted(
            key
            for key, spec in features.items()
            if key.startswith("observation.images.") and spec.get("dtype") == "image"
        )
        existing_image_keys = sorted(
            key
            for key, spec in existing_features.items()
            if key.startswith("observation.images.") and spec.get("dtype") == "image"
        )
        if existing_image_keys != expected_image_keys:
            raise RuntimeError(
                "Existing dataset image schema does not match the requested converter schema "
                f"(existing={existing_image_keys}, expected={expected_image_keys}). "
                "请为 tactile/VTLA 数据使用新的输出目录，避免把三路 RGB 和五路 VTLA schema 混用。"
            )
        print(f"[INFO] Existing dataset detected at {out_root}; resuming append mode.")
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=out_root,
            image_writer_processes=0,
            image_writer_threads=0,
            streaming_encoding=False,
            vcodec="libsvtav1",
        )

    if out_root.exists():
        remaining_entries = [p for p in out_root.iterdir()]
        # Allow a manually pre-created empty dataset root, but refuse to create
        # inside a non-empty directory that lacks LeRobot metadata.
        visible_entries = [p for p in remaining_entries if p.name not in {".DS_Store"}]
        if visible_entries:
            has_nested_content = any(p.is_file() for p in visible_entries) or any(
                any(child for child in p.iterdir()) for p in visible_entries if p.is_dir()
            )
            if has_nested_content:
                raise RuntimeError(
                    f"Output directory {out_root} exists but is not a valid LeRobot dataset root. "
                    "Delete old contents or keep meta/info.json for resume mode."
                )
        for child in sorted(visible_entries, reverse=True):
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        out_root.rmdir()

    print(f"[INFO] Creating new dataset at {out_root}.")
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        robot_type=robot_type,
        use_videos=False,
    )


# -----------------------------
# Episode-level camera-anchor alignment
# -----------------------------
def _pack_position_state(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.shape != (STATE_DIM,):
        raise ValueError(f"Expected state shape {(STATE_DIM,)}, got {qpos.shape}")
    return qpos


def _pack_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Expected action shape {(ACTION_DIM,)}, got {action.shape}")
    return action


def _pad_vec(vec: np.ndarray, target_dim: int) -> np.ndarray:
    if vec.shape[0] >= target_dim:
        return vec[:target_dim].astype(np.float32, copy=False)
    return np.pad(vec, (0, target_dim - vec.shape[0]), constant_values=0.0).astype(np.float32)


@dataclass
class FrozenImageFrame:
    data: bytes
    encoding: Optional[str] = None
    height: int = 0
    width: int = 0
    step: int = 0
    is_bigendian: int = 0
    format: Optional[str] = None


def _freeze_image_msg(msg) -> FrozenImageFrame:
    return FrozenImageFrame(
        data=bytes(msg.data),
        encoding=getattr(msg, "encoding", None),
        height=int(getattr(msg, "height", 0) or 0),
        width=int(getattr(msg, "width", 0) or 0),
        step=int(getattr(msg, "step", 0) or 0),
        is_bigendian=int(getattr(msg, "is_bigendian", 0) or 0),
        format=getattr(msg, "format", None),
    )


@dataclass
class NumericSeriesBuffer:
    name: str
    timestamps: list[float] = field(default_factory=list)
    values: list[np.ndarray] = field(default_factory=list)


@dataclass
class ImageSeriesBuffer:
    name: str
    timestamps: list[float] = field(default_factory=list)
    frames: list[FrozenImageFrame] = field(default_factory=list)


def _make_tactile_buffers(side: str) -> dict[str, ImageSeriesBuffer]:
    return {
        taxel_id: ImageSeriesBuffer(f"{side}_tactile_{taxel_id}")
        for taxel_id in TACTILE_IDS
    }


@dataclass
class EpisodeCapture:
    top_img: ImageSeriesBuffer = field(default_factory=lambda: ImageSeriesBuffer("top_img"))
    left_img: ImageSeriesBuffer = field(default_factory=lambda: ImageSeriesBuffer("left_img"))
    right_img: ImageSeriesBuffer = field(default_factory=lambda: ImageSeriesBuffer("right_img"))
    left_tactile: dict[str, ImageSeriesBuffer] = field(default_factory=lambda: _make_tactile_buffers("left"))
    right_tactile: dict[str, ImageSeriesBuffer] = field(default_factory=lambda: _make_tactile_buffers("right"))
    arm_q: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("arm_q"))
    arm_cmd: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("arm_cmd"))
    left_ee_position: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("left_ee_position"))
    left_ee_rpy: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("left_ee_rpy"))
    right_ee_position: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("right_ee_position"))
    right_ee_rpy: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("right_ee_rpy"))
    left_hand_q: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("left_hand_q"))
    right_hand_q: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("right_hand_q"))
    left_hand_cmd: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("left_hand_cmd"))
    right_hand_cmd: NumericSeriesBuffer = field(default_factory=lambda: NumericSeriesBuffer("right_hand_cmd"))
    last_vla_action_stamp: Optional[float] = None
    last_vla_proprio_stamp: Optional[float] = None
    vla_skew_warn_count: int = 0
    max_vla_skew_s: float = 0.0

    def check_vla_stamp_alignment(self, tol_s: float = 0.050) -> None:
        if self.last_vla_action_stamp is None or self.last_vla_proprio_stamp is None:
            return
        skew = abs(self.last_vla_action_stamp - self.last_vla_proprio_stamp)
        if skew > tol_s:
            self.vla_skew_warn_count += 1
            self.max_vla_skew_s = max(self.max_vla_skew_s, skew)
            if self.vla_skew_warn_count <= 5:
                print(
                    "[WARN] VLA action/proprio timestamp skew too large: "
                    f"{skew:.6f}s > {tol_s:.6f}s; keeping mcap and relying on per-frame gates"
                )


def _stamp_to_sec(msg) -> Optional[float]:
    stamp = getattr(getattr(msg, "header", None), "stamp", None)
    if stamp is None:
        return None
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        if stamp.sec == 0 and stamp.nanosec == 0:
            return None
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "nanoseconds"):
        if stamp.nanoseconds == 0:
            return None
        return float(stamp.nanoseconds) * 1e-9
    return None


def _require_header_stamp(msg, topic: str) -> float:
    t = _stamp_to_sec(msg)
    if t is None:
        raise RuntimeError(
            f"{topic} header.stamp is zero; converter would fall back to log_time "
            "and break cmd/state alignment."
        )
    return t


def _extract_hand_position(msg) -> Optional[np.ndarray]:
    position = getattr(msg, "position", None)
    if not position:
        return None
    return _pad_vec(np.asarray(list(position), dtype=np.float32), HAND_DOF)


def _extract_vector3(msg, topic: str) -> np.ndarray:
    if all(hasattr(msg, attr) for attr in ("x", "y", "z")):
        return np.asarray([float(msg.x), float(msg.y), float(msg.z)], dtype=np.float32)
    data = getattr(msg, "data", None)
    if data is not None:
        vec = np.asarray(list(data), dtype=np.float32)
        if vec.shape[0] >= 3:
            return vec[:3]
    position = getattr(msg, "position", None)
    if position is not None:
        vec = np.asarray(list(position), dtype=np.float32)
        if vec.shape[0] >= 3:
            return vec[:3]
    raise RuntimeError(f"{topic} must be geometry_msgs/Point, geometry_msgs/Vector3, or a 3D numeric array.")


def _extract_ee_pose(msg, topic: str) -> np.ndarray:
    raw_pose = _extract_pose_like_vector(msg)
    if raw_pose is None:
        raise RuntimeError(
            f"{topic} does not look like Pose/PoseStamped/TransformStamped or a numeric array. "
            "请确认末端位姿话题类型，并在 _extract_pose_like_vector() 中补充解析逻辑。"
        )
    if raw_pose.shape[0] == 7 and EE_POSE_DOF == 6:
        raw_pose = np.concatenate([raw_pose[:3], _quat_xyzw_to_rpy(raw_pose[3:7])], axis=0)
    if raw_pose.shape[0] != EE_POSE_DOF:
        raise RuntimeError(
            f"{topic} pose dimension is {raw_pose.shape[0]}, expected {EE_POSE_DOF}. "
            "TODO(硬件确认): 若上游为 6D euler 位姿，请把 EE_POSE_DOF 改为 6；"
            "若为 7D quaternion 位姿，请发布 [x,y,z,qx,qy,qz,qw]。"
        )
    return _normalize_ee_pose(raw_pose.astype(np.float32, copy=False))


def _extract_pose_like_vector(msg) -> Optional[np.ndarray]:
    # 兼容 ROS 常见 PoseStamped / Pose / TransformStamped / Odometry，以及 Float*MultiArray。
    pose = getattr(msg, "pose", None)
    if pose is not None and hasattr(pose, "pose"):
        pose = pose.pose
    transform = getattr(msg, "transform", None)
    if transform is not None:
        return _pose_components_to_vec(
            position=getattr(transform, "translation", None),
            orientation=getattr(transform, "rotation", None),
        )
    if pose is not None:
        return _pose_components_to_vec(
            position=getattr(pose, "position", None),
            orientation=getattr(pose, "orientation", None),
        )
    if hasattr(msg, "position") and hasattr(msg, "orientation"):
        return _pose_components_to_vec(
            position=getattr(msg, "position", None),
            orientation=getattr(msg, "orientation", None),
        )
    data = getattr(msg, "data", None)
    if data is not None:
        return np.asarray(list(data), dtype=np.float32)
    position = getattr(msg, "position", None)
    if position is not None:
        return np.asarray(list(position), dtype=np.float32)
    return None


def _pose_components_to_vec(position, orientation) -> Optional[np.ndarray]:
    if position is None:
        return None
    xyz = [
        float(getattr(position, "x")),
        float(getattr(position, "y")),
        float(getattr(position, "z")),
    ]
    if orientation is None:
        return np.asarray(xyz, dtype=np.float32)
    quat = [
        float(getattr(orientation, "x")),
        float(getattr(orientation, "y")),
        float(getattr(orientation, "z")),
        float(getattr(orientation, "w")),
    ]
    return np.asarray(xyz + quat, dtype=np.float32)


def _quat_xyzw_to_rpy(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-6:
        raise RuntimeError("EE pose quaternion norm is zero; cannot convert orientation to euler.")
    x, y, z, w = [float(v) for v in (quat / norm)]
    # TODO(硬件确认): 仅当你把 EE_POSE_DOF 改成 6 时才会使用该分支；
    # 默认 7D quaternion 会原样进入 state，避免欧拉角奇异性。
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.asarray([roll, pitch, yaw], dtype=np.float32)


def _normalize_ee_pose(pose: np.ndarray) -> np.ndarray:
    if EE_POSE_DOF != 7:
        return pose
    quat_norm = float(np.linalg.norm(pose[3:7]))
    if quat_norm <= 1e-6:
        raise RuntimeError("EE pose quaternion norm is zero; cannot normalize orientation.")
    pose = pose.copy()
    # 线性插值四元数后做单位化，避免把非单位四元数作为 state 喂给模型。
    pose[3:7] = pose[3:7] / quat_norm
    return pose


def _sanitize_numeric_series(
    series: NumericSeriesBuffer,
) -> tuple[np.ndarray, np.ndarray, int]:
    kept_timestamps: list[float] = []
    kept_values: list[np.ndarray] = []
    dropped = 0

    for t, value in zip(series.timestamps, series.values):
        if not np.isfinite(t):
            dropped += 1
            continue
        if kept_timestamps and t <= kept_timestamps[-1]:
            dropped += 1
            continue
        kept_timestamps.append(float(t))
        kept_values.append(np.asarray(value, dtype=np.float32))

    if not kept_timestamps:
        return np.empty((0,), dtype=np.float64), np.empty((0, 0), dtype=np.float32), dropped

    return (
        np.asarray(kept_timestamps, dtype=np.float64),
        np.stack(kept_values, axis=0).astype(np.float32),
        dropped,
    )


def _sanitize_image_series(
    series: ImageSeriesBuffer,
) -> tuple[np.ndarray, list[FrozenImageFrame], int]:
    kept_timestamps: list[float] = []
    kept_frames: list[FrozenImageFrame] = []
    dropped = 0

    for t, frame in zip(series.timestamps, series.frames):
        if not np.isfinite(t):
            dropped += 1
            continue
        if kept_timestamps and t <= kept_timestamps[-1]:
            dropped += 1
            continue
        kept_timestamps.append(float(t))
        kept_frames.append(frame)

    return np.asarray(kept_timestamps, dtype=np.float64), kept_frames, dropped


def _sanitize_tactile_series(
    buffers: dict[str, ImageSeriesBuffer],
) -> tuple[dict[str, tuple[np.ndarray, list[FrozenImageFrame]]], int]:
    sanitized: dict[str, tuple[np.ndarray, list[FrozenImageFrame]]] = {}
    dropped_total = 0
    for taxel_id, series in buffers.items():
        timestamps, frames, dropped = _sanitize_image_series(series)
        sanitized[taxel_id] = (timestamps, frames)
        dropped_total += dropped
    return sanitized, dropped_total


@dataclass
class TactilePreprocessStats:
    baseline_by_side_id: dict[tuple[str, str], np.ndarray]
    pressure_scale: float
    delta_scale: float


@dataclass
class TactileRenderState:
    previous_pressure: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def reset(self) -> None:
        self.previous_pressure.clear()


def tactile_side_key(side: str) -> str:
    if side == "left":
        return OBS_LEFT_TACTILE_KEY
    if side == "right":
        return OBS_RIGHT_TACTILE_KEY
    raise ValueError(f"Unsupported tactile side: {side}")


def _decode_tactile_frame(frame: FrozenImageFrame) -> np.ndarray:
    """Decode mono tactile Image frames to float32 HxW arrays."""
    _ensure_runtime_dependencies()
    encoding = (frame.encoding or "").lower()
    if frame.height <= 0 or frame.width <= 0:
        raise ValueError(f"Invalid tactile image shape: {(frame.height, frame.width)}")

    if encoding in {"mono16", "16uc1"}:
        dtype = np.dtype(np.uint16).newbyteorder(">" if frame.is_bigendian else "<")
        bytes_per_pixel = 2
    elif encoding == "mono8":
        dtype = np.dtype(np.uint8)
        bytes_per_pixel = 1
    else:
        raise ValueError(f"Unsupported tactile image encoding: {frame.encoding}")

    expected_row_bytes = frame.width * bytes_per_pixel
    step = frame.step or expected_row_bytes
    data = bytes(frame.data)
    if len(data) < step * frame.height:
        raise ValueError(
            f"Tactile frame data is too short: got {len(data)} bytes, "
            f"expected at least {step * frame.height}"
        )

    if step == expected_row_bytes:
        arr = np.frombuffer(data[: expected_row_bytes * frame.height], dtype=dtype)
        arr = arr.reshape(frame.height, frame.width)
    else:
        rows = []
        for row_idx in range(frame.height):
            start = row_idx * step
            end = start + expected_row_bytes
            rows.append(np.frombuffer(data[start:end], dtype=dtype))
        arr = np.stack(rows, axis=0)
    return arr.astype(np.float32, copy=False)


def _decode_cached_tactile_frame(
    frames: list[FrozenImageFrame],
    index: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if index not in cache:
        cache[index] = _decode_tactile_frame(frames[index])
    return cache[index]


def _build_tactile_preprocess_stats(
    tactile_by_side: dict[str, dict[str, tuple[np.ndarray, list[FrozenImageFrame]]]],
    *,
    anchor_start_t: float,
    baseline_seconds: float,
    pressure_scale_override: float | None,
    delta_scale_override: float | None,
) -> TactilePreprocessStats:
    baseline_by_side_id: dict[tuple[str, str], np.ndarray] = {}
    pressure_samples: list[np.ndarray] = []
    delta_samples: list[np.ndarray] = []

    for side, tactile_series in tactile_by_side.items():
        for taxel_id, (timestamps, frames) in tactile_series.items():
            if timestamps.size == 0 or not frames:
                raise RuntimeError(f"{side}_tactile_{taxel_id}: no samples")

            baseline_indices = np.where(timestamps <= anchor_start_t + float(baseline_seconds))[0]
            if baseline_indices.size == 0:
                baseline_indices = np.arange(min(10, len(frames)))
            baseline_stack = np.stack(
                [_decode_tactile_frame(frames[int(idx)]) for idx in baseline_indices],
                axis=0,
            )
            baseline = np.median(baseline_stack, axis=0).astype(np.float32)
            baseline_by_side_id[(side, taxel_id)] = baseline

            previous_pressure = None
            for frame in frames:
                pressure = np.abs(_decode_tactile_frame(frame) - baseline).astype(np.float32, copy=False)
                pressure_samples.append(pressure.reshape(-1))
                if previous_pressure is not None and previous_pressure.shape == pressure.shape:
                    delta_samples.append(np.abs(pressure - previous_pressure).reshape(-1))
                previous_pressure = pressure

    pressure_scale = _resolve_tactile_scale(
        pressure_samples,
        override=pressure_scale_override,
        name="pressure_scale",
    )
    delta_scale = _resolve_tactile_scale(
        delta_samples,
        override=delta_scale_override,
        name="delta_scale",
    )
    return TactilePreprocessStats(
        baseline_by_side_id=baseline_by_side_id,
        pressure_scale=pressure_scale,
        delta_scale=delta_scale,
    )


def _resolve_tactile_scale(
    samples: list[np.ndarray],
    *,
    override: float | None,
    name: str,
) -> float:
    if override is not None and override > 0.0:
        return float(override)
    if not samples:
        return 1.0
    values = np.concatenate(samples, axis=0).astype(np.float32, copy=False)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    scale = float(np.percentile(values, TACTILE_SCALE_PERCENTILE))
    if scale <= 1e-6:
        print(f"[WARN] tactile {name} percentile is near zero; using 1.0")
        return 1.0
    return scale


def _sample_tactile_indices(
    tactile_series: dict[str, tuple[np.ndarray, list[FrozenImageFrame]]],
    target_t: float,
    *,
    side: str,
    sample_func=None,
) -> tuple[dict[str, int], list[str]]:
    if sample_func is None:
        sample_func = _sample_nearest_index_with_gate
    indices: dict[str, int] = {}
    errors: list[str] = []
    for taxel_id in TACTILE_IDS:
        timestamps, _frames = tactile_series[taxel_id]
        idx, err = sample_func(
            timestamps,
            target_t,
            name=f"{side}_tactile_{taxel_id}",
        )
        if err is not None:
            errors.append(err)
            continue
        assert idx is not None
        indices[taxel_id] = idx
    return indices, errors


def _sample_nearest_index_no_gate(
    source_timestamps: np.ndarray,
    target_t: float,
    *,
    name: str,
) -> tuple[Optional[int], Optional[str]]:
    if source_timestamps.size == 0:
        return None, f"{name}: no samples"
    return _nearest_index(source_timestamps, target_t), None


def _render_tactile_hand_image(
    *,
    side: str,
    indices: dict[str, int],
    tactile_series: dict[str, tuple[np.ndarray, list[FrozenImageFrame]]],
    decode_caches: dict[str, dict[int, np.ndarray]],
    stats: TactilePreprocessStats,
    render_state: TactileRenderState,
) -> np.ndarray:
    _ensure_runtime_dependencies()
    canvas = np.zeros(EXPECTED_IMAGE_SHAPE, dtype=np.uint8)
    for taxel_id in TACTILE_IDS:
        _timestamps, frames = tactile_series[taxel_id]
        raw = _decode_cached_tactile_frame(
            frames,
            indices[taxel_id],
            decode_caches.setdefault(taxel_id, {}),
        )
        baseline = stats.baseline_by_side_id[(side, taxel_id)]
        if raw.shape != baseline.shape:
            raise RuntimeError(
                f"{side}_tactile_{taxel_id}: raw shape {raw.shape} does not match baseline {baseline.shape}"
            )
        pressure = np.abs(raw - baseline).astype(np.float32, copy=False)
        prev_key = (side, taxel_id)
        previous_pressure = render_state.previous_pressure.get(prev_key)
        if previous_pressure is None or previous_pressure.shape != pressure.shape:
            delta = np.zeros_like(pressure)
        else:
            delta = pressure - previous_pressure
        render_state.previous_pressure[prev_key] = pressure.copy()

        tile = _tactile_pressure_delta_to_rgb(
            pressure=pressure,
            delta=delta,
            pressure_scale=stats.pressure_scale,
            delta_scale=stats.delta_scale,
        )
        y0, y1, x0, x1 = TACTILE_LAYOUT[taxel_id]
        canvas[y0:y1, x0:x1, :] = cv2.resize(
            tile,
            (x1 - x0, y1 - y0),
            interpolation=cv2.INTER_NEAREST,
        )
    return canvas


def _tactile_pressure_delta_to_rgb(
    *,
    pressure: np.ndarray,
    delta: np.ndarray,
    pressure_scale: float,
    delta_scale: float,
) -> np.ndarray:
    red = np.clip(pressure / max(float(pressure_scale), 1e-6), 0.0, 1.0)
    green = np.clip(np.maximum(delta, 0.0) / max(float(delta_scale), 1e-6), 0.0, 1.0)
    blue = np.clip(np.maximum(-delta, 0.0) / max(float(delta_scale), 1e-6), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return np.round(rgb * 255.0).astype(np.uint8)


def _nearest_index(source_timestamps: np.ndarray, target_t: float) -> int:
    if source_timestamps.size == 0:
        raise RuntimeError("Cannot sample from an empty stream")

    right_idx = int(np.searchsorted(source_timestamps, target_t, side="left"))
    if right_idx >= source_timestamps.shape[0]:
        return source_timestamps.shape[0] - 1
    if right_idx == 0:
        return 0

    left_idx = right_idx - 1
    left_dist = abs(target_t - float(source_timestamps[left_idx]))
    right_dist = abs(float(source_timestamps[right_idx]) - target_t)
    return right_idx if right_dist < left_dist else left_idx


def _sample_nearest_index_with_gate(
    source_timestamps: np.ndarray,
    target_t: float,
    *,
    name: str,
) -> tuple[Optional[int], Optional[str]]:
    if source_timestamps.size == 0:
        return None, f"{name}: no samples"

    idx = _nearest_index(source_timestamps, target_t)
    delta_s = abs(float(source_timestamps[idx]) - target_t)
    if delta_s > EXTRAPOLATION_TOLERANCE_S:
        return None, (
            f"{name}: nearest sample is {delta_s * 1000.0:.1f} ms away "
            f"(gate {EXTRAPOLATION_TOLERANCE_S * 1000.0:.0f} ms)"
        )
    return idx, None


def _sample_linear_numeric_at(
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
    target_t: float,
    *,
    name: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if source_timestamps.size == 0:
        return None, f"{name}: no samples"

    if source_timestamps.shape[0] != source_values.shape[0]:
        return None, (
            f"{name}: timestamp/value count mismatch "
            f"({source_timestamps.shape[0]} vs {source_values.shape[0]})"
        )

    nearest_idx, gate_error = _sample_nearest_index_with_gate(source_timestamps, target_t, name=name)
    if gate_error is not None:
        return None, gate_error
    assert nearest_idx is not None

    right_idx = int(np.searchsorted(source_timestamps, target_t, side="left"))
    if right_idx < source_timestamps.shape[0] and float(source_timestamps[right_idx]) == target_t:
        return np.asarray(source_values[right_idx], dtype=np.float32), None

    left_idx = right_idx - 1
    if left_idx < 0 or right_idx >= source_timestamps.shape[0]:
        return None, f"{name}: missing bracketing samples around anchor {target_t:.6f}s"

    left_t = float(source_timestamps[left_idx])
    right_t = float(source_timestamps[right_idx])
    prev_age_s = target_t - left_t
    next_age_s = right_t - target_t
    if prev_age_s > EXTRAPOLATION_TOLERANCE_S or next_age_s > EXTRAPOLATION_TOLERANCE_S:
        return None, (
            f"{name}: bracketing samples are too far from anchor "
            f"(prev {prev_age_s * 1000.0:.1f} ms, next {next_age_s * 1000.0:.1f} ms)"
        )

    denom = right_t - left_t
    if denom <= 0.0:
        return None, f"{name}: non-monotonic bracketing timestamps around {target_t:.6f}s"

    alpha = (target_t - left_t) / denom
    left_value = np.asarray(source_values[left_idx], dtype=np.float32)
    right_value = np.asarray(source_values[right_idx], dtype=np.float32)
    value = (1.0 - alpha) * left_value + alpha * right_value
    return np.asarray(value, dtype=np.float32), None


def _sample_continuous_linear_numeric_at(
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
    target_t: float,
    *,
    name: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Linearly resample numeric streams at every anchor timestamp.

    Unlike _sample_linear_numeric_at(), this mode does not enforce the 50 ms
    bracketing gate. It interpolates across recording gaps and holds the nearest
    endpoint outside the stream range, so every top-camera anchor can receive a
    numeric state/action value when the stream is non-empty.
    """
    if source_timestamps.size == 0:
        return None, f"{name}: no samples"

    if source_timestamps.shape[0] != source_values.shape[0]:
        return None, (
            f"{name}: timestamp/value count mismatch "
            f"({source_timestamps.shape[0]} vs {source_values.shape[0]})"
        )

    right_idx = int(np.searchsorted(source_timestamps, target_t, side="left"))
    if right_idx <= 0:
        return np.asarray(source_values[0], dtype=np.float32), None
    if right_idx >= source_timestamps.shape[0]:
        return np.asarray(source_values[-1], dtype=np.float32), None
    if float(source_timestamps[right_idx]) == target_t:
        return np.asarray(source_values[right_idx], dtype=np.float32), None

    left_idx = right_idx - 1
    left_t = float(source_timestamps[left_idx])
    right_t = float(source_timestamps[right_idx])
    denom = right_t - left_t
    if denom <= 0.0:
        return None, f"{name}: non-monotonic bracketing timestamps around {target_t:.6f}s"

    alpha = (target_t - left_t) / denom
    left_value = np.asarray(source_values[left_idx], dtype=np.float32)
    right_value = np.asarray(source_values[right_idx], dtype=np.float32)
    value = (1.0 - alpha) * left_value + alpha * right_value
    return np.asarray(value, dtype=np.float32), None


def _sample_previous_numeric_at(
    source_timestamps: np.ndarray,
    source_values: np.ndarray,
    target_t: float,
    *,
    name: str,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    if source_timestamps.size == 0:
        return None, f"{name}: no samples"

    if source_timestamps.shape[0] != source_values.shape[0]:
        return None, (
            f"{name}: timestamp/value count mismatch "
            f"({source_timestamps.shape[0]} vs {source_values.shape[0]})"
        )

    prev_idx = int(np.searchsorted(source_timestamps, target_t, side="right")) - 1
    if prev_idx < 0:
        return None, f"{name}: no sample at or before anchor {target_t:.6f}s"

    age_s = target_t - float(source_timestamps[prev_idx])
    if age_s > EXTRAPOLATION_TOLERANCE_S:
        return None, (
            f"{name}: previous sample is {age_s * 1000.0:.1f} ms old "
            f"(gate {EXTRAPOLATION_TOLERANCE_S * 1000.0:.0f} ms)"
        )

    return np.asarray(source_values[prev_idx], dtype=np.float32), None


def _require_expected_image_shape(img: np.ndarray, topic: str) -> np.ndarray:
    if tuple(img.shape) != EXPECTED_IMAGE_SHAPE:
        raise RuntimeError(
            f"{topic} decoded image shape {tuple(img.shape)} != expected {EXPECTED_IMAGE_SHAPE}. "
            "Visual schema is fixed to offline-preprocessed 224x224x3."
        )
    return img


def _decode_cached_image(
    frames: list[FrozenImageFrame],
    index: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if index not in cache:
        cache[index] = ros_image_to_numpy_rgb(frames[index])
    return cache[index]


def _preprocess_cached_image(
    frames: list[FrozenImageFrame],
    index: int,
    decode_cache: dict[int, np.ndarray],
    preprocess_cache: dict[int, np.ndarray],
) -> np.ndarray:
    if index not in preprocess_cache:
        raw = _decode_cached_image(frames, index, decode_cache)
        preprocess_cache[index] = preprocess_to_vla_shape(raw)
    return preprocess_cache[index]


def _median_anchor_fps(target_timestamps: np.ndarray) -> Optional[float]:
    if target_timestamps.shape[0] < 2:
        return None
    dt = np.diff(target_timestamps)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def _log_dt_stats(
    mcap_path: Path,
    chunk_index: int,
    frame_count: int,
    deltas_s: list[float],
    fps: int,
) -> None:
    if frame_count <= 0:
        return
    if not deltas_s:
        print(f"[DT] {mcap_path.name} chunk {chunk_index}: frames={frame_count}, dt_count=0 (single-frame chunk)")
        return

    # dt = current_target_t - previous_target_t，表示相邻两帧真实时间间隔。。
    nominal_ms = 1000.0 / float(fps)
    dt_ms = np.asarray(deltas_s, dtype=np.float64) * 1000.0
    print(
        f"[DT] {mcap_path.name} chunk {chunk_index}: "
        f"frames={frame_count}, dt_count={dt_ms.size}, "
        f"mean={dt_ms.mean():.2f}ms, median={np.median(dt_ms):.2f}ms, "
        f"min={dt_ms.min():.2f}ms, p95={np.percentile(dt_ms, 95):.2f}ms, "
        f"p99={np.percentile(dt_ms, 99):.2f}ms, max={dt_ms.max():.2f}ms, "
        f">1.25x={int(np.sum(dt_ms > nominal_ms * 1.25))}, "
        f">2x={int(np.sum(dt_ms > nominal_ms * 2.0))}, "
        f">3x={int(np.sum(dt_ms > nominal_ms * 3.0))}"
    )


def _in_time_window(
    log_t: float,
    time_window: Optional[Tuple[float, float]],
    time_offset_s: float,
) -> bool:
    if time_window is None:
        return True
    start_s, end_s = time_window
    t_rel = log_t - time_offset_s
    return start_s <= t_rel <= end_s


def _convert_one_mcap_into_dataset(
    ds: LeRobotDataset,
    mcap_path: Path,
    task: str,
    fps: int,
    image_size: int,
    time_window: Optional[Tuple[float, float]] = None,
    time_offset_s: float = 0.0,
    numeric_interp_mode: str = "strict",
    include_tactile: bool = False,
    tactile_baseline_seconds: float = TACTILE_BASELINE_SECONDS,
    tactile_pressure_scale: float | None = None,
    tactile_delta_scale: float | None = None,
) -> tuple[int, int, int]:
    _ = image_size  # deprecated: visual schema is fixed to EXPECTED_IMAGE_SHAPE
    if numeric_interp_mode not in {"strict", "continuous"}:
        raise ValueError(f"Unsupported numeric_interp_mode={numeric_interp_mode!r}")
    topics = [
        TOPIC_RIGHT_IMG,
        TOPIC_LEFT_IMG,
        TOPIC_TOP_IMG,
        TOPIC_VLA_ARM_STATE,
        TOPIC_LEFT_EE_POSITION,
        TOPIC_LEFT_EE_RPY,
        TOPIC_RIGHT_EE_POSITION,
        TOPIC_RIGHT_EE_RPY,
        TOPIC_LEFT_HAND_JOINT,
        TOPIC_RIGHT_HAND_JOINT,
        TOPIC_VLA_ARM_CMD,
        TOPIC_RIGHT_HAND_CMD,
        TOPIC_LEFT_HAND_CMD,
    ]
    if include_tactile:
        for side in ("left", "right"):
            topics.extend(TACTILE_TOPIC_BY_SIDE[side].values())

    captured = EpisodeCapture()
    any_msg = False

    for m in read_ros2_messages(str(mcap_path), topics=topics):
        any_msg = True
        log_t = m.log_time_ns / 1e9
        topic = m.channel.topic
        msg = m.ros_msg
        in_window = _in_time_window(log_t, time_window, time_offset_s)

        if topic == TOPIC_TOP_IMG:
            if not in_window:
                continue
            t = _stamp_to_sec(msg)
            if t is None:
                continue
            captured.top_img.timestamps.append(t)
            captured.top_img.frames.append(_freeze_image_msg(msg))
            continue

        if topic in (TOPIC_LEFT_IMG, TOPIC_RIGHT_IMG):
            if time_window is not None:
                start_s, end_s = time_window
                t_rel = log_t - time_offset_s
                if (t_rel < (start_s - EXTRAPOLATION_TOLERANCE_S)) or (
                    t_rel > (end_s + EXTRAPOLATION_TOLERANCE_S)
                ):
                    continue
            t = _stamp_to_sec(msg)
            if t is None:
                continue
            stream = captured.left_img if topic == TOPIC_LEFT_IMG else captured.right_img
            stream.timestamps.append(t)
            stream.frames.append(_freeze_image_msg(msg))
            continue

        if include_tactile and topic in TACTILE_TOPIC_TO_SIDE_ID:
            if time_window is not None:
                start_s, end_s = time_window
                t_rel = log_t - time_offset_s
                if (t_rel < (start_s - EXTRAPOLATION_TOLERANCE_S)) or (
                    t_rel > (end_s + EXTRAPOLATION_TOLERANCE_S)
                ):
                    continue
            t = _stamp_to_sec(msg)
            if t is None:
                continue
            side, taxel_id = TACTILE_TOPIC_TO_SIDE_ID[topic]
            tactile_buffers = captured.left_tactile if side == "left" else captured.right_tactile
            tactile_buffers[taxel_id].timestamps.append(t)
            tactile_buffers[taxel_id].frames.append(_freeze_image_msg(msg))
            continue

        if topic == TOPIC_VLA_ARM_STATE:
            t = _require_header_stamp(msg, topic)
            pos = _pad_vec(np.asarray(list(msg.position), dtype=np.float32), 12)
            captured.arm_q.timestamps.append(t)
            captured.arm_q.values.append(pos)
            captured.last_vla_proprio_stamp = t
            captured.check_vla_stamp_alignment()
            continue

        if topic == TOPIC_VLA_ARM_CMD:
            t = _require_header_stamp(msg, topic)
            cmd = _pad_vec(np.asarray(list(msg.position), dtype=np.float32), 12)
            captured.arm_cmd.timestamps.append(t)
            captured.arm_cmd.values.append(cmd)
            captured.last_vla_action_stamp = t
            captured.check_vla_stamp_alignment()
            continue

        if topic in (TOPIC_LEFT_EE_POSITION, TOPIC_LEFT_EE_RPY, TOPIC_RIGHT_EE_POSITION, TOPIC_RIGHT_EE_RPY):
            t = _stamp_to_sec(msg)
            if t is None:
                t = log_t
            value = _extract_vector3(msg, topic)
            if topic == TOPIC_LEFT_EE_POSITION:
                stream = captured.left_ee_position
            elif topic == TOPIC_LEFT_EE_RPY:
                stream = captured.left_ee_rpy
            elif topic == TOPIC_RIGHT_EE_POSITION:
                stream = captured.right_ee_position
            else:
                stream = captured.right_ee_rpy
            stream.timestamps.append(t)
            stream.values.append(value)
            continue

        t = _require_header_stamp(msg, topic)

        hand_pos = _extract_hand_position(msg)

        if topic == TOPIC_LEFT_HAND_JOINT and hand_pos is not None:
            captured.left_hand_q.timestamps.append(t)
            captured.left_hand_q.values.append(hand_pos)
        elif topic == TOPIC_RIGHT_HAND_JOINT and hand_pos is not None:
            captured.right_hand_q.timestamps.append(t)
            captured.right_hand_q.values.append(hand_pos)
        elif topic == TOPIC_LEFT_HAND_CMD and hand_pos is not None:
            captured.left_hand_cmd.timestamps.append(t)
            captured.left_hand_cmd.values.append(hand_pos)
        elif topic == TOPIC_RIGHT_HAND_CMD and hand_pos is not None:
            captured.right_hand_cmd.timestamps.append(t)
            captured.right_hand_cmd.values.append(hand_pos)

    if not any_msg:
        raise RuntimeError(f"No ROS2 messages found in {mcap_path}")

    if captured.vla_skew_warn_count > 0:
        print(
            f"[WARN] {mcap_path.name}: observed {captured.vla_skew_warn_count} "
            "VLA action/proprio skew samples above "
            f"{EXTRAPOLATION_TOLERANCE_S * 1000:.0f} ms "
            f"(max {captured.max_vla_skew_s * 1000:.1f} ms). "
            "No whole-file skip; invalid anchor frames will be dropped by per-frame gates."
        )

    top_ts, top_frames, top_drop = _sanitize_image_series(captured.top_img)
    left_ts, left_frames, left_drop = _sanitize_image_series(captured.left_img)
    right_ts, right_frames, right_drop = _sanitize_image_series(captured.right_img)
    left_tactile_series: dict[str, tuple[np.ndarray, list[FrozenImageFrame]]] = {}
    right_tactile_series: dict[str, tuple[np.ndarray, list[FrozenImageFrame]]] = {}
    left_tactile_drop = 0
    right_tactile_drop = 0
    if include_tactile:
        left_tactile_series, left_tactile_drop = _sanitize_tactile_series(captured.left_tactile)
        right_tactile_series, right_tactile_drop = _sanitize_tactile_series(captured.right_tactile)

    arm_q_ts, arm_q, arm_q_drop = _sanitize_numeric_series(captured.arm_q)
    arm_cmd_ts, arm_cmd, arm_cmd_drop = _sanitize_numeric_series(captured.arm_cmd)
    left_ee_position_ts, left_ee_position, left_ee_position_drop = _sanitize_numeric_series(
        captured.left_ee_position
    )
    left_ee_rpy_ts, left_ee_rpy, left_ee_rpy_drop = _sanitize_numeric_series(captured.left_ee_rpy)
    right_ee_position_ts, right_ee_position, right_ee_position_drop = _sanitize_numeric_series(
        captured.right_ee_position
    )
    right_ee_rpy_ts, right_ee_rpy, right_ee_rpy_drop = _sanitize_numeric_series(captured.right_ee_rpy)
    left_hand_q_ts, left_hand_q, left_hand_q_drop = _sanitize_numeric_series(captured.left_hand_q)
    right_hand_q_ts, right_hand_q, right_hand_q_drop = _sanitize_numeric_series(captured.right_hand_q)
    left_hand_cmd_ts, left_hand_cmd, left_hand_cmd_drop = _sanitize_numeric_series(captured.left_hand_cmd)
    right_hand_cmd_ts, right_hand_cmd, right_hand_cmd_drop = _sanitize_numeric_series(
        captured.right_hand_cmd
    )

    dropped_non_monotonic = (
        top_drop
        + left_drop
        + right_drop
        + arm_q_drop
        + arm_cmd_drop
        + left_ee_position_drop
        + left_ee_rpy_drop
        + right_ee_position_drop
        + right_ee_rpy_drop
        + left_hand_q_drop
        + right_hand_q_drop
        + left_hand_cmd_drop
        + right_hand_cmd_drop
        + left_tactile_drop
        + right_tactile_drop
    )

    if top_ts.size == 0:
        raise RuntimeError(f"No valid anchor frames found in {mcap_path} on {TOPIC_TOP_IMG}")
    if left_ts.size == 0 or right_ts.size == 0:
        raise RuntimeError(f"Missing wrist camera frames in {mcap_path}")

    anchor_timestamps = top_ts
    observed_fps = _median_anchor_fps(anchor_timestamps)
    if observed_fps is not None and abs(observed_fps - float(fps)) > 1.0:
        raise RuntimeError(
            f"Dataset metadata fps={fps} but anchor camera median fps is {observed_fps:.2f} "
            f"for {mcap_path.name}. 请使用与相机 anchor 一致的 --fps。"
        )

    tactile_by_side: dict[str, dict[str, tuple[np.ndarray, list[FrozenImageFrame]]]] = {}
    tactile_stats: TactilePreprocessStats | None = None
    if include_tactile:
        tactile_by_side = {
            "left": left_tactile_series,
            "right": right_tactile_series,
        }
        tactile_stats = _build_tactile_preprocess_stats(
            tactile_by_side,
            anchor_start_t=float(anchor_timestamps[0]),
            baseline_seconds=tactile_baseline_seconds,
            pressure_scale_override=tactile_pressure_scale,
            delta_scale_override=tactile_delta_scale,
        )
        print(
            f"[TACTILE] {mcap_path.name}: layout={TACTILE_LAYOUT_VERSION} "
            f"pressure_scale={tactile_stats.pressure_scale:.3f} "
            f"delta_scale={tactile_stats.delta_scale:.3f} "
            f"baseline_seconds={tactile_baseline_seconds:.2f}"
        )

    ds.clear_episode_buffer(delete_images=True)

    left_decode_cache: dict[int, np.ndarray] = {}
    right_decode_cache: dict[int, np.ndarray] = {}
    left_preprocess_cache: dict[int, np.ndarray] = {}
    right_preprocess_cache: dict[int, np.ndarray] = {}
    left_tactile_decode_caches: dict[str, dict[int, np.ndarray]] = {}
    right_tactile_decode_caches: dict[str, dict[int, np.ndarray]] = {}
    tactile_render_state = TactileRenderState()
    frames_added = 0
    dropped_unreliable = 0
    last_valid_target_t = None
    split_count = 0
    current_chunk_frames = 0
    current_chunk_deltas_s: list[float] = []
    sample_numeric = (
        _sample_continuous_linear_numeric_at
        if numeric_interp_mode == "continuous"
        else _sample_linear_numeric_at
    )
    sample_tactile = (
        _sample_nearest_index_no_gate
        if numeric_interp_mode == "continuous"
        else _sample_nearest_index_with_gate
    )

    for i, target_t in enumerate(anchor_timestamps):
        # TOP camera hardware timestamps are the only valid episode timeline.
        # Every other modality must be sampled relative to this absolute anchor.
        drop_reasons: list[str] = []

        left_idx, left_err = _sample_nearest_index_with_gate(left_ts, target_t, name="left_wrist_image")
        if left_err is not None:
            drop_reasons.append(left_err)

        right_idx, right_err = _sample_nearest_index_with_gate(
            right_ts, target_t, name="right_wrist_image"
        )
        if right_err is not None:
            drop_reasons.append(right_err)

        left_tactile_indices: dict[str, int] = {}
        right_tactile_indices: dict[str, int] = {}
        if include_tactile:
            left_tactile_indices, left_tactile_errors = _sample_tactile_indices(
                left_tactile_series,
                target_t,
                side="left",
                sample_func=sample_tactile,
            )
            right_tactile_indices, right_tactile_errors = _sample_tactile_indices(
                right_tactile_series,
                target_t,
                side="right",
                sample_func=sample_tactile,
            )
            drop_reasons.extend(left_tactile_errors)
            drop_reasons.extend(right_tactile_errors)

        # Arms are sampled at the camera anchor. Strict mode enforces the 50 ms
        # bracketing gate; continuous mode linearly interpolates across gaps.
        arm_q_value, arm_q_err = sample_numeric(arm_q_ts, arm_q, target_t, name="arm_q")
        if arm_q_err is not None:
            drop_reasons.append(arm_q_err)

        arm_cmd_value, arm_cmd_err = sample_numeric(
            arm_cmd_ts, arm_cmd, target_t, name="arm_cmd"
        )
        if arm_cmd_err is not None:
            drop_reasons.append(arm_cmd_err)

        # 末端位姿由 position(xyz) 与 rpy 两条 60Hz 派生流分别插值后拼成 6D。
        left_ee_position_value, left_ee_position_err = sample_numeric(
            left_ee_position_ts, left_ee_position, target_t, name="left_ee_position"
        )
        if left_ee_position_err is not None:
            drop_reasons.append(left_ee_position_err)

        left_ee_rpy_value, left_ee_rpy_err = sample_numeric(
            left_ee_rpy_ts, left_ee_rpy, target_t, name="left_ee_rpy"
        )
        if left_ee_rpy_err is not None:
            drop_reasons.append(left_ee_rpy_err)

        right_ee_position_value, right_ee_position_err = sample_numeric(
            right_ee_position_ts, right_ee_position, target_t, name="right_ee_position"
        )
        if right_ee_position_err is not None:
            drop_reasons.append(right_ee_position_err)

        right_ee_rpy_value, right_ee_rpy_err = sample_numeric(
            right_ee_rpy_ts, right_ee_rpy, target_t, name="right_ee_rpy"
        )
        if right_ee_rpy_err is not None:
            drop_reasons.append(right_ee_rpy_err)

        left_ee_pose_value = None
        if left_ee_position_value is not None and left_ee_rpy_value is not None:
            left_ee_pose_value = np.concatenate([left_ee_position_value[:3], left_ee_rpy_value[:3]], axis=0)

        right_ee_pose_value = None
        if right_ee_position_value is not None and right_ee_rpy_value is not None:
            right_ee_pose_value = np.concatenate([right_ee_position_value[:3], right_ee_rpy_value[:3]], axis=0)

        # Hands follow the same numeric interpolation mode as arms/EE.
        left_hand_q_value, left_hand_q_err = sample_numeric(
            left_hand_q_ts, left_hand_q, target_t, name="left_hand_q"
        )
        if left_hand_q_err is not None:
            drop_reasons.append(left_hand_q_err)

        right_hand_q_value, right_hand_q_err = sample_numeric(
            right_hand_q_ts, right_hand_q, target_t, name="right_hand_q"
        )
        if right_hand_q_err is not None:
            drop_reasons.append(right_hand_q_err)

        left_hand_cmd_value, left_hand_cmd_err = sample_numeric(
            left_hand_cmd_ts, left_hand_cmd, target_t, name="left_hand_cmd"
        )
        if left_hand_cmd_err is not None:
            drop_reasons.append(left_hand_cmd_err)

        right_hand_cmd_value, right_hand_cmd_err = sample_numeric(
            right_hand_cmd_ts, right_hand_cmd, target_t, name="right_hand_cmd"
        )
        if right_hand_cmd_err is not None:
            drop_reasons.append(right_hand_cmd_err)

        if not drop_reasons:
            for name, value in (
                ("arm_q", arm_q_value),
                ("arm_cmd", arm_cmd_value),
                ("left_ee_pose", left_ee_pose_value),
                ("right_ee_pose", right_ee_pose_value),
                ("left_hand_q", left_hand_q_value),
                ("right_hand_q", right_hand_q_value),
                ("left_hand_cmd", left_hand_cmd_value),
                ("right_hand_cmd", right_hand_cmd_value),
            ):
                if value is None or not np.all(np.isfinite(value)):
                    drop_reasons.append(f"{name}: non-finite sampled value")

        if drop_reasons:
            dropped_unreliable += 1
            print(
                f"[WARN] Drop anchor frame at t={target_t:.6f}s in {mcap_path.name}: "
                + "; ".join(drop_reasons)
            )
            continue

        assert left_idx is not None and right_idx is not None
        assert arm_q_value is not None and arm_cmd_value is not None
        assert left_ee_pose_value is not None and right_ee_pose_value is not None
        assert left_hand_q_value is not None and right_hand_q_value is not None
        assert left_hand_cmd_value is not None and right_hand_cmd_value is not None

        if last_valid_target_t is not None:
            time_gap = target_t - last_valid_target_t
            split_gap_s = max(0.050, 1.5 / float(fps))
            if time_gap > split_gap_s:
                print(
                    f"[INFO] Time gap {time_gap * 1000:.1f}ms > {split_gap_s * 1000:.1f}ms detected at "
                    f"t={target_t:.3f}s. Splitting episode to prevent pseudo-{fps}Hz."
                )
                _log_dt_stats(
                    mcap_path,
                    split_count + 1,
                    current_chunk_frames,
                    current_chunk_deltas_s,
                    fps,
                )
                ds.save_episode()
                ds.clear_episode_buffer(delete_images=False)
                tactile_render_state.reset()
                split_count += 1
                current_chunk_frames = 0
                current_chunk_deltas_s = []
            else:
                current_chunk_deltas_s.append(time_gap)

        last_valid_target_t = target_t

        state = np.concatenate(
            [
                arm_q_value[6:12],
                arm_q_value[:6],
                left_hand_q_value[:HAND_DOF],
                right_hand_q_value[:HAND_DOF],
                left_ee_pose_value[:EE_POSE_DOF],
                right_ee_pose_value[:EE_POSE_DOF],
            ],
            axis=0,
        ).astype(np.float32)
        action14 = np.concatenate(
            [
                arm_cmd_value[6:12],
                arm_cmd_value[:6],
                left_hand_cmd_value[:HAND_DOF],
                right_hand_cmd_value[:HAND_DOF],
            ],
            axis=0,
        ).astype(np.float32)

        raw_top = ros_image_to_numpy_rgb(top_frames[i])
        top_img = _require_expected_image_shape(
            preprocess_to_vla_shape(raw_top),
            TOPIC_TOP_IMG,
        )

        left_img = _require_expected_image_shape(
            _preprocess_cached_image(
                left_frames,
                left_idx,
                left_decode_cache,
                left_preprocess_cache,
            ),
            TOPIC_LEFT_IMG,
        )

        right_img = _require_expected_image_shape(
            _preprocess_cached_image(
                right_frames,
                right_idx,
                right_decode_cache,
                right_preprocess_cache,
            ),
            TOPIC_RIGHT_IMG,
        )

        frame = {
            "task": task,
            OBS_TOP_KEY: top_img,
            OBS_LEFT_WRIST_KEY: left_img,
            OBS_RIGHT_WRIST_KEY: right_img,
            "observation.state": _pack_position_state(state),
            # action 仍保持 14 维，不能随着 state 加入 EE pose 而扩维。
            "action": _pack_action(action14),
        }
        if include_tactile:
            assert tactile_stats is not None
            frame[OBS_LEFT_TACTILE_KEY] = _require_expected_image_shape(
                _render_tactile_hand_image(
                    side="left",
                    indices=left_tactile_indices,
                    tactile_series=left_tactile_series,
                    decode_caches=left_tactile_decode_caches,
                    stats=tactile_stats,
                    render_state=tactile_render_state,
                ),
                OBS_LEFT_TACTILE_KEY,
            )
            frame[OBS_RIGHT_TACTILE_KEY] = _require_expected_image_shape(
                _render_tactile_hand_image(
                    side="right",
                    indices=right_tactile_indices,
                    tactile_series=right_tactile_series,
                    decode_caches=right_tactile_decode_caches,
                    stats=tactile_stats,
                    render_state=tactile_render_state,
                ),
                OBS_RIGHT_TACTILE_KEY,
            )

        ds.add_frame(frame)
        frames_added += 1
        current_chunk_frames += 1

    if frames_added == 0:
        raise RuntimeError(f"No frames were produced from {mcap_path}. Check topics/fps/data.")

    if dropped_unreliable > 0:
        print(
            f"[WARN] Dropped {dropped_unreliable} anchor frames in {mcap_path.name} due to the "
            f"{EXTRAPOLATION_TOLERANCE_S * 1000:.0f} ms alignment gate"
        )

    _log_dt_stats(mcap_path, split_count + 1, current_chunk_frames, current_chunk_deltas_s, fps)
    ds.save_episode()
    return frames_added, dropped_non_monotonic, split_count


def convert_mcap_to_lerobot(
    mcap_path: Path,
    out_root: Path,
    repo_id: str,
    task: str,
    fps: int = 60,
    image_size: int = EXPECTED_IMAGE_SHAPE[0],
    robot_type: str = "dual_arm_inspire_hand",
    numeric_interp_mode: str = "strict",
    include_tactile: bool = False,
    tactile_baseline_seconds: float = TACTILE_BASELINE_SECONDS,
    tactile_pressure_scale: float | None = None,
    tactile_delta_scale: float | None = None,
):
    features = build_features_pi05(include_tactile=include_tactile)
    ds = open_or_resume_dataset(out_root, repo_id, fps, features, robot_type)

    frames_added, dropped_non_monotonic, split_count = _convert_one_mcap_into_dataset(
        ds,
        mcap_path,
        task,
        fps,
        image_size,
        numeric_interp_mode=numeric_interp_mode,
        include_tactile=include_tactile,
        tactile_baseline_seconds=tactile_baseline_seconds,
        tactile_pressure_scale=tactile_pressure_scale,
        tactile_delta_scale=tactile_delta_scale,
    )
    ds.finalize()
    print(
        f"Converted {mcap_path} -> episode with {frames_added} frames; "
        f"Dropped {dropped_non_monotonic} non-monotonic frames; "
        f"Split into {split_count + 1} sub-episodes due to drops"
    )
    print(f"Dataset written to {out_root} (repo_id={repo_id})")


def _pick_text_value(node) -> Optional[str]:
    if isinstance(node, dict):
        if node.get("en"):
            return str(node["en"]).strip()
        if node.get("zh"):
            return str(node["zh"]).strip()
    if isinstance(node, str):
        return node.strip()
    return None


def _render_skill_text(skill: Optional[str], obj: Optional[str], target: Optional[str]) -> Optional[str]:
    if not skill:
        return None
    rendered = skill
    if obj:
        rendered = rendered.replace("{A}", obj)
    if target:
        rendered = rendered.replace("{B}", target)
    return rendered


def _build_task_from_annotation(ann: dict, default_task: str) -> str:
    desc = _pick_text_value(ann.get("description"))
    if desc:
        return desc
    segment_name = _pick_text_value(ann.get("segmentName"))
    if segment_name:
        return segment_name
    skill = _pick_text_value(ann.get("skill"))
    obj = _pick_text_value(ann.get("object"))
    target = _pick_text_value(ann.get("target"))
    adv = _pick_text_value(ann.get("adverbial"))
    skill_rendered = _render_skill_text(skill, obj, target)
    parts = []
    if skill_rendered:
        parts.append(skill_rendered)
    if obj and (not skill_rendered or obj not in skill_rendered):
        parts.append(obj)
    if target and (not skill_rendered or target not in skill_rendered):
        parts.append(target)
    if adv:
        parts.append(adv)
    return " ".join(parts) if parts else default_task


def _load_annotation_segments(json_path: Path, default_task: str) -> tuple[list[dict], Optional[str]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    file_name = (payload.get("file") or {}).get("fileName")
    segments = []
    for ann in payload.get("annotations", []) or []:
        start_ns = ann.get("startTimeNs")
        end_ns = ann.get("endTimeNs")
        if start_ns is None or end_ns is None:
            continue
        try:
            start_s = float(start_ns) * 1e-9
            end_s = float(end_ns) * 1e-9
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        task = _build_task_from_annotation(ann, default_task)
        segments.append(
            {
                "id": ann.get("id"),
                "start_s": start_s,
                "end_s": end_s,
                "task": task,
            }
        )
    segments.sort(key=lambda x: x["start_s"])
    return segments, file_name


def _get_mcap_start_log_time_s(mcap_path: Path, topics: list[str]) -> float:
    for m in read_ros2_messages(str(mcap_path), topics=topics):
        return m.log_time_ns / 1e9
    raise RuntimeError(f"No ROS2 messages found in {mcap_path}")


def _convert_one_mcap_with_annotations_into_dataset(
    ds: LeRobotDataset,
    mcap_path: Path,
    anno_json_path: Path,
    task: str,
    fps: int,
    image_size: int,
    numeric_interp_mode: str = "strict",
    include_tactile: bool = False,
    tactile_baseline_seconds: float = TACTILE_BASELINE_SECONDS,
    tactile_pressure_scale: float | None = None,
    tactile_delta_scale: float | None = None,
) -> int:
    segments, file_name = _load_annotation_segments(anno_json_path, default_task=task)
    if not segments:
        raise RuntimeError(f"No valid annotations found in {anno_json_path}")
    if file_name and file_name != mcap_path.name:
        print(f"[WARN] JSON fileName={file_name} does not match mcap={mcap_path.name}")

    topics = [
        TOPIC_RIGHT_IMG, TOPIC_LEFT_IMG, TOPIC_TOP_IMG,
        TOPIC_VLA_ARM_STATE, TOPIC_LEFT_EE_POSITION, TOPIC_LEFT_EE_RPY,
        TOPIC_RIGHT_EE_POSITION, TOPIC_RIGHT_EE_RPY,
        TOPIC_LEFT_HAND_JOINT, TOPIC_RIGHT_HAND_JOINT,
    ]
    if include_tactile:
        for side in ("left", "right"):
            topics.extend(TACTILE_TOPIC_BY_SIDE[side].values())
    t0 = _get_mcap_start_log_time_s(mcap_path, topics)

    total_frames = 0
    total_dropped_non_monotonic = 0
    total_split_count = 0
    for idx, seg in enumerate(segments):
        print(
            f"[{idx+1}/{len(segments)}] Segment id={seg['id']} "
            f"{seg['start_s']:.3f}-{seg['end_s']:.3f}s task={seg['task']}"
        )
        frames_added, dropped_non_monotonic, split_count = _convert_one_mcap_into_dataset(
            ds=ds,
            mcap_path=mcap_path,
            task=seg["task"],
            fps=fps,
            image_size=image_size,
            time_window=(seg["start_s"], seg["end_s"]),
            time_offset_s=t0,
            numeric_interp_mode=numeric_interp_mode,
            include_tactile=include_tactile,
            tactile_baseline_seconds=tactile_baseline_seconds,
            tactile_pressure_scale=tactile_pressure_scale,
            tactile_delta_scale=tactile_delta_scale,
        )
        total_frames += frames_added
        total_dropped_non_monotonic += dropped_non_monotonic
        total_split_count += split_count
    print(
        f"Converted {mcap_path} -> {len(segments)} episodes with {total_frames} frames; "
        f"Dropped {total_dropped_non_monotonic} non-monotonic frames; "
        f"Split into {len(segments) + total_split_count} sub-episodes due to drops"
    )
    return len(segments)


def convert_mcap_to_lerobot_with_annotations(
    mcap_path: Path,
    anno_json_path: Path,
    out_root: Path,
    repo_id: str,
    task: str,
    fps: int = 60,
    image_size: int = EXPECTED_IMAGE_SHAPE[0],
    robot_type: str = "dual_arm_inspire_hand",
    numeric_interp_mode: str = "strict",
    include_tactile: bool = False,
    tactile_baseline_seconds: float = TACTILE_BASELINE_SECONDS,
    tactile_pressure_scale: float | None = None,
    tactile_delta_scale: float | None = None,
):
    features = build_features_pi05(include_tactile=include_tactile)
    ds = open_or_resume_dataset(out_root, repo_id, fps, features, robot_type)

    _convert_one_mcap_with_annotations_into_dataset(
        ds=ds,
        mcap_path=mcap_path,
        anno_json_path=anno_json_path,
        task=task,
        fps=fps,
        image_size=image_size,
        numeric_interp_mode=numeric_interp_mode,
        include_tactile=include_tactile,
        tactile_baseline_seconds=tactile_baseline_seconds,
        tactile_pressure_scale=tactile_pressure_scale,
        tactile_delta_scale=tactile_delta_scale,
    )

    ds.finalize()
    print(f"Dataset written to {out_root} (repo_id={repo_id})")


def _find_annotation_for_mcap(mcap_path: Path, anno_json_dir: Path) -> Path:
    return anno_json_dir / (mcap_path.stem + ".json")


def convert_mcap_dir_to_lerobot(
    mcap_dir: Path,
    out_root: Path,
    repo_id: str,
    task: str,
    fps: int = 60,
    image_size: int = EXPECTED_IMAGE_SHAPE[0],
    robot_type: str = "dual_arm_inspire_hand",
    anno_json_dir: Optional[Path] = None,
    numeric_interp_mode: str = "strict",
    include_tactile: bool = False,
    tactile_baseline_seconds: float = TACTILE_BASELINE_SECONDS,
    tactile_pressure_scale: float | None = None,
    tactile_delta_scale: float | None = None,
):
    mcap_files = sorted(p for p in mcap_dir.glob("*.mcap") if p.is_file())
    if not mcap_files:
        raise RuntimeError(f"No .mcap files found in {mcap_dir}")

    features = build_features_pi05(include_tactile=include_tactile)
    ds = open_or_resume_dataset(out_root, repo_id, fps, features, robot_type)

    ok_count = 0
    fail_list = []  # [(path_str, error_str), ...]

    for idx, mcap_path in enumerate(mcap_files):
        print(f"[{idx+1}/{len(mcap_files)}] Converting {mcap_path} ...")
        try:
            if anno_json_dir is None:
                frames_added, dropped_non_monotonic, split_count = _convert_one_mcap_into_dataset(
                    ds,
                    mcap_path,
                    task,
                    fps,
                    image_size,
                    numeric_interp_mode=numeric_interp_mode,
                    include_tactile=include_tactile,
                    tactile_baseline_seconds=tactile_baseline_seconds,
                    tactile_pressure_scale=tactile_pressure_scale,
                    tactile_delta_scale=tactile_delta_scale,
                )
                print(
                    f"Converted {mcap_path} -> episode with {frames_added} frames; "
                    f"Dropped {dropped_non_monotonic} non-monotonic frames; "
                    f"Split into {split_count + 1} sub-episodes due to drops"
                )
            else:
                anno_json_path = _find_annotation_for_mcap(mcap_path, anno_json_dir)
                if not anno_json_path.exists():
                    raise FileNotFoundError(f"Missing annotation JSON: {anno_json_path}")
                _convert_one_mcap_with_annotations_into_dataset(
                    ds=ds,
                    mcap_path=mcap_path,
                    anno_json_path=anno_json_path,
                    task=task,
                    fps=fps,
                    image_size=image_size,
                    numeric_interp_mode=numeric_interp_mode,
                    include_tactile=include_tactile,
                    tactile_baseline_seconds=tactile_baseline_seconds,
                    tactile_pressure_scale=tactile_pressure_scale,
                    tactile_delta_scale=tactile_delta_scale,
                )
            ok_count += 1
        except Exception as e:
            # 关键：跳过坏 episode，不中断整体
            err = f"{type(e).__name__}: {e}"
            fail_list.append((str(mcap_path), err))
            print(f"[SKIP] {mcap_path} -> {err}")

            # 防止极少数“半写入 episode_buffer”导致占内存（可选但推荐）
            try:
                ds.clear_episode_buffer(delete_images=True)
            except Exception:
                pass
            continue

    ds.finalize()

    fail_count = len(fail_list)
    total = len(mcap_files)
    print("\n========== Conversion Summary ==========")
    print(f"Total files : {total}")
    print(f"Success     : {ok_count}")
    print(f"Failed      : {fail_count}")

    if fail_count > 0:
        print("\nFailed episodes:")
        for p, err in fail_list:
            print(f" - {p}\n   {err}")

    print(f"\nAll done. Dataset root: {out_root} (repo_id={repo_id})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=str, required=False, help="Input single .mcap file")
    parser.add_argument("--mcap-dir", type=str, required=False, help="Directory containing multiple .mcap")
    parser.add_argument(
        "--anno-json",
        type=str,
        required=False,
        help="Annotation JSON file; split into episodes by segments when set",
    )
    parser.add_argument(
        "--anno-json-dir",
        type=str,
        required=False,
        help="Directory of annotation JSON files for --mcap-dir (same basename as .mcap)",
    )
    parser.add_argument("--out", type=str, required=True, help="Output LeRobot dataset root")
    parser.add_argument("--repo-id", type=str, required=True, help="LeRobot dataset repo_id (local can be any string)")
    parser.add_argument("--task", type=str, default="octopus_teleop", help="Task string")
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Dataset metadata fps; camera-anchor alignment uses image header.stamp (default 60)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=EXPECTED_IMAGE_SHAPE[0],
        help="Deprecated, ignored. Visual schema is fixed to 224x224 after offline preprocessing.",
    )
    parser.add_argument(
        "--numeric-interp-mode",
        choices=("strict", "continuous"),
        default="strict",
        help=(
            "How to resample arm/hand/EE numeric streams at top-camera anchors. "
            "strict enforces the 50 ms bracketing gate; continuous interpolates across gaps "
            "and holds endpoints to keep more top-camera frames."
        ),
    )
    parser.add_argument(
        "--include-tactile",
        action="store_true",
        help=(
            "Convert Inspire hand mono16 tactile topics into "
            "observation.images.left_tactile/right_tactile pseudo images."
        ),
    )
    parser.add_argument(
        "--tactile-baseline-seconds",
        type=float,
        default=TACTILE_BASELINE_SECONDS,
        help="Seconds from the first top-camera anchor used for tactile median baseline.",
    )
    parser.add_argument(
        "--tactile-pressure-scale",
        type=float,
        default=0.0,
        help=(
            "Fixed tactile pressure scale. Use 0 to infer each episode scale from the "
            f"{TACTILE_SCALE_PERCENTILE:g}th percentile."
        ),
    )
    parser.add_argument(
        "--tactile-delta-scale",
        type=float,
        default=0.0,
        help=(
            "Fixed tactile delta scale. Use 0 to infer each episode scale from the "
            f"{TACTILE_SCALE_PERCENTILE:g}th percentile."
        ),
    )
    parser.add_argument("--robot-type", type=str, default="dual_arm_inspire_hand", help="robot_type in metadata")

    args = parser.parse_args()
    _ensure_runtime_dependencies()

    if bool(args.mcap) == bool(args.mcap_dir):
        raise SystemExit("Must specify exactly one of --mcap OR --mcap-dir")
    if args.anno_json and args.mcap_dir:
        raise SystemExit("--anno-json is only supported with --mcap (single file)")
    if args.anno_json_dir and not args.mcap_dir:
        raise SystemExit("--anno-json-dir is only supported with --mcap-dir")
    if args.image_size != EXPECTED_IMAGE_SHAPE[0]:
        print(
            f"[WARN] --image-size={args.image_size} is ignored; "
            f"visual schema is fixed to {EXPECTED_IMAGE_SHAPE[0]}x{EXPECTED_IMAGE_SHAPE[1]}."
        )
    if args.tactile_baseline_seconds <= 0.0:
        raise SystemExit("--tactile-baseline-seconds must be positive")

    out_root = Path(args.out)
    tactile_pressure_scale = args.tactile_pressure_scale if args.tactile_pressure_scale > 0.0 else None
    tactile_delta_scale = args.tactile_delta_scale if args.tactile_delta_scale > 0.0 else None

    if args.mcap_dir:
        convert_mcap_dir_to_lerobot(
            mcap_dir=Path(args.mcap_dir),
            out_root=out_root,
            repo_id=args.repo_id,
            task=args.task,
            fps=args.fps,
            image_size=args.image_size,
            robot_type=args.robot_type,
            anno_json_dir=Path(args.anno_json_dir) if args.anno_json_dir else None,
            numeric_interp_mode=args.numeric_interp_mode,
            include_tactile=args.include_tactile,
            tactile_baseline_seconds=args.tactile_baseline_seconds,
            tactile_pressure_scale=tactile_pressure_scale,
            tactile_delta_scale=tactile_delta_scale,
        )
    else:
        if args.anno_json:
            convert_mcap_to_lerobot_with_annotations(
                mcap_path=Path(args.mcap),
                anno_json_path=Path(args.anno_json),
                out_root=out_root,
                repo_id=args.repo_id,
                task=args.task,
                fps=args.fps,
                image_size=args.image_size,
                robot_type=args.robot_type,
                numeric_interp_mode=args.numeric_interp_mode,
                include_tactile=args.include_tactile,
                tactile_baseline_seconds=args.tactile_baseline_seconds,
                tactile_pressure_scale=tactile_pressure_scale,
                tactile_delta_scale=tactile_delta_scale,
            )
        else:
            convert_mcap_to_lerobot(
                mcap_path=Path(args.mcap),
                out_root=out_root,
                repo_id=args.repo_id,
                task=args.task,
                fps=args.fps,
                image_size=args.image_size,
                robot_type=args.robot_type,
                numeric_interp_mode=args.numeric_interp_mode,
                include_tactile=args.include_tactile,
                tactile_baseline_seconds=args.tactile_baseline_seconds,
                tactile_pressure_scale=tactile_pressure_scale,
                tactile_delta_scale=tactile_delta_scale,
            )


if __name__ == "__main__":
    main()
