"""Dataset wrapper for Pi0.5-style LeRobot offline training."""

import os
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


DEFAULT_CAMERAS = ("top", "left_wrist", "right_wrist")
TACTILE_CAMERAS = frozenset({"left_tactile", "right_tactile"})
# 触觉合并顺序固定为先左手、后右手。该顺序决定输出张量第 0 维的位置信息：
# 索引 0-10 对应左手，索引 11-21 对应右手。
TACTILE_SIDES = ("left_tactile", "right_tactile")
# 每只手内部严格按此 ID 顺序排列，不能按伪图像中的空间行顺序重新排序。
TACTILE_IDS = ("12", "13", "22", "23", "32", "33", "42", "43", "52", "54", "61")
# 每个触觉 ID 在 224x224 触觉伪图像中的区域，坐标格式为 (y0, y1, x0, x1)。
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
# 每个触觉区域最终统一缩放为单通道 14x14。
TACTILE_PATCH_SIZE = 14
# 单样本返回历史窗口 [W,22,1,14,14]；DataLoader 合批后变成 [B,W,22,1,14,14]。
TACTILE_OUTPUT_KEY = "tactile"
DEFAULT_IMAGE_SIZE = 224
# PI0.5 模型内部会把 [0, 1] 图像再映射到 [-1, 1]；这里保留 Normalize 步骤但使用 identity，
# 避免提前做 ImageNet 标准化导致模型侧二次归一化错误。
IMAGE_NORMALIZE_MEAN = (0.0, 0.0, 0.0)
IMAGE_NORMALIZE_STD = (1.0, 1.0, 1.0)


class Pi05LeRobotDataset(Dataset):
    """基于 LeRobotDataset 的 Pi0.5 训练数据包装器。

    这个类负责把原始 LeRobot 样本整理成训练循环直接使用的格式：
    - 配置指定的 RGB/触觉图像，统一 resize / crop，并可选做在线颜色扰动；
    - 当前时刻的机器人状态 state；
    - 从当前帧开始的未来动作序列 action_chunk；
    - 当前任务文本 task。
    """

    DEFAULT_TASK = "bimanual manipulation"

    def __init__(
        self,
        dataset_path: str | Path,
        chunk_size: int = 30,
        use_color_jitter: bool = True,
        image_size: int = DEFAULT_IMAGE_SIZE,
        state_dim: int | None = None,
        action_dim: int | None = None,
        cameras: tuple[str, ...] | list[str] = DEFAULT_CAMERAS,
        state_normalizer: Callable[[torch.Tensor], torch.Tensor] | Any | None = None,
        action_normalizer: Callable[[torch.Tensor], torch.Tensor] | Any | None = None,
        tactile_history_steps: int = 1,
        tactile_history_stride: int = 1,
    ) -> None:
        # 本地 LeRobot 数据集目录；LeRobotDataset 需要 repo_id 和 root，
        # 本地训练时 repo_id 只取目录名即可。
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        # action_chunk 的时间长度。每次 __getitem__ 返回当前帧之后 chunk_size 个动作。
        self.chunk_size = chunk_size
        self.image_size = int(image_size)
        self.cameras = self._normalize_cameras(cameras)
        self.tactile_history_steps = int(tactile_history_steps)
        self.tactile_history_stride = int(tactile_history_stride)
        if self.tactile_history_steps <= 0 or self.tactile_history_stride <= 0:
            raise ValueError("Tactile history steps and stride must be positive.")
        self.lerobot_dataset = LeRobotDataset(repo_id=self.dataset_path.name, root=self.dataset_path)
        self.dataset = self.lerobot_dataset

        # 从数据集 metadata 读取真实维度，而不是完全信任外部配置。
        # 这样换数据集后如果 state/action 维度变了，代码仍以数据集为准。
        features = self.lerobot_dataset.features
        self.state_dim = self._feature_dim(features, "observation.state")
        self.action_dim = self._feature_dim(features, "action")
        self._warn_if_stale_expected_dim("observation.state", state_dim, self.state_dim)
        self._warn_if_stale_expected_dim("action", action_dim, self.action_dim)
        self.image_keys = self._resolve_image_keys()
        self.tactile_keys = self._resolve_tactile_keys()
        # 构造未来 action chunk 时只读取 action 列，避免重复处理图像等无关字段。
        self.action_dataset = self.lerobot_dataset.select_columns("action")
        self._validate_vector_features()

        # 颜色扰动只在取样时在线执行，不改变磁盘上的原始图像。
        self.color_jitter = (
            ColorJitter(brightness=0.08, contrast=0.08, saturation=0.08, hue=0.03)
            if use_color_jitter
            else None
        )
        # normalizer 可以是可调用对象，也可以是带 normalize() 方法的对象。
        self.state_normalizer = state_normalizer
        self.action_normalizer = action_normalizer
        # 预先缓存每个 episode 在全局 dataset 中的 [start, end) 范围，
        # 后面构造 action_chunk 时需要防止跨 episode 取动作。
        self.episode_ranges = self._build_episode_ranges()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        item = self.dataset[idx]

        images = {}
        for output_key, dataset_key in self.image_keys.items():
            camera_name = self._camera_from_output_key(output_key)
            images[output_key] = self._prepare_image(item[dataset_key], camera_name=camera_name)
        # 当前帧机器人状态，转换为 float32 后再按需归一化。
        state = torch.as_tensor(item["observation.state"], dtype=torch.float32)
        state = self._apply_normalizer(state, self.state_normalizer)

        # LeRobot 的 episode_index 用来确定当前样本属于哪个 episode。
        # 动作 chunk 只能在同一个 episode 内向后取。
        episode_index = self._to_int(item["episode_index"])
        if self.tactile_keys:
            tactile_window, tactile_window_mask = self._get_tactile_window(idx, episode_index)
            images[TACTILE_OUTPUT_KEY] = tactile_window
            images[f"{TACTILE_OUTPUT_KEY}_mask"] = tactile_window_mask
        action_chunk = self._get_action_chunk(idx, episode_index)
        action_chunk = self._apply_normalizer(action_chunk, self.action_normalizer)
        task = self._resolve_task(item)

        return {
            **images,
            "state": state,
            "action_chunk": action_chunk,
            "task": task,
        }

    def _resolve_image_keys(self) -> dict[str, str]:
        camera_keys = list(self.dataset.meta.camera_keys)
        if not camera_keys:
            raise ValueError("No image keys found in LeRobot dataset metadata.")

        image_key_map = {
            self._output_image_key(camera): self._dataset_image_key(camera)
            for camera in self.cameras
            if camera not in TACTILE_CAMERAS
        }
        missing = [dataset_key for dataset_key in image_key_map.values() if dataset_key not in camera_keys]
        if missing:
            raise ValueError(
                "Dataset is missing required Pi0.5 image keys: "
                f"{missing}. Available keys: {camera_keys}"
            )
        return image_key_map

    def _resolve_tactile_keys(self) -> dict[str, str]:
        # 只有同时配置左、右触觉，并保持 left_tactile -> right_tactile 顺序时才生成触觉张量。
        configured = [camera for camera in self.cameras if camera in TACTILE_CAMERAS]
        if not configured:
            return {}
        if tuple(configured) != TACTILE_SIDES:
            raise ValueError(
                "Tactile input requires both sides in fixed order: "
                f"{TACTILE_SIDES}; configured tactile cameras: {tuple(configured)}"
            )

        camera_keys = set(self.dataset.meta.camera_keys)
        tactile_keys = {side: self._dataset_image_key(side) for side in TACTILE_SIDES}
        missing = [key for key in tactile_keys.values() if key not in camera_keys]
        if missing:
            raise ValueError(
                f"Dataset is missing required tactile image keys: {missing}. "
                f"Available keys: {sorted(camera_keys)}"
            )
        return tactile_keys

    def _validate_vector_features(self) -> None:
        # 再次校验 state/action 的维度，避免加载过程中 metadata 或 features 不一致。
        features = self.lerobot_dataset.features
        for key, expected_dim in (("observation.state", self.state_dim), ("action", self.action_dim)):
            actual_dim = self._feature_dim(features, key)
            if actual_dim != expected_dim:
                raise ValueError(f"Dataset {key} shape changed while loading: got {actual_dim}, expected {expected_dim}.")

    def _build_episode_ranges(self) -> dict[int, tuple[int, int]]:
        # 将每个 episode 的全局索引范围整理成字典：
        # {episode_index: (dataset_from_index, dataset_to_index)}。
        ranges: dict[int, tuple[int, int]] = {}
        for episode_index in range(self.dataset.meta.total_episodes):
            episode = self.dataset.meta.episodes[episode_index]
            start = self._to_int(episode["dataset_from_index"])
            end = self._to_int(episode["dataset_to_index"])
            ranges[episode_index] = (start, end)
        return ranges

    def _get_action_chunk(self, idx: int, episode_index: int) -> torch.Tensor:
        # 如果 idx + step 超过当前 episode 的最后一帧，就重复最后一个有效动作。
        # 这样 action_chunk 长度恒定，同时不会误取到下一个 episode 的动作。
        _, episode_end = self.episode_ranges[episode_index]
        last_valid_idx = episode_end - 1

        actions = []
        for step in range(self.chunk_size):
            query_idx = min(idx + step, last_valid_idx)
            action = self.action_dataset[query_idx]["action"]
            action_tensor = torch.as_tensor(action, dtype=torch.float32).flatten()
            if action_tensor.numel() != self.action_dim:
                raise ValueError(
                    f"Action at dataset index {query_idx} has dim {action_tensor.numel()}, expected {self.action_dim}."
                )
            actions.append(action_tensor)

        return torch.stack(actions, dim=0)

    def _get_tactile_window(
        self,
        idx: int,
        episode_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        episode_start, _ = self.episode_ranges[episode_index]
        tactile_frames = []
        valid_frames = []

        for history_offset in reversed(range(self.tactile_history_steps)):
            raw_idx = idx - history_offset * self.tactile_history_stride
            query_idx = max(raw_idx, episode_start)
            tactile_frames.append(self._prepare_tactile(self.dataset[query_idx]))
            valid_frames.append(raw_idx >= episode_start)

        return (
            torch.stack(tactile_frames, dim=0),
            torch.tensor(valid_frames, dtype=torch.bool),
        )

    def _prepare_image(self, image: Any, *, camera_name: str) -> torch.Tensor:
        image_tensor = self._to_image_tensor(image)
        interpolation = InterpolationMode.BILINEAR
        if tuple(image_tensor.shape[-2:]) != (self.image_size, self.image_size):
            image_tensor = TF.resize(
                image_tensor,
                [self.image_size, self.image_size],
                interpolation=interpolation,
                antialias=True,
            )
        image_tensor = TF.center_crop(image_tensor, [self.image_size, self.image_size])
        if self.color_jitter is not None:
            image_tensor = self.color_jitter(image_tensor)
        image_tensor = TF.normalize(
            image_tensor,
            mean=IMAGE_NORMALIZE_MEAN,
            std=IMAGE_NORMALIZE_STD,
        )
        return image_tensor

    def _prepare_tactile(self, item: dict[str, Any]) -> torch.Tensor:
        # 此处按 TACTILE_SIDES 固定顺序读取左右手，避免后续模型丢失手侧位置信息。
        side_images = [
            self._to_image_tensor(item[self.tactile_keys[side]])
            for side in TACTILE_SIDES
        ]
        return split_tactile_images(side_images)

    @staticmethod
    def _normalize_cameras(cameras: tuple[str, ...] | list[str]) -> tuple[str, ...]:
        normalized = tuple(str(camera).strip() for camera in cameras if str(camera).strip())
        if not normalized:
            raise ValueError("At least one image camera/key must be configured.")
        duplicates = sorted({camera for camera in normalized if normalized.count(camera) > 1})
        if duplicates:
            raise ValueError(f"Duplicate camera names are not allowed: {duplicates}")
        return normalized

    @staticmethod
    def _dataset_image_key(camera: str) -> str:
        return f"observation.images.{camera}"

    @staticmethod
    def _output_image_key(camera: str) -> str:
        return f"image_{camera}"

    @staticmethod
    def _camera_from_output_key(output_key: str) -> str:
        if not output_key.startswith("image_"):
            raise ValueError(f"Invalid dataset image output key: {output_key}")
        return output_key.removeprefix("image_")

    def _resolve_task(self, item: dict[str, Any]) -> str:
        # 优先使用样本里已经展开的 task 字符串；没有时再通过 task_index 查 metadata。
        task = item.get("task")
        if task is None and "task_index" in item:
            task = self._task_from_index(item["task_index"])
        if isinstance(task, str) and task.strip():
            resolved_task = task.strip()
            self._print_first_task(resolved_task)
            return resolved_task
        if not getattr(self, "_warned_missing_task", False):
            self._warned_missing_task = True
            if self._should_print_dataset_message():
                print(
                    "[dataset] Warning: failed to resolve task from sample; "
                    f"falling back to DEFAULT_TASK={self.DEFAULT_TASK!r}. "
                    f"task={item.get('task')!r}, task_index={item.get('task_index')!r}"
                )
        self._print_first_task(self.DEFAULT_TASK)
        return self.DEFAULT_TASK

    def _print_first_task(self, task: str) -> None:
        if getattr(self, "_printed_first_task", False):
            return
        self._printed_first_task = True
        if self._should_print_dataset_message():
            print(f"[dataset] First task: {task!r}")

    @staticmethod
    def _should_print_dataset_message() -> bool:
        worker_info = get_worker_info()
        is_primary_worker = worker_info is None or worker_info.id == 0
        is_primary_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0
        return is_primary_worker and is_primary_rank

    def _task_from_index(self, task_index: Any) -> str | None:
        # 不同版本 LeRobot 的 meta.tasks 可能是 DataFrame，也可能是 dict，
        # 这里兼容两种常见结构。
        task_idx = self._to_int(task_index)
        tasks = getattr(self.lerobot_dataset.meta, "tasks", None)
        if tasks is None:
            return None
        try:
            if hasattr(tasks, "iloc"):
                row = tasks.iloc[task_idx]
                for key in ("task", "tasks", "description"):
                    if key in row and isinstance(row[key], str):
                        return row[key]
                row_name = getattr(row, "name", None)
                if isinstance(row_name, str) and row_name.strip():
                    return row_name
                if "task_index" in tasks:
                    matches = tasks[tasks["task_index"] == task_idx]
                    if len(matches) > 0:
                        match_name = matches.iloc[0].name
                        if isinstance(match_name, str) and match_name.strip():
                            return match_name
            if isinstance(tasks, dict):
                value = tasks.get(task_idx, tasks.get(str(task_idx)))
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    return value.get("task")
        except (IndexError, KeyError, TypeError, ValueError):
            return None
        return None

    def _to_image_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            image_tensor = image.detach().clone()
            if image_tensor.ndim != 3:
                raise ValueError(f"Expected 3D image tensor, got shape {tuple(image_tensor.shape)}")
            if image_tensor.shape[0] != 3 and image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
            image_tensor = image_tensor.to(dtype=torch.float32)
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor / 255.0
            return image_tensor

        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
            return TF.to_tensor(image)

        return TF.to_tensor(image)

    @staticmethod
    def _to_int(value: Any) -> int:
        if isinstance(value, torch.Tensor):
            return int(value.item())
        if isinstance(value, np.ndarray):
            return int(value.item())
        return int(value)

    @staticmethod
    def _apply_normalizer(
        tensor: torch.Tensor,
        normalizer: Callable[[torch.Tensor], torch.Tensor] | Any | None,
    ) -> torch.Tensor:
        # 兼容两种 normalizer 接口：normalizer.normalize(x) 或 normalizer(x)。
        if normalizer is None:
            return tensor
        if hasattr(normalizer, "normalize"):
            return normalizer.normalize(tensor)
        return normalizer(tensor)

    @staticmethod
    def _feature_dim(features: dict[str, dict], key: str) -> int:
        # LeRobot features 中的 shape 可能是多维，这里展开成训练侧使用的一维总维度。
        if key not in features:
            raise ValueError(f"Dataset is missing required feature '{key}'.")
        shape = tuple(features[key].get("shape", ()))
        if not shape:
            raise ValueError(f"Dataset feature '{key}' must have a non-empty shape.")
        return int(np.prod(shape))

    @staticmethod
    def _warn_if_stale_expected_dim(key: str, expected_dim: int | None, actual_dim: int) -> None:
        # 外部配置里的 expected_dim 只作为提示；真正使用的数据维度来自数据集 metadata。
        if expected_dim is None or int(expected_dim) == actual_dim:
            return
        warnings.warn(
            f"Ignoring configured {key} dim {expected_dim}; dataset metadata reports {actual_dim}.",
            RuntimeWarning,
            stacklevel=3,
        )


def split_tactile_images(side_images: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Convert left/right tactile pseudo-images into fixed-order pressure patches.

    Output order is left hand followed by right hand. Within each hand the
    tactile IDs are ordered as 12,13,22,23,32,33,42,43,52,54,61. The returned
    single-sample shape is [22,1,14,14].
    """
    if len(side_images) != len(TACTILE_SIDES):
        raise ValueError(f"Expected {len(TACTILE_SIDES)} tactile side images, got {len(side_images)}")

    patches: list[torch.Tensor] = []
    for side, image in zip(TACTILE_SIDES, side_images, strict=True):
        if image.ndim != 3 or image.shape[0] < 1:
            raise ValueError(f"{side} tactile image must have shape (C,H,W), got {tuple(image.shape)}")
        if tuple(image.shape[-2:]) != (224, 224):
            raise ValueError(f"{side} tactile image must be 224x224, got {tuple(image.shape[-2:])}")

        # 触觉伪图通道含义为 R=接触压力、G=正向变化、B=负向变化；这里只保留 R 通道。
        pressure = image[0:1]
        for tactile_id in TACTILE_IDS:
            y0, y1, x0, x1 = TACTILE_LAYOUT[tactile_id]
            patch = pressure[:, y0:y1, x0:x1]
            # 各 ID 原始区域尺寸不同，使用最近邻插值统一为 [1,14,14]，避免混入平滑后的伪值。
            patch = TF.resize(
                patch,
                [TACTILE_PATCH_SIZE, TACTILE_PATCH_SIZE],
                interpolation=InterpolationMode.NEAREST,
                antialias=False,
            )
            patches.append(patch)

    # 堆叠顺序由 TACTILE_SIDES 和 TACTILE_IDS 共同保证，输出为 [22,1,14,14]。
    return torch.stack(patches, dim=0)
