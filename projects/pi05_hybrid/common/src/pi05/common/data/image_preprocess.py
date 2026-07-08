"""Image preprocessing shared by Pi0.5 training checks and deployment.

The helpers here operate on RGB numpy arrays and return channel-first float
tensors in the same fixed square shape expected by the policy preprocessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


ResizeMode = Literal["resize_crop", "resize_pad"]


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """Configuration for deterministic deployment image preprocessing."""

    image_size: int = 224
    mode: ResizeMode = "resize_pad"


def preprocess_rgb_image(rgb: np.ndarray, config: ImagePreprocessConfig) -> torch.Tensor:
    """Convert an RGB image into a normalized CHW float tensor."""
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape (H, W, 3), got {rgb.shape}")
    if int(config.image_size) <= 0:
        raise ValueError("image_size must be positive")
    if config.mode == "resize_pad":
        return _resize_pad(rgb, int(config.image_size))
    if config.mode == "resize_crop":
        return _resize_crop(rgb, int(config.image_size))
    raise ValueError(f"Unsupported image preprocess mode: {config.mode}")


def _resize_crop(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    tensor = TF.to_tensor(rgb)
    tensor = TF.resize(
        tensor,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = TF.center_crop(tensor, [image_size, image_size])
    return tensor.to(dtype=torch.float32)


def _resize_pad(rgb: np.ndarray, image_size: int) -> torch.Tensor:
    src_h, src_w = rgb.shape[:2]
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"Invalid image size: {(src_h, src_w)}")

    scale = min(image_size / float(src_w), image_size / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = _cv2_resize_rgb(rgb, width=new_w, height=new_h)

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    pad_top = (image_size - new_h) // 2
    pad_left = (image_size - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized
    return TF.to_tensor(canvas).to(dtype=torch.float32)


def _cv2_resize_rgb(rgb: np.ndarray, *, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - cv2 is present in the robot env.
        raise RuntimeError("resize_pad image preprocessing requires opencv-python") from exc
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
