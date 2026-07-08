from __future__ import annotations

import torch
from torch.utils.data._utils.collate import default_collate

from pi05.train.data.dataset import (
    Pi05LeRobotDataset,
    TACTILE_IDS,
    TACTILE_LAYOUT,
    split_tactile_images,
)


def _tactile_image(offset: int) -> torch.Tensor:
    image = torch.zeros(3, 224, 224, dtype=torch.float32)
    for index, tactile_id in enumerate(TACTILE_IDS):
        y0, y1, x0, x1 = TACTILE_LAYOUT[tactile_id]
        # R 通道写入可识别的 ID 顺序值；G/B 写入干扰值，验证转换确实只保留 R。
        image[0, y0:y1, x0:x1] = float(offset + index)
        image[1, y0:y1, x0:x1] = 999.0
        image[2, y0:y1, x0:x1] = 999.0
    return image


def test_split_tactile_images_uses_fixed_side_and_id_order() -> None:
    tactile = split_tactile_images([_tactile_image(0), _tactile_image(100)])

    assert tactile.shape == (22, 1, 14, 14)
    # 前 11 个是左手指定 ID 顺序，后 11 个是右手同一 ID 顺序。
    expected = list(range(11)) + list(range(100, 111))
    assert [float(patch.unique().item()) for patch in tactile] == expected


def test_split_tactile_images_collates_batch_dimension() -> None:
    tactile = split_tactile_images([_tactile_image(0), _tactile_image(100)])
    batch = default_collate([{"tactile": tactile}, {"tactile": tactile}])

    # 单样本 [22,1,14,14] 经 DataLoader/default_collate 后增加最前面的 B 维。
    assert batch["tactile"].shape == (2, 22, 1, 14, 14)


def test_tactile_history_window_pads_without_crossing_episode() -> None:
    dataset = Pi05LeRobotDataset.__new__(Pi05LeRobotDataset)
    dataset.tactile_history_steps = 4
    dataset.tactile_history_stride = 2
    dataset.episode_ranges = {3: (10, 20)}
    dataset.dataset = [{"value": index} for index in range(30)]
    dataset._prepare_tactile = lambda item: torch.full((22, 1, 14, 14), float(item["value"]))

    window, mask = dataset._get_tactile_window(idx=13, episode_index=3)

    assert window.shape == (4, 22, 1, 14, 14)
    assert [float(frame[0, 0, 0, 0]) for frame in window] == [10.0, 10.0, 11.0, 13.0]
    assert mask.tolist() == [False, False, True, True]
