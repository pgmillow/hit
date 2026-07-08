"""训练构建器：负责数据、优化器和学习率调度器的创建。"""

from __future__ import annotations

import math

import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from pi05.common.config.schema import DataConfig, TactileConfig, TrainingConfig
from pi05.common.data.normalization import build_state_action_normalizers
from pi05.train.data.dataset import Pi05LeRobotDataset


def build_train_dataloader(
    data_cfg: DataConfig,
    train_cfg: TrainingConfig,
    tactile_cfg: TactileConfig | None = None,
) -> DataLoader:
    """构建训练 DataLoader，并基于数据集统计量创建归一化器。"""
    dataset_path = data_cfg.resolved_dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # 先用一个 bootstrap 数据集估计 state/action 的归一化统计量。
    bootstrap_dataset = Pi05LeRobotDataset(
        dataset_path=dataset_path,
        chunk_size=data_cfg.chunk_size,
        use_color_jitter=data_cfg.use_color_jitter,
        image_size=data_cfg.image_size,
        state_dim=data_cfg.state_dim,
        action_dim=data_cfg.action_dim,
        cameras=data_cfg.cameras,
        tactile_history_steps=tactile_cfg.history_steps if tactile_cfg and tactile_cfg.enabled else 1,
        tactile_history_stride=tactile_cfg.history_stride if tactile_cfg and tactile_cfg.enabled else 1,
    )
    state_normalizer, action_normalizer = build_state_action_normalizers(bootstrap_dataset.dataset)

    # 复用已加载的数据集，避免对大型 parquet 数据完整构造两次。
    dataset = bootstrap_dataset
    dataset.state_normalizer = state_normalizer
    dataset.action_normalizer = action_normalizer
    return DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def count_training_steps(
    dataloader: DataLoader,
    train_cfg: TrainingConfig,
    *,
    num_processes: int = 1,
) -> int:
    """计算每个进程真实优化步数（real step），用于学习率调度。"""
    # Accelerate/DDP 会把 DataLoader 分片到各进程；scheduler 总步数必须按每个 rank
    # 实际迭代的 batch 数计算，否则多卡时 cosine decay 会按 world size 倍数变慢。
    batches_per_process = math.ceil(len(dataloader) / max(1, int(num_processes)))
    update_steps_per_epoch = math.ceil(batches_per_process / train_cfg.gradient_accumulation_steps)
    if train_cfg.max_steps_per_epoch is not None:
        # 若设置了每个 epoch 的上限，则按上限截断。
        update_steps_per_epoch = min(update_steps_per_epoch, train_cfg.max_steps_per_epoch)
    return update_steps_per_epoch * train_cfg.epochs


def build_optimizer(model: torch.nn.Module, train_cfg: TrainingConfig) -> Optimizer:
    """只优化可训练参数（如 LoRA 参数），避免更新冻结权重。"""
    return AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=train_cfg.lr,
    )


def build_lr_scheduler(optimizer: Optimizer, train_cfg: TrainingConfig, num_training_steps: int):
    """Warmup + Cosine 学习率调度。"""
    # 前 warmup_steps 线性升温，随后在 total steps 内余弦下降到接近 0。
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )
