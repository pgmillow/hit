"""Main training loop for Pi0.5 LoRA fine-tuning.

Pi0.5 LoRA 微调的主训练循环。
基于 HuggingFace Accelerate 实现，负责串联：配置 → 数据/模型构建 → 训练循环 → 导出适配器与部署包。
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from pi05.common.config.schema import ExperimentConfig
from pi05.common.model.builder import build_pi05_with_lora, get_pi05_policy_config
from pi05.common.runtime.bundle import export_deploy_bundle
from pi05.train.engine.batches import to_lerobot_pi05_batch
from pi05.train.engine.builders import (
    build_lr_scheduler,
    build_optimizer,
    build_train_dataloader,
    count_training_steps,
)
from pi05.train.engine.checkpoints import (
    export_final_adapter,
    maybe_resume,
    save_step_adapter_checkpoints,
)
from pi05.train.utils.logging import configure_logging, log_run_summary
from pi05.train.utils.seed import set_training_seed
from pi05.train.utils.tensorboard import launch_tensorboard


LOGGER = logging.getLogger(__name__)


@dataclass
class _AverageMeter:
    """累加器：记录某个指标的累计和与计数，用于计算窗口内平均值。"""

    total: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def avg(self) -> float:
        # 计数为 0 时返回 0，避免除零
        return 0.0 if self.count == 0 else self.total / self.count


class _LogWindow:
    """日志窗口：聚合若干 step 内的多个指标（loss/lr/grad_norm 等），到点统一输出平均值。"""

    def __init__(self) -> None:
        self._meters: dict[str, _AverageMeter] = {}

    def update(self, **metrics: float | None) -> None:
        # 逐个指标累加；值为 None 的指标跳过（例如未触发梯度裁剪时的 grad_norm）
        for key, value in metrics.items():
            if value is None:
                continue
            meter = self._meters.setdefault(key, _AverageMeter())
            meter.update(float(value))

    def as_dict(self) -> dict[str, float]:
        # 返回每个指标的窗口平均值
        return {key: meter.avg for key, meter in self._meters.items() if meter.count > 0}

    @property
    def count(self) -> int:
        # 当前窗口内累计的样本数（取任一指标的计数即可，各指标计数一致）
        if not self._meters:
            return 0
        first_meter = next(iter(self._meters.values()))
        return first_meter.count

    def reset(self) -> None:
        # 清空窗口，开始下一段统计
        self._meters.clear()


@dataclass(frozen=True)
class _LearningRateOverride:
    lr: float | None = None
    cosine_k: float | None = None


class _LearningRateControl:
    """Runtime LR override loaded from a small text file."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._last_signature: tuple[int, int] | None = None
        self._last_value: _LearningRateOverride | None = None
        self._last_error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def read(self, accelerator: Accelerator) -> _LearningRateOverride | None:
        if self.path is None or not self.path.exists():
            self._last_signature = None
            self._last_value = None
            self._last_error = None
            return None

        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature == self._last_signature:
            return self._last_value
        self._last_signature = signature

        try:
            value = self._parse(self.path.read_text(encoding="utf-8"))
        except ValueError as exc:
            message = str(exc)
            if message != self._last_error:
                accelerator.print(f"[train] warning: ignoring invalid lr_control_file: {message}")
            self._last_error = message
            self._last_value = None
            return None

        self._last_error = None
        self._last_value = value
        if value is None:
            accelerator.print(f"[train] lr control cleared by: {self.path}")
        elif value.lr is not None:
            accelerator.print(f"[train] lr override from {self.path}: lr={value.lr:.3e}")
        elif value.cosine_k is not None:
            accelerator.print(f"[train] lr override from {self.path}: cosine_k={value.cosine_k:g}")
        return value

    @staticmethod
    def _parse(text: str) -> _LearningRateOverride | None:
        lr: float | None = None
        cosine_k: float | None = None
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line:
                key, value = line.split(":", 1)
            else:
                key, value = "lr", line
            key = key.strip().lower()
            if key not in {"lr", "cosine_k", "k"}:
                continue
            try:
                parsed = float(value.strip())
            except ValueError as exc:
                raise ValueError(f"cannot parse {key} from {raw_line!r}") from exc
            if parsed < 0.0:
                raise ValueError(f"{key} must be >= 0, got {parsed}")
            if key == "lr":
                lr = parsed
            else:
                cosine_k = parsed
        if lr is None and cosine_k is None:
            return None
        return _LearningRateOverride(lr=lr, cosine_k=cosine_k)


class Pi05LoraTrainer:
    """Coordinates config, Accelerate, model/data builders, and the train loop.

    训练器主类：统筹配置、Accelerate、模型/数据构建器以及训练循环。
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> None:
        """训练入口：完成初始化、构建组件、执行训练循环的完整流程。"""
        # 配置日志与随机种子，保证可复现
        configure_logging()
        set_training_seed(self.config.training.seed)

        # 初始化 Accelerate（混合精度/梯度累积/多卡）后再估算总训练步数。
        # 多卡时 dataloader 会按进程分片，scheduler 必须按每个 rank 的真实优化步数计算。
        accelerator = self._build_accelerator()
        dataloader = build_train_dataloader(self.config.data, self.config.training, self.config.tactile)
        num_training_steps = count_training_steps(
            dataloader,
            self.config.training,
            num_processes=accelerator.num_processes,
        )

        # 初始化日志追踪器（TensorBoard）
        self._init_trackers(accelerator)
        if accelerator.is_main_process:
            # 仅主进程打印本次运行的配置摘要
            log_run_summary(LOGGER, self.config.run_summary())

        # 加载 Pi0.5 预训练权重并注入 LoRA 适配器（仅训练 LoRA 参数）
        # 多卡时按 rank 错峰串行加载，避免 N 个进程同时把 ~14GB fp32 权重读入 CPU 内存，
        # 触发主机 OOM（OOM Killer 会以 SIGKILL/-9 杀掉某个 rank）。
        model = self._load_model_staggered(accelerator)
        # 取出策略配置并对齐设备，构建前处理流水线（归一化/tokenize 等）
        policy_config = get_pi05_policy_config(model)
        policy_config.device = str(accelerator.device)
        preprocessor, _ = make_pi05_pre_post_processors(policy_config, dataset_stats=None)

        # 构建优化器与学习率调度器
        optimizer = build_optimizer(model, self.config.training)
        lr_scheduler = build_lr_scheduler(
            optimizer=optimizer,
            train_cfg=self.config.training,
            num_training_steps=num_training_steps,
        )

        # 交给 Accelerate 统一包装（分布式、混合精度、设备搬运）
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            dataloader,
            lr_scheduler,
        )
        # 如配置了断点续训，则从 checkpoint 恢复训练状态
        maybe_resume(accelerator, self.config.training.resume_from_checkpoint)
        self._train_loop(
            accelerator=accelerator,
            model=model,
            dataloader=dataloader,
            preprocessor=preprocessor,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_training_steps=num_training_steps,
            warmup_steps=self.config.training.warmup_steps,
        )

    def _build_accelerator(self) -> Accelerator:
        """构建 Accelerator：配置梯度累积、混合精度与日志后端。"""
        train_cfg = self.config.training
        log_cfg = self.config.logging
        mixed_precision = train_cfg.mixed_precision
        # 未显式指定时：有 GPU 用 bf16，否则不使用混合精度
        if mixed_precision is None:
            mixed_precision = "bf16" if torch.cuda.is_available() else "no"

        return Accelerator(
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard" if log_cfg.use_tensorboard else None,
            project_dir=str(log_cfg.resolved_tensorboard_dir) if log_cfg.use_tensorboard else None,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                # Staggered loading leaves the other ranks waiting while each
                # large PI0.5 model is loaded. The default 10-minute timeout is
                # too short for this machine and four sequential model loads.
                InitProcessGroupKwargs(timeout=timedelta(hours=2)),
            ],
        )

    def _load_model_staggered(self, accelerator: Accelerator) -> torch.nn.Module:
        """按本地 rank 顺序错峰加载模型，降低多进程同时加载权重时的 CPU 内存峰值。

        每个 rank 仅在轮到自己时执行 ``build_pi05_with_lora``（读取 ~14GB 权重 + 转 dtype
        的瞬时翻倍），加载后立即 gc，再统一 barrier，从而把瞬时峰值从 N×降到约 1×。
        """
        model: torch.nn.Module | None = None
        for rank in range(accelerator.num_processes):
            if accelerator.local_process_index == rank:
                accelerator.print(
                    f"[train] loading pretrained model on local_rank={rank} "
                    f"(staggered to avoid host OOM)"
                )
                model = build_pi05_with_lora(
                    config=self.config,
                    pretrained_path=self.config.model.pretrained_path,
                    init_adapter_from=self.config.training.init_adapter_from,
                )
                gc.collect()
            accelerator.wait_for_everyone()
        assert model is not None, "model failed to load during staggered loading"
        return model

    def _init_trackers(self, accelerator: Accelerator) -> None:
        """初始化 TensorBoard 追踪器；如配置开启则在主进程自动拉起 TensorBoard 服务。"""
        log_cfg = self.config.logging
        if not log_cfg.use_tensorboard:
            return
        accelerator.init_trackers(
            project_name=log_cfg.project_name,
            config=self.config.to_tracker_config(),
            init_kwargs={"tensorboard": {"flush_secs": 10}},
        )
        if accelerator.is_main_process and log_cfg.tensorboard_auto_launch:
            launch_tensorboard(
                logdir=log_cfg.resolved_tensorboard_dir,
                host=log_cfg.tensorboard_host,
                port=log_cfg.tensorboard_port,
            )

    def _train_loop(
        self,
        *,
        accelerator: Accelerator,
        model: torch.nn.Module,
        dataloader: Any,
        preprocessor: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        num_training_steps: int,
        warmup_steps: int,
    ) -> None:
        """核心训练循环：按 epoch 遍历数据，前向计算 loss、反向传播、更新参数并记录日志。"""
        train_cfg = self.config.training
        log_cfg = self.config.logging
        run_output_dir = log_cfg.run_output_dir

        model.train()
        real_steps = 0  # 全局优化步数（optimizer.step 次数）
        micro_steps = 0  # 全局 micro-batch 数（每次 dataloader 迭代累加）
        eff_batch_size = train_cfg.batch_size * accelerator.num_processes * train_cfg.gradient_accumulation_steps
        state_dim: int | None = None
        log_window = _LogWindow()
        lr_control = _LearningRateControl(train_cfg.lr_control_file)
        lr_override: _LearningRateOverride | None = None
        if lr_control.enabled:
            accelerator.print(f"[train] lr control file: {lr_control.path}")
        epoch_steps = 0
        accum_loss_sum: torch.Tensor | None = None
        accum_loss_count = 0
        try:
            for epoch in range(train_cfg.epochs):
                epoch_steps = 0
                for batch in dataloader:
                    micro_steps += 1
                    # 首个 batch 记录 state 维度，便于排查数据格式
                    if state_dim is None:
                        state_dim = int(batch["state"].shape[-1])
                        accelerator.print(f"[train] state_dim={state_dim}")

                    grad_norm_value: float | None = None
                    # accumulate 上下文：自动处理梯度累积，攒够 N 个 micro-batch 才同步一次
                    with accelerator.accumulate(model):
                        # 将原始 batch 转成 lerobot pi05 格式，再过前处理流水线
                        processed_batch = preprocessor(to_lerobot_pi05_batch(batch))
                        # 前向：返回 Flow Matching 损失
                        loss, loss_dict = model(processed_batch)
                        loss_for_log = loss.detach().float()
                        accum_loss_sum = (
                            loss_for_log
                            if accum_loss_sum is None
                            else accum_loss_sum + loss_for_log
                        )
                        accum_loss_count += 1

                        # 反向传播（Accelerate 会处理混合精度与梯度累积缩放）
                        accelerator.backward(loss)
                        # 仅在梯度真正同步的那一步执行梯度裁剪
                        if accelerator.sync_gradients and train_cfg.grad_clip_norm is not None:
                            grad_norm = accelerator.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
                            grad_norm_value = self._to_float(
                                accelerator.gather(grad_norm.unsqueeze(0)).mean()
                            )
                        optimizer.step()
                        if accelerator.sync_gradients:
                            lr_scheduler.step()
                            self._apply_lr_override(
                                optimizer,
                                self._resolve_lr_override(optimizer, lr_scheduler, lr_override),
                            )
                        optimizer.zero_grad(set_to_none=True)

                    # sync_gradients 为 True 表示完成了一个完整优化步
                    if accelerator.sync_gradients:
                        real_steps += 1
                        epoch_steps += 1
                        train_loss_value = self._resolve_accumulated_loss(
                            accelerator,
                            loss_sum=accum_loss_sum,
                            loss_count=accum_loss_count,
                        )
                        accum_loss_sum = None
                        accum_loss_count = 0
                        if lr_control.enabled and real_steps % log_cfg.log_freq == 0:
                            lr_override = lr_control.read(accelerator)
                            self._apply_lr_override(
                                optimizer,
                                self._resolve_lr_override(optimizer, lr_scheduler, lr_override),
                            )
                        log_window.update(
                            train_loss=train_loss_value,
                            lr=self._get_lr(optimizer, lr_scheduler),
                            grad_norm=grad_norm_value,
                        )

                        # 每隔 log_freq 步，输出一次窗口内的平均指标
                        if real_steps % log_cfg.log_freq == 0:
                            self._flush_log_window(
                                accelerator=accelerator,
                                model=model,
                                epoch=epoch,
                                real_steps=real_steps,
                                micro_steps=micro_steps,
                                epoch_steps=epoch_steps,
                                num_training_steps=num_training_steps,
                                warmup_steps=warmup_steps,
                                log_window=log_window,
                                eff_batch_size=eff_batch_size,
                            )

                        # 按真实优化步保存 checkpoint，保持 rolling+latest 两份（覆盖写）
                        if real_steps % train_cfg.checkpoint_freq_steps == 0:
                            save_step_adapter_checkpoints(
                                accelerator=accelerator,
                                model=model,
                                run_output_dir=run_output_dir,
                                real_step=real_steps,
                            )

                    # 达到单 epoch 最大步数上限则提前结束本 epoch（常用于快速调试）
                    if (
                        train_cfg.max_steps_per_epoch is not None
                        and epoch_steps >= train_cfg.max_steps_per_epoch
                    ):
                        accelerator.print(
                            f"[train] reached max_steps_per_epoch={train_cfg.max_steps_per_epoch}; "
                            f"ending epoch {epoch}"
                        )
                        break

            # 训练结束：刷新最后一段未输出的日志
            self._flush_log_window(
                accelerator=accelerator,
                model=model,
                epoch=train_cfg.epochs - 1,
                real_steps=real_steps,
                micro_steps=micro_steps,
                epoch_steps=epoch_steps,
                num_training_steps=num_training_steps,
                warmup_steps=warmup_steps,
                log_window=log_window,
                eff_batch_size=eff_batch_size,
            )

            # 即使最后一个 real_step 不是保存间隔，也更新 latest/rolling 到当前状态。
            if real_steps > 0 and real_steps % train_cfg.checkpoint_freq_steps != 0:
                save_step_adapter_checkpoints(
                    accelerator=accelerator,
                    model=model,
                    run_output_dir=run_output_dir,
                    real_step=real_steps,
                )

            # 所有进程必须共同进入 export_final_adapter 内部的 barrier。
            final_adapter_dir = export_final_adapter(accelerator, model, run_output_dir)
            # 仅主进程尝试打包成可部署 bundle。
            if accelerator.is_main_process:
                self._maybe_export_deploy_bundle(
                    accelerator=accelerator,
                    final_adapter_dir=final_adapter_dir,
                )
        finally:
            # 无论成功与否，确保 TensorBoard 追踪器正确收尾
            if log_cfg.use_tensorboard:
                accelerator.end_training()

    def _maybe_export_deploy_bundle(
        self,
        *,
        accelerator: Accelerator,
        final_adapter_dir,
    ) -> None:
        """尝试导出部署包；失败不影响训练结果，仅打印告警。"""
        try:
            bundle_dir = export_deploy_bundle(
                self.config,
                adapter_dir=final_adapter_dir,
                output_dir=self.config.logging.run_export_dir,
                overwrite=True,
            )
        except Exception as exc:  # pragma: no cover
            accelerator.print(f"[train] warning: failed to export deploy bundle: {exc}")
            return
        accelerator.print(f"[train] exported deploy bundle to: {bundle_dir}")

    def _flush_log_window(
        self,
        *,
        accelerator: Accelerator,
        model: torch.nn.Module,
        epoch: int,
        real_steps: int,
        micro_steps: int,
        epoch_steps: int,
        num_training_steps: int,
        warmup_steps: int,
        log_window: _LogWindow,
        eff_batch_size: int = 1,
    ) -> None:
        """输出并清空日志窗口：写入 TensorBoard，并由主进程打印到控制台。"""
        # 窗口为空或尚未走过任何步则跳过
        if log_window.count == 0 or real_steps == 0:
            return
        payload = log_window.as_dict()
        payload.update(self._tactile_gate_metrics(accelerator, model))
        step_fields = self._step_log_fields(
            micro_steps=micro_steps,
            real_steps=real_steps,
            epoch_steps=epoch_steps,
            num_training_steps=num_training_steps,
            warmup_steps=warmup_steps,
            eff_batch_size=eff_batch_size,
        )
        if self.config.logging.use_tensorboard:
            accelerator.log(
                {
                    **payload,
                    **{key: float(value) for key, value in step_fields.items()},
                },
                step=real_steps,
            )
        if accelerator.is_main_process:
            self._log_train_step(
                accelerator=accelerator,
                epoch=epoch,
                step_fields=step_fields,
                metrics=payload,
                window_size=log_window.count,
            )
        log_window.reset()

    @staticmethod
    def _tactile_gate_metrics(accelerator: Accelerator, model: torch.nn.Module) -> dict[str, float]:
        """Return the latest raw/effective tactile gate values for logging."""
        unwrapped_model = accelerator.unwrap_model(model)
        for module in unwrapped_model.modules():
            tactile_encoder = getattr(module, "tactile_encoder", None)
            gate = getattr(tactile_encoder, "gate", None)
            if isinstance(gate, torch.Tensor):
                raw_gate = gate.detach().float()
                if raw_gate.numel() != 1:
                    return {}
                if accelerator.num_processes > 1:
                    raw_gate = accelerator.gather(raw_gate.reshape(1)).mean()
                raw_value = float(raw_gate.item())
                return {
                    "tactile_gate": raw_value,
                    "tactile_gate_scale": float(torch.tanh(raw_gate).item()),
                }
        return {}

    @staticmethod
    def _step_log_fields(
        *,
        micro_steps: int,
        real_steps: int,
        epoch_steps: int,
        num_training_steps: int,
        warmup_steps: int,
        eff_batch_size: int = 1,
    ) -> dict[str, int]:
        return {
            "micro_step": micro_steps,
            "real_step": real_steps,
            "epoch_steps": epoch_steps,
            "num_training_steps": num_training_steps,
            "warmup_steps": warmup_steps,
            "eff_batch_size": eff_batch_size,
        }

    @staticmethod
    def _get_lr(optimizer: torch.optim.Optimizer, lr_scheduler: Any) -> float:
        """获取当前学习率：优先从调度器读取，否则回退到优化器参数组。"""
        return float(optimizer.param_groups[0]["lr"])

    @staticmethod
    def _apply_lr_override(optimizer: torch.optim.Optimizer, lr: float | None) -> None:
        if lr is None:
            return
        for group in optimizer.param_groups:
            group["lr"] = lr

    @staticmethod
    def _resolve_lr_override(
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        override: _LearningRateOverride | None,
    ) -> float | None:
        if override is None:
            return None
        if override.lr is not None:
            return override.lr
        if override.cosine_k is None:
            return None
        return Pi05LoraTrainer._get_scheduler_lr(optimizer, lr_scheduler) * override.cosine_k

    @staticmethod
    def _get_scheduler_lr(optimizer: torch.optim.Optimizer, lr_scheduler: Any) -> float:
        if lr_scheduler is not None and hasattr(lr_scheduler, "get_last_lr"):
            last_lr = lr_scheduler.get_last_lr()
            if last_lr:
                return float(last_lr[0])
        return float(optimizer.param_groups[0]["lr"])

    @staticmethod
    def _resolve_loss_value(
        loss: torch.Tensor,
        loss_dict: dict[str, Any],
        accelerator: Accelerator | None = None,
    ) -> float:
        """多卡时 gather 全局平均 loss，单卡回退到本地标量。"""
        if accelerator is not None and accelerator.num_processes > 1:
            gathered = accelerator.gather(loss.detach().unsqueeze(0))
            return float(gathered.mean().item())
        return Pi05LoraTrainer._to_float(loss_dict.get("loss", loss))

    @staticmethod
    def _resolve_accumulated_loss(
        accelerator: Accelerator,
        *,
        loss_sum: torch.Tensor | None,
        loss_count: int,
    ) -> float:
        """Average loss across all micro-batches and ranks in one optimizer step."""
        if loss_sum is None or loss_count <= 0:
            return 0.0
        payload = torch.stack(
            [
                loss_sum.detach().to(device=accelerator.device, dtype=torch.float32),
                torch.tensor(float(loss_count), device=accelerator.device),
            ]
        )
        if accelerator.num_processes > 1:
            # This payload contains fixed-size per-rank statistics, not samples
            # from the current dataloader batch. gather_for_metrics may truncate
            # it using the final batch's remainder.
            gathered = accelerator.gather(payload)
            gathered = gathered.reshape(accelerator.num_processes, 2)
            total_loss = gathered[:, 0].sum()
            total_count = gathered[:, 1].sum().clamp_min(1.0)
            return float((total_loss / total_count).item())
        return float((payload[0] / payload[1].clamp_min(1.0)).item())

    @staticmethod
    def _to_float(value: Any) -> float:
        """统一将张量/数值转为 Python float，便于日志记录。"""
        if isinstance(value, torch.Tensor):
            return float(value.detach().item())
        return float(value)

    @staticmethod
    def _log_train_step(
        *,
        accelerator: Accelerator,
        epoch: int,
        step_fields: dict[str, int],
        metrics: dict[str, float],
        window_size: int,
    ) -> None:
        """将窗口平均指标格式化为一行日志打印到控制台。"""
        micro_steps = step_fields["micro_step"]
        real_steps = step_fields["real_step"]
        ratio = (micro_steps / real_steps) if real_steps > 0 else 0.0
        real_progress = (
            (100.0 * real_steps / step_fields["num_training_steps"])
            if step_fields["num_training_steps"] > 0
            else 0.0
        )
        total_steps = step_fields["num_training_steps"]
        eff_bs = step_fields["eff_batch_size"]
        parts = [
            f"epoch={epoch:03d}",
            f"batch={eff_bs}",
            f"micro_step={micro_steps:07d}",
            f"real_step={real_steps}/{total_steps}",
            f"sample={real_steps * eff_bs}/{total_steps * eff_bs}",
            f"epoch_steps={step_fields['epoch_steps']:07d}",
            f"real_progress={real_progress:05.2f}%",
            f"micro_per_real={ratio:.2f}",
            f"avg_over={window_size}",
        ]
        if "train_loss" in metrics:
            parts.append(f"train_loss={metrics['train_loss']:.6f}")
        if "grad_norm" in metrics:
            parts.append(f"grad_norm={metrics['grad_norm']:.4f}")
        if "lr" in metrics:
            parts.append(f"lr={metrics['lr']:.2e}")
        if "tactile_gate" in metrics:
            parts.append(f"tactile_gate={metrics['tactile_gate']:.6f}")
        if "tactile_gate_scale" in metrics:
            parts.append(f"tactile_gate_scale={metrics['tactile_gate_scale']:.6f}")
        accelerator.print(" ".join(parts))


def train_from_config(config: ExperimentConfig) -> None:
    """对外训练入口：根据实验配置创建训练器并启动训练。"""
    Pi05LoraTrainer(config).run()
