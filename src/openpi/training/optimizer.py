import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        inner = optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class WarmupLinearDecaySchedule(LRScheduleConfig):
    """Linear warmup to a peak LR, then linear decay to a target LR.

    After `decay_end_step`, the schedule holds `decay_lr`.
    """

    warmup_steps: int = 200
    peak_lr: float = 8e-6
    decay_end_step: int = 10_000
    decay_lr: float = 8e-7
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        if self.decay_end_step <= self.warmup_steps:
            raise ValueError("decay_end_step must be greater than warmup_steps.")
        if self.peak_lr <= 0 or self.decay_lr < 0:
            raise ValueError("Learning rates must be non-negative, and peak_lr must be positive.")

        schedules = [
            optax.linear_schedule(
                init_value=self.peak_lr / (self.warmup_steps + 1),
                end_value=self.peak_lr,
                transition_steps=self.warmup_steps,
            ),
            optax.linear_schedule(
                init_value=self.peak_lr,
                end_value=self.decay_lr,
                transition_steps=self.decay_end_step - self.warmup_steps,
            ),
            optax.constant_schedule(self.decay_lr),
        ]
        inner = optax.join_schedules(schedules, [self.warmup_steps, self.decay_end_step])
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class StagedCosineSchedule(LRScheduleConfig):
    """Piecewise cosine LR: fixed-length stages, each decays to start / decay_factor.

    Set total_steps to auto-fit stages (e.g. 1 epoch); the last stage may be shorter.
    """

    init_lr: float = 1.5e-6
    steps_per_stage: int = 3000
    num_stages: int | None = None
    total_steps: int | None = None
    decay_factor: float = 5.0
    # Shift the schedule so training can restart from params-only while keeping the LR curve.
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        if self.num_stages is None:
            if self.total_steps is None:
                raise ValueError("StagedCosineSchedule requires num_stages or total_steps.")
            num_stages = (self.total_steps + self.steps_per_stage - 1) // self.steps_per_stage
        else:
            num_stages = self.num_stages

        schedules = []
        boundaries = []
        for stage_idx in range(num_stages):
            stage_init = self.init_lr / (self.decay_factor**stage_idx)
            if self.total_steps is not None and stage_idx == num_stages - 1:
                stage_steps = self.total_steps - stage_idx * self.steps_per_stage
            else:
                stage_steps = self.steps_per_stage
            schedules.append(
                optax.cosine_decay_schedule(
                    init_value=stage_init,
                    decay_steps=stage_steps,
                    alpha=1.0 / self.decay_factor,
                )
            )
            if stage_idx < num_stages - 1:
                boundaries.append((stage_idx + 1) * self.steps_per_stage)
        inner = optax.join_schedules(schedules, boundaries)
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class GxdV5WarmCosineLinearSchedule(LRScheduleConfig):
    """V5 finetune LR: warmup→peak, cosine stages every 3k, 10k→1e-8, linear to 0.

    Breakpoints (step → LR):
      warmup_steps: peak_lr
      3000: first_stage_end_lr
      6000: - lr_step_down
      9000: - 2*lr_step_down
      10000: milestone_10k_lr
      total_steps: 0
    """

    peak_lr: float = 1e-5
    first_stage_end_lr: float = 1e-6
    lr_step_down: float = 3e-7
    milestone_10k_lr: float = 1e-8
    steps_per_stage: int = 3000
    warmup_steps: int = 500
    cosine_end_step: int = 10_000
    total_steps: int = 30_000
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        if self.cosine_end_step % self.steps_per_stage != 0:
            # Last cosine segment before 10k may be shorter (e.g. 9k→10k).
            pass
        if self.total_steps <= self.cosine_end_step:
            raise ValueError("total_steps must be greater than cosine_end_step.")

        # Stage 0: warmup to peak, then cosine to first_stage_end_lr within first 3k steps.
        decay_steps = self.steps_per_stage - self.warmup_steps
        schedules: list[optax.Schedule] = [
            optax.warmup_cosine_decay_schedule(
                init_value=self.peak_lr / (self.warmup_steps + 1),
                peak_value=self.peak_lr,
                warmup_steps=self.warmup_steps,
                decay_steps=decay_steps,
                end_value=self.first_stage_end_lr,
            )
        ]
        boundaries: list[int] = [self.steps_per_stage]

        # Subsequent 3k cosine stages, each ending lr lower by lr_step_down.
        lr = self.first_stage_end_lr
        step = self.steps_per_stage
        while step < self.cosine_end_step:
            stage_steps = min(self.steps_per_stage, self.cosine_end_step - step)
            end_lr = lr - self.lr_step_down
            if step + stage_steps >= self.cosine_end_step:
                end_lr = self.milestone_10k_lr
            alpha = end_lr / lr if lr > 0 else 0.0
            schedules.append(
                optax.cosine_decay_schedule(
                    init_value=lr,
                    decay_steps=stage_steps,
                    alpha=alpha,
                )
            )
            boundaries.append(step + stage_steps)
            lr = end_lr
            step += stage_steps

        # Linear decay from milestone_10k_lr to 0 through total_steps.
        linear_steps = self.total_steps - self.cosine_end_step
        schedules.append(
            optax.linear_schedule(
                init_value=self.milestone_10k_lr,
                end_value=0.0,
                transition_steps=linear_steps,
            )
        )

        inner = optax.join_schedules(schedules, boundaries)
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class GxdMilestoneSchedule(LRScheduleConfig):
    """Piecewise LR: plateau, two linear decays, then linear to zero.

    Default (30k total):
      0 ~ 3k:   hold plateau_lr (5e-6)
      3k ~ 6k:  linear → lr_at_6k (1e-6)   over decay_steps_to_1e6 (3k)
      6k ~ 12k: linear → lr_at_12k (1e-7)  over decay_steps_to_1e7 (6k)
      12k ~ end: linear → 0
    """

    plateau_lr: float = 5e-6
    lr_at_6k: float = 1e-6
    lr_at_12k: float = 1e-7
    plateau_end: int = 3_000
    decay_steps_to_1e6: int = 3_000
    decay_steps_to_1e7: int = 6_000
    total_steps: int = 30_000
    schedule_offset: int = 0

    @property
    def milestone_6k(self) -> int:
        return self.plateau_end + self.decay_steps_to_1e6

    @property
    def milestone_12k(self) -> int:
        return self.milestone_6k + self.decay_steps_to_1e7

    def create(self) -> optax.Schedule:
        m6 = self.milestone_6k
        m12 = self.milestone_12k
        if not (0 < self.plateau_end < m6 < m12 < self.total_steps):
            raise ValueError(
                f"Milestones must satisfy 0 < plateau_end({self.plateau_end}) "
                f"< 6k({m6}) < 12k({m12}) < total_steps({self.total_steps})."
            )

        schedules: list[optax.Schedule] = [
            optax.constant_schedule(self.plateau_lr),
            optax.linear_schedule(
                init_value=self.plateau_lr,
                end_value=self.lr_at_6k,
                transition_steps=self.decay_steps_to_1e6,
            ),
            optax.linear_schedule(
                init_value=self.lr_at_6k,
                end_value=self.lr_at_12k,
                transition_steps=self.decay_steps_to_1e7,
            ),
            optax.linear_schedule(
                init_value=self.lr_at_12k,
                end_value=0.0,
                transition_steps=self.total_steps - m12,
            ),
        ]
        boundaries = [self.plateau_end, m6, m12]
        inner = optax.join_schedules(schedules, boundaries)
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class GxdStepHoldSchedule(LRScheduleConfig):
    """Piecewise constant LR: hold each level for a fixed number of steps."""

    learning_rates: tuple[float, ...]
    hold_steps: tuple[int, ...]
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        if len(self.learning_rates) != len(self.hold_steps):
            raise ValueError("learning_rates and hold_steps must have the same length.")
        if any(steps <= 0 for steps in self.hold_steps):
            raise ValueError("Each hold_steps entry must be positive.")

        schedules = [optax.constant_schedule(lr) for lr in self.learning_rates]
        boundaries: list[int] = []
        step = 0
        for hold in self.hold_steps[:-1]:
            step += hold
            boundaries.append(step)
        inner = optax.join_schedules(schedules, boundaries)
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class GxdResumeWarmupStepHoldSchedule(LRScheduleConfig):
    """Step-hold LR with a short warmup after restoring from an existing checkpoint.

    The underlying step-hold schedule still uses the absolute training step, so stage
    boundaries remain unchanged. During [warmup_start_step, warmup_start_step + warmup_steps),
    LR is linearly ramped from warmup_init_lr to the LR that the base schedule would use.
    """

    learning_rates: tuple[float, ...]
    hold_steps: tuple[int, ...]
    warmup_start_step: int
    warmup_steps: int
    warmup_init_lr: float = 0.0
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        if self.warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        if self.warmup_start_step < 0:
            raise ValueError("warmup_start_step must be non-negative.")

        base = GxdStepHoldSchedule(
            learning_rates=self.learning_rates,
            hold_steps=self.hold_steps,
            schedule_offset=0,
        ).create()

        def schedule(step):
            effective_step = step + self.schedule_offset
            base_lr = base(effective_step)
            warmup_end_step = self.warmup_start_step + self.warmup_steps
            warmup_progress = (effective_step - self.warmup_start_step) / self.warmup_steps
            warmup_progress = jnp.clip(warmup_progress, 0.0, 1.0)
            warmup_target_lr = base(warmup_end_step)
            warmup_lr = self.warmup_init_lr + warmup_progress * (warmup_target_lr - self.warmup_init_lr)
            in_warmup = (effective_step >= self.warmup_start_step) & (effective_step < warmup_end_step)
            return jnp.where(in_warmup, warmup_lr, base_lr)

        return schedule


@dataclasses.dataclass(frozen=True)
class GxdPiecewiseLinearSchedule(LRScheduleConfig):
    """Piecewise-linear LR through explicit milestone steps and values."""

    milestone_steps: tuple[int, ...]
    milestone_lrs: tuple[float, ...]
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        if len(self.milestone_steps) != len(self.milestone_lrs):
            raise ValueError("milestone_steps and milestone_lrs must have the same length.")
        if len(self.milestone_steps) < 2:
            raise ValueError("At least two milestones are required.")
        if self.milestone_steps[0] != 0:
            raise ValueError("The first milestone step must be 0.")
        if any(
            step <= prev
            for prev, step in zip(self.milestone_steps[:-1], self.milestone_steps[1:], strict=True)
        ):
            raise ValueError("milestone_steps must be strictly increasing.")
        if any(lr < 0 for lr in self.milestone_lrs):
            raise ValueError("milestone_lrs must be non-negative.")

        schedules: list[optax.Schedule] = []
        boundaries: list[int] = []
        for start_step, end_step, start_lr, end_lr in zip(
            self.milestone_steps[:-1],
            self.milestone_steps[1:],
            self.milestone_lrs[:-1],
            self.milestone_lrs[1:],
            strict=True,
        ):
            schedules.append(
                optax.linear_schedule(
                    init_value=start_lr,
                    end_value=end_lr,
                    transition_steps=end_step - start_step,
                )
            )
            boundaries.append(end_step)
        schedules.append(optax.constant_schedule(self.milestone_lrs[-1]))
        inner = optax.join_schedules(schedules, boundaries)
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class WarmLinearCosineSchedule(LRScheduleConfig):
    """Warmup -> linear decay -> cosine decay -> (optional) final linear decay.

    Phase 1 [0, warmup_steps):              linear 0 -> peak_lr
    Phase 2 [warmup_steps, linear_end_step): linear peak_lr -> linear_end_lr
    Phase 3 [linear_end_step, cosine_end_step or total_steps): cosine linear_end_lr -> cosine_end_lr
    Phase 4 [cosine_end_step, total_steps):  linear cosine_end_lr -> final_end_lr  (only if cosine_end_step is set and < total_steps)
    """

    peak_lr: float = 2e-5
    linear_end_lr: float = 1e-5
    cosine_end_lr: float = 1e-7
    warmup_steps: int = 1_000
    linear_end_step: int = 6_000
    # If None, phase 3 runs to total_steps (no phase 4). If set and < total_steps, phase 4 is a linear decay.
    cosine_end_step: int | None = None
    # Final LR at total_steps when phase 4 is enabled. Defaults to cosine_end_lr (i.e. phase 4 is a flat hold if equal).
    final_end_lr: float | None = None
    total_steps: int = 40_000
    # When continuing from a checkpoint without resume, shift the schedule so local step 0
    # uses the LR that the base schedule would assign at schedule_offset.
    schedule_offset: int = 0

    def create(self) -> optax.Schedule:
        linear_decay_steps = self.linear_end_step - self.warmup_steps
        cosine_end_step = self.cosine_end_step if self.cosine_end_step is not None else self.total_steps
        cosine_steps = cosine_end_step - self.linear_end_step
        if cosine_steps <= 0:
            raise ValueError(
                f"cosine_end_step ({cosine_end_step}) must be greater than linear_end_step ({self.linear_end_step})."
            )
        alpha = self.cosine_end_lr / self.linear_end_lr if self.linear_end_lr > 0 else 0.0

        schedules = [
            optax.linear_schedule(0.0, self.peak_lr, self.warmup_steps),
            optax.linear_schedule(self.peak_lr, self.linear_end_lr, linear_decay_steps),
            optax.cosine_decay_schedule(self.linear_end_lr, cosine_steps, alpha=alpha),
        ]
        boundaries = [self.warmup_steps, self.linear_end_step]

        if self.cosine_end_step is not None and self.cosine_end_step < self.total_steps:
            final_end_lr = self.final_end_lr if self.final_end_lr is not None else self.cosine_end_lr
            final_linear_steps = self.total_steps - self.cosine_end_step
            schedules.append(
                optax.linear_schedule(self.cosine_end_lr, final_end_lr, final_linear_steps)
            )
            boundaries.append(self.cosine_end_step)

        inner = optax.join_schedules(schedules, boundaries)
        if self.schedule_offset:
            return lambda step: inner(step + self.schedule_offset)
        return inner


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-6
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
