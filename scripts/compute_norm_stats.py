"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class ConstantDimOverride:
    """Forces a specific state/action dim to a known constant value when its measured q99-q01 span is small.

    This replaces manually patching norm_stats.json after the fact (see git history /
    scripts/patch_state_norm_stats.py): if the dim is already near a fixed target value in the raw data
    (e.g. a hand that is essentially always commanded fully-closed at ~1000), we snap q01=q99=target so
    that transforms.py's near-constant-dim handling (span < 0.005) kicks in and always outputs `target`
    on unnormalize, regardless of small sensor/actuator noise around it.
    """

    key: str  # "state" or "actions"
    dim: int
    threshold: float  # if measured (q99 - q01) < threshold, snap to target
    target: float


# left_hand_qpos0 (state[12]) / left_hand_cmd_pos0 (actions[12]): raw readout hovers in ~[950, 1000]
# (span usually ~2), i.e. effectively always fully-closed. Snap to an exact constant 1000 whenever the
# measured span is < 50, so this is robust even if the actual sensor noise for a given dataset is a bit
# larger than what we've observed so far.
CONSTANT_DIM_OVERRIDES = [
    ConstantDimOverride(key="state", dim=12, threshold=50.0, target=1000.0),
    ConstantDimOverride(key="actions", dim=12, threshold=50.0, target=1000.0),
]


def _apply_constant_dim_overrides(norm_stats: dict, overrides: list[ConstantDimOverride]) -> None:
    """Mutates `norm_stats[key].q01/q99/mean/std[dim]` in place for each override whose span < threshold."""
    for override in overrides:
        stats = norm_stats.get(override.key)
        if stats is None or stats.q01 is None or stats.q99 is None:
            continue
        dim = override.dim
        if dim >= len(stats.q01):
            continue
        span = float(stats.q99[dim] - stats.q01[dim])
        if span < override.threshold:
            print(
                f"[compute_norm_stats] {override.key}[{dim}]: measured span={span:.6f} < "
                f"threshold={override.threshold} -> snapping q01=q99=mean=std0 to constant {override.target}"
            )
            stats.q01[dim] = override.target
            stats.q99[dim] = override.target
            stats.mean[dim] = override.target
            stats.std[dim] = 0.0
        else:
            print(
                f"[compute_norm_stats] {override.key}[{dim}]: measured span={span:.6f} >= "
                f"threshold={override.threshold} -> left as real (non-constant) data, no override applied"
            )


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    _apply_constant_dim_overrides(norm_stats, CONSTANT_DIM_OVERRIDES)

    data_factory = config.data
    assets_dir = pathlib.Path(data_factory.assets.assets_dir or config.assets_dirs)
    asset_id = data_factory.assets.asset_id or data_config.repo_id
    output_path = assets_dir / asset_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
