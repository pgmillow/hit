import dataclasses
import functools
import logging
import math
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.tensorboard import SummaryWriter
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def launch_tensorboard_server(logdir: str, port: int = 6006, host: str = "0.0.0.0") -> None:
    """Launch TensorBoard server in a background thread for remote access."""
    import subprocess
    import threading

    def _run():
        cmd = [
            "tensorboard",
            "--logdir", logdir,
            "--port", str(port),
            "--host", host,
            "--bind_all",
        ]
        logging.info("[stage] tensorboard: launching server on %s:%s", host, port)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logging.info("[stage] tensorboard: server started (http://localhost:%s)", port)


def init_tensorboard(config: _config.TrainConfig, *, enabled: bool = True) -> SummaryWriter | None:
    if not enabled:
        return None

    tb_subdir = getattr(config, "tensorboard_subdir", "tb")
    tb_dir = config.checkpoint_dir / tb_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    logging.info("[stage] tensorboard: writing logs to %s", tb_dir)

    # Launch TensorBoard server for remote access
    tb_port = getattr(config, "tensorboard_port", 6006)
    launch_tensorboard_server(str(tb_dir), port=tb_port)

    return SummaryWriter(log_dir=str(tb_dir))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    logging.info("[stage] init_train_state: creating optimizer")
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    if config.gradient_accumulation_steps > 1:
        logging.info(
            "[stage] init_train_state: enabling gradient accumulation steps=%s",
            config.gradient_accumulation_steps,
        )
        tx = optax.MultiSteps(tx, every_k_schedule=config.gradient_accumulation_steps)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

#    logging.info("[stage] init_train_state: evaluating model/train-state shapes")
    train_state_shape = jax.eval_shape(init, init_rng)
#    logging.info("[stage] init_train_state: creating FSDP sharding")
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
#        logging.info("[stage] init_train_state: resume=True, returning shape for checkpoint restore")
        return train_state_shape, state_sharding

#    logging.info("[stage] init_train_state: loading base checkpoint weights")
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
#    logging.info("[stage] init_train_state: base checkpoint weights loaded and validated")
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
#    logging.info("[stage] init_train_state: JIT initializing train state with loaded weights")
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    logging.info("[stage] init_train_state: train state initialized")

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    def tree_all_finite(tree):
        finite_tree = jax.tree.map(lambda x: jnp.all(jnp.isfinite(x)), tree)
        return jax.tree.reduce(jnp.logical_and, finite_tree, initializer=jnp.array(True))

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        train_image_augment = getattr(config, "train_image_augment", True)
        # Use compute_loss_with_debug so we can log per-dim loss contribution (loss_dim/00 .. loss_dim/{action_dim-1}).
        chunked_loss, debug = model.compute_loss_with_debug(
            rng, observation, actions, train=train_image_augment
        )
        mean_loss = jnp.mean(chunked_loss)
        # stop_gradient on debug so value_and_grad does not track it (saves memory; aux only).
        return mean_loss, jax.lax.stop_gradient(debug)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_debug), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )
    grad_norm = optax.global_norm(grads)
    grads_finite = jnp.isfinite(grad_norm)
    loss_finite = jnp.isfinite(loss)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    update_abs_max = jax.tree.reduce(
        jnp.maximum,
        jax.tree.map(lambda update: jnp.max(jnp.abs(update)), updates),
        initializer=jnp.array(0.0),
    )
    candidate_params = optax.apply_updates(params, updates)
    update_finite = jnp.logical_and(
        jnp.logical_and(loss_finite, grads_finite),
        jnp.logical_and(tree_all_finite(updates), tree_all_finite(candidate_params)),
    )
    updates = jax.tree.map(lambda update: jnp.where(update_finite, update, jnp.zeros_like(update)), updates)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_opt_state = jax.tree.map(lambda new, old: jnp.where(update_finite, new, old), new_opt_state, state.opt_state)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": grad_norm,
        "update_abs_max": update_abs_max,
        "param_norm": optax.global_norm(kernel_params),
        "nonfinite": 1.0 - update_finite.astype(jnp.float32),
        # Per-dim flow-matching loss contribution (only loss_action_dims entries will be non-zero).
        **{k: v for k, v in loss_debug.items() if k.startswith("loss_dim/")},
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info("[stage] startup: validating device and batch configuration")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    logging.info(
        "[stage] startup: batch_size=%s gradient_accumulation_steps=%s effective_batch_size=%s",
        config.batch_size,
        config.gradient_accumulation_steps,
        effective_batch_size,
    )

    logging.info(
        "[stage] startup: jax backend=%s device_count=%s devices=%s",
        jax.default_backend(),
        jax.device_count(),
        jax.devices(),
    )
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    logging.info("[stage] startup: creating RNG and sharding mesh")
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    logging.info("[stage] checkpoint: initializing checkpoint directory")
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    logging.info("[stage] checkpoint: ready, resuming=%s dir=%s", resuming, config.checkpoint_dir)
    # External warm-restart: restore full train_state from another experiment's checkpoint.
    external_restore_dir = getattr(config, "resume_from_checkpoint_dir", None)
    external_restore_step = getattr(config, "resume_from_checkpoint_step", None)
    external_restore = (not resuming) and external_restore_dir is not None
    if external_restore:
        logging.info(
            "[stage] checkpoint: will restore full train_state from %s step=%s into %s",
            external_restore_dir,
            external_restore_step if external_restore_step is not None else "latest",
            config.checkpoint_dir,
        )
    logging.info("[stage] wandb: initializing")
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    logging.info("[stage] wandb: initialized")
    logging.info("[stage] tensorboard: initializing")
    tensorboard_enabled = getattr(config, "tensorboard_enabled", True)
    tb_writer = init_tensorboard(config, enabled=tensorboard_enabled)
    logging.info("[stage] tensorboard: initialized")

    logging.info("[stage] data: creating data loader")
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    steps_per_epoch = None
    if data_loader.dataset_size is not None:
        steps_per_epoch = math.ceil(data_loader.dataset_size / effective_batch_size)
        logging.info(
            "[stage] data: dataset_size=%s steps_per_epoch=%s configured_num_train_steps=%s",
            data_loader.dataset_size,
            steps_per_epoch,
            config.num_train_steps,
        )
    logging.info("[stage] data: fetching first batch")
    data_iter = iter(data_loader)
    batch = next(data_iter)

    # Log images from first batch to sanity check.
    if config.wandb_enabled:
        logging.info("[stage] wandb: logging first batch camera images")
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)
        logging.info("[stage] wandb: first batch camera images logged")
    else:
        logging.info("[stage] wandb: disabled, skipping first batch camera image logging")

    logging.info("[stage] model: initializing train state")
    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming or external_restore
    )
    logging.info("[stage] model: waiting for train state readiness")
    jax.block_until_ready(train_state)

    if resuming:
        logging.info("[stage] checkpoint: restoring train state")
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        logging.info("[stage] checkpoint: train state restored")
    elif external_restore:
        logging.info("[stage] checkpoint: restoring full train state from external source")
        source_manager = _checkpoints.open_readonly_checkpoint_manager(external_restore_dir)
        train_state = _checkpoints.restore_state(
            source_manager, train_state, data_loader, step=external_restore_step
        )
        logging.info(
            "[stage] checkpoint: external train state restored (step=%s)", int(train_state.step)
        )

    logging.info("[stage] train: creating JIT train step")
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    logging.info("[stage] train: entering training loop")
    lr_schedule = config.lr_schedule.create()
    log_step_offset = getattr(config, "log_step_offset", 0)

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            reduced_info["lr"] = float(jax.device_get(lr_schedule(step)))
            global_step = step + log_step_offset
            if steps_per_epoch is not None:
                reduced_info["epoch"] = global_step / steps_per_epoch
            info_str = ", ".join(
                f"{k}={v:.3e}" if k == "lr" else f"{k}={v:.4f}"
                for k, v in reduced_info.items()
                if not k.startswith("loss_dim/")
            )
            pbar.write(f"Step {global_step}: {info_str}")
            wandb.log(reduced_info, step=global_step)
            if tb_writer is not None:
                for key, value in reduced_info.items():
                    tb_writer.add_scalar(key, value, global_step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # Name the final checkpoint after num_train_steps so stage milestones align (e.g. 9000).
            save_step = config.num_train_steps if step == config.num_train_steps - 1 else step

            # ── DEBUG: check encoder_norm.bias before/after save ──
            if config.debug_checkpoint_bias:
                _flat = traverse_util.flatten_dict(train_state.params.to_pure_dict(), sep='/')
                # Try both key formats (with and without /value suffix)
                _bias_key = None
                for _candidate in ('PaliGemma/img/Transformer/encoder_norm/bias/value',
                                   'PaliGemma/img/Transformer/encoder_norm/bias'):
                    if _candidate in _flat:
                        _bias_key = _candidate
                        break
                if _bias_key:
                    _bias = _flat[_bias_key]
                    logging.info(
                        "[DEBUG] BEFORE save: encoder_norm.bias min=%.6e max=%.6e mean=%.6e",
                        float(jnp.min(_bias)), float(jnp.max(_bias)), float(jnp.mean(_bias)),
                    )
                else:
                    logging.info("[DEBUG] BEFORE save: encoder_norm.bias key NOT FOUND, keys containing 'encoder_norm': %s",
                                 [k for k in _flat.keys() if 'encoder_norm' in k.lower()])

            logging.info("[stage] checkpoint: saving step %s", save_step)
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, save_step)
            logging.info("[stage] checkpoint: save requested for step %s", save_step)

            if config.debug_checkpoint_bias:
                # Immediately restore and compare
                _restored = _checkpoints.restore_state(
                    checkpoint_manager, train_state, data_loader, step=save_step
                )
                _flat2 = traverse_util.flatten_dict(_restored.params.to_pure_dict(), sep='/')
                if _bias_key and _bias_key in _flat2:
                    _bias2 = _flat2[_bias_key]
                    logging.info(
                        "[DEBUG] AFTER reload: encoder_norm.bias min=%.6e max=%.6e mean=%.6e",
                        float(jnp.min(_bias2)), float(jnp.max(_bias2)), float(jnp.mean(_bias2)),
                    )
                    _diff = float(jnp.max(jnp.abs(_bias2 - _bias)))
                    logging.info("[DEBUG] save/reload DIFF: max_abs_diff=%.6e", _diff)
                    if _diff > 1.0:
                        logging.warning("[DEBUG] *** DATA CORRUPTION DETECTED: save/reload diff = %.6e ***", _diff)
                else:
                    # Try finding the key in the restored state
                    _restored_keys = [k for k in _flat2.keys() if 'encoder_norm' in k.lower()]
                    logging.info("[DEBUG] AFTER reload: keys containing 'encoder_norm': %s", _restored_keys)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main(_config.cli())
