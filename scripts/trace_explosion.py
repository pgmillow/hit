"""Trace explosion across available checkpoints."""
import orbax.checkpoint as ocp
import jax.numpy as jnp
import flax.traverse_util as tu
import os

CKPT_DIR = '/data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad'
BIAS_KEY = 'params/PaliGemma/img/Transformer/encoder_norm/bias/value'
EXTREME_IDX = [576, 640, 704, 768, 832, 896, 960]

# Find all available steps
steps = sorted(int(d) for d in os.listdir(CKPT_DIR) if d.isdigit())
print(f"Available steps: {steps}\n")

for step in steps:
    path = f'{CKPT_DIR}/{step}/params'
    if not os.path.exists(path):
        print(f"Step {step}: no params dir, skipping")
        continue
    try:
        p = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).restore(path)
        v = tu.flatten_dict(p, sep='/')[BIAS_KEY]
        mx = float(jnp.max(jnp.abs(v)))
        vals = [float(v[i]) for i in EXTREME_IDX]
        n_extreme = int(jnp.sum(jnp.abs(v) > 1e10))
        print(f"Step {step:5d}: max_abs={mx:.4e}  extreme_count={n_extreme}")
        for idx, val in zip(EXTREME_IDX, vals):
            print(f"           bias[{idx}] = {val:.4e}")
    except Exception as e:
        print(f"Step {step}: ERROR - {e}")
