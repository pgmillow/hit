"""Compare params between pi05_base and the 12000-step checkpoint.
Loads one at a time, aggressively frees GPU memory between loads.
"""
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import sys
import gc
import flax.traverse_util as traverse_util

path_a = sys.argv[1] if len(sys.argv) > 1 else '/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params'
path_b = sys.argv[2] if len(sys.argv) > 2 else '/data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/12000/params'

checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

# ── Load pi05_base ──
print("Loading pi05_base...")
params_a = checkpointer.restore(path_a)
flat_a = traverse_util.flatten_dict(params_a, sep='/')
# extract just the info we need: key -> (shape, max_abs)
info_a = {}
for k, v in flat_a.items():
    info_a[k] = (v.shape, float(jnp.max(jnp.abs(v))))
del params_a, flat_a
gc.collect()
# Force JAX to free GPU buffers
for _ in range(3):
    gc.collect()

print(f"  pi05_base params: {len(info_a)}")

# ── Load 12000-step checkpoint ──
print("Loading 12000-step checkpoint...")
params_b = checkpointer.restore(path_b)
flat_b = traverse_util.flatten_dict(params_b, sep='/')
info_b = {}
for k, v in flat_b.items():
    info_b[k] = (v.shape, float(jnp.max(jnp.abs(v))))
del params_b, flat_b
gc.collect()
for _ in range(3):
    gc.collect()

print(f"  checkpoint params: {len(info_b)}")

# ── Compare using extracted info (no GPU memory needed) ──
common_keys = set(info_a.keys()) & set(info_b.keys())
only_a = set(info_a.keys()) - set(info_b.keys())
only_b = set(info_b.keys()) - set(info_a.keys())
print(f"\nCommon: {len(common_keys)}, Only base: {len(only_a)}, Only ckpt: {len(only_b)}")

# Most-changed param
max_ratio = 0.0
max_ratio_key = None
for key in sorted(common_keys):
    shape_a, mx_a = info_a[key]
    shape_b, mx_b = info_b[key]
    if mx_a > 0:
        ratio = mx_b / mx_a
        if ratio > max_ratio:
            max_ratio = ratio
            max_ratio_key = (key, shape_b, mx_a, mx_b, ratio)

if max_ratio_key:
    key, shape, mx_a, mx_b, ratio = max_ratio_key
    print(f"\n=== MOST CHANGED PARAM ===")
    print(f"  Key:        {key}")
    print(f"  Shape:      {shape}")
    print(f"  pi05_base:  {mx_a:.4e}")
    print(f"  checkpoint: {mx_b:.4e}")
    print(f"  Ratio:      {ratio:.2e}x")

# All extreme params in checkpoint
print(f"\n=== EXTREME IN CHECKPOINT (max_abs > 10) ===")
found = 0
for key in sorted(info_b.keys()):
    shape_b, mx_b = info_b[key]
    if mx_b > 10:
        found += 1
        shape_a, mx_a = info_a.get(key, (None, -1))
        print(f"  [{found}] {key}")
        print(f"      shape={shape_b}, pi05_base={mx_a:.4e}, ckpt={mx_b:.4e}")
