"""Check distribution of extreme values in encoder_norm/bias."""
import orbax.checkpoint as ocp
import jax.numpy as jnp
import flax.traverse_util as tu

print("Loading checkpoint...")
p = ocp.Checkpointer(ocp.PyTreeCheckpointHandler()).restore(
    '/data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/12000/params'
)
flat = tu.flatten_dict(p, sep='/')
v = flat['params/PaliGemma/img/Transformer/encoder_norm/bias/value']

print(f"shape={v.shape}  dtype={v.dtype}")
print(f"min={float(jnp.min(v)):.4e}")
print(f"max={float(jnp.max(v)):.4e}")
print(f"mean={float(jnp.mean(v)):.4e}")
print(f"std={float(jnp.std(v)):.4e}")
print()
print(f"p1  ={float(jnp.percentile(v, 1)):.4e}")
print(f"p10 ={float(jnp.percentile(v, 10)):.4e}")
print(f"p50 ={float(jnp.percentile(v, 50)):.4e}")
print(f"p90 ={float(jnp.percentile(v, 90)):.4e}")
print(f"p99 ={float(jnp.percentile(v, 99)):.4e}")
print()

for threshold, label in [(1e5, "1e5"), (1e10, "1e10"), (1e15, "1e15"), (1e20, "1e20"), (1e25, "1e25")]:
    n = int(jnp.sum(jnp.abs(v) > float(threshold)))
    print(f"elements > {label}: {n}/1152")
