"""Find which parameter has extreme values."""
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import sys
import flax.traverse_util as traverse_util

path = sys.argv[1] if len(sys.argv) > 1 else '/data/gxdcheckpoint/dataV5_final_v3src_pad/dataV5_final_v3src_pad/12000/params'
path_train_state = sys.argv[2] if len(sys.argv) > 2 else ''

print(f"Loading checkpoint from: {path}")
checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
restored = checkpointer.restore(path)

flat_dict = traverse_util.flatten_dict(restored, sep='/')
print(f"Total param entries: {len(flat_dict)}")

for key, val in sorted(flat_dict.items()):
    mx = float(jnp.max(jnp.abs(val)))
    if mx > 10:
        print(f"  key: {key}")
        print(f"    max_abs={mx:.4e}, shape={val.shape}, dtype={val.dtype}")

# Also check if opt_state was restored in the training
if path_train_state:
    print(f"\nLoading train_state from: {path_train_state}")
    ts = checkpointer.restore(path_train_state)
    ts_flat = traverse_util.flatten_dict(ts, sep='/')
    print(f"Train state keys: {list(ts_flat.keys())[:10]}")
    if 'opt_state' in dir(ts) or hasattr(ts, 'opt_state'):
        print("Has opt_state attribute")
