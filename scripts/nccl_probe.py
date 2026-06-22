import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

print("devices:", jax.devices(), flush=True)
mesh = Mesh(jax.devices()[:2], ("x",))
x = jnp.ones((2, 1024 * 1024))
xs = jax.device_put(x, NamedSharding(mesh, P("x")))
print("sharded, launching all-reduce...", flush=True)
y = jax.jit(lambda a: jnp.sum(a))(xs)
y.block_until_ready()
print("all-reduce done, result =", float(y), flush=True)
