import jax.numpy as jnp
from jax import vmap

vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

a = jnp.array([1,1,1])
b = jnp.array([2,1,0])
A = a[None]
print(A)
print(mv(A,b))