"""Extra functions"""

import jax
import jax.numpy as jnp


def exponential(
	x: jax.Array, x_opt: jax.Array, r: jax.Array, q: jax.Array, num_dims: int
) -> jax.Array:
	max_num_dims = x.shape[0]
	mask = jnp.arange(max_num_dims) < num_dims

	z = jnp.matmul(r, x - x_opt)

	return 1 - jnp.exp(-0.5 * jnp.sum(z**2, where=mask)), jnp.array(0.0)


def ackley(x: jax.Array, x_opt: jax.Array, r: jax.Array, q: jax.Array, num_dims: int) -> jax.Array:
	max_num_dims = x.shape[0]
	mask = jnp.arange(max_num_dims) < num_dims

	z = jnp.matmul(r, x - x_opt)

	a, b, c = 20, 0.2, 2 * jnp.pi
	out_1 = jnp.exp(-b * jnp.sqrt(jnp.sum(z**2, where=mask) / num_dims))
	out_2 = jnp.exp(jnp.sum(jnp.cos(c * z), where=mask) / num_dims)
	return -a * out_1 - out_2 + a + jnp.exp(1), jnp.array(0.0)


def dixon_price(
	x: jax.Array, x_opt: jax.Array, r: jax.Array, q: jax.Array, num_dims: int
) -> jax.Array:
	max_num_dims = x.shape[0]
	mask = jnp.arange(max_num_dims) < num_dims

	z = jnp.matmul(r, x - x_opt)

	out_1 = (z[0] - 1) ** 2
	out_2 = jnp.sum(
		mask[1:] * jnp.arange(2, max_num_dims + 1) * (2 * z[1:] ** 2 - z[: max_num_dims - 1]) ** 2
	)
	return out_1 + out_2, jnp.array(0.0)


def griewank(
	x: jax.Array, x_opt: jax.Array, r: jax.Array, q: jax.Array, num_dims: int
) -> jax.Array:
	max_num_dims = x.shape[0]
	mask = jnp.arange(max_num_dims) < num_dims

	z = jnp.matmul(r, x - x_opt)

	out_1 = 1 + jnp.sum(z**2, where=mask) / 4000
	out_2 = jnp.prod(jnp.cos(z / jnp.sqrt(jnp.arange(1, max_num_dims + 1))), where=mask)
	return out_1 - out_2, jnp.array(0.0)


def salomon(x: jax.Array, x_opt: jax.Array, r: jax.Array, q: jax.Array, num_dims: int) -> jax.Array:
	max_num_dims = x.shape[0]
	mask = jnp.arange(max_num_dims) < num_dims

	z = jnp.matmul(r, x - x_opt)

	return 1 - jnp.cos(2 * jnp.pi * jnp.linalg.norm(z * mask)) + 0.1 * jnp.linalg.norm(
		z * mask
	), jnp.array(0.0)


def levy(x: jax.Array, x_opt: jax.Array, r: jax.Array, q: jax.Array, num_dims: int) -> jax.Array:
	max_num_dims = x.shape[0]

	z = jnp.matmul(r, x - x_opt)

	w = 1 + (z - 1) / 4

	out_1 = jnp.sin(jnp.pi * w[0]) ** 2
	out_2 = jnp.sum(
		(w - 1) ** 2 * (1 + 10 * jnp.sin(jnp.pi * w + 1) ** 2),
		where=jnp.arange(max_num_dims) < num_dims - 1,
	)
	out_3 = jnp.sum(
		(w - 1) ** 2 * (1 + jnp.sin(2 * jnp.pi * w) ** 2),
		where=jnp.arange(max_num_dims) == num_dims - 1,
	)
	return out_1 + out_2 + out_3, jnp.array(0.0)


extra_fns = {
	"exponential": exponential,
	"ackley": ackley,
	"dixon_price": dixon_price,
	"griewank": griewank,
	"salomon": salomon,
	"levy": levy,
}
