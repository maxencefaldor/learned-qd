from functools import partial

import jax
import jax.numpy as jnp

reduce_ops = {
	"min": jnp.nanmin,
	"mean": jnp.nanmean,
	"max": jnp.nanmax,
	"median": partial(jnp.nanpercentile, q=50),
	"sum": jnp.nansum,
}


class MetaObjective:
	def __init__(self, meta_objective_fn: str = "qd_grid", op_task: str = "median"):
		"""Construction of meta-objective."""
		assert op_task in ["median", "mean"]

		self.meta_objective_fn = meta_objective_fn
		self.op_tasks = reduce_ops[op_task]

	@partial(jax.jit, static_argnames=("self",))
	def apply(self, metrics: jax.Array) -> jax.Array:
		"""Shape meta-fitness objective."""
		# metrics shape: (meta_population_size, meta_batch_size, population_size)

		# Aggregate objective across population and standardize across tasks
		objective = self.meta_objective_fn(metrics)  # (meta_population_size, meta_batch_size)

		# Aggregate objective across tasks
		objective = self.op_tasks(objective, axis=-1)  # (meta_population_size,)

		return objective


def fitness_max(metrics, epsilon=1e-8):
	fitness_max = jnp.nanmax(metrics["fitness"], axis=-1)
	fitness_max_norm = jax.nn.standardize(fitness_max, axis=0, epsilon=epsilon)

	return fitness_max_norm


def novelty_mean(metrics, epsilon=1e-8):
	novelty_mean = jnp.nanmean(metrics["novelty"], axis=-1)
	novelty_mean_norm = jax.nn.standardize(novelty_mean, axis=0, epsilon=epsilon)

	return novelty_mean_norm


def qd(metrics, alpha=0.5, epsilon=1e-8):
	fitness_mean = jnp.nanmean(metrics["fitness"], axis=-1)
	fitness_mean_norm = jax.nn.standardize(fitness_mean, axis=0, epsilon=epsilon)

	novelty_mean = jnp.nanmean(metrics["novelty"], axis=-1)
	novelty_mean_norm = jax.nn.standardize(novelty_mean, axis=0, epsilon=epsilon)

	return alpha * fitness_mean_norm + (1 - alpha) * novelty_mean_norm
