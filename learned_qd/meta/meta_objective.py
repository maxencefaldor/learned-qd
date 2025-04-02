from functools import partial

import jax
import jax.numpy as jnp
from evosax import FitnessShaper

reduce_ops = {
	"min": jnp.nanmin,
	"mean": jnp.nanmean,
	"max": jnp.nanmax,
	"median": partial(jnp.nanpercentile, q=50),
	"sum": jnp.nansum,
}


def zscore_tasks(fitness_tasks: jax.Array, n_devices: int = 1):
	"""Zscoring of task performances across popmembers."""
	if n_devices == 1:
		return vmap_zscore(fitness_tasks)
	else:
		return pmap_zscore(fitness_tasks)


def vmap_zscore(fitness_tasks: jax.Array) -> jax.Array:
	"""Z-score across tasks for relative performance."""

	def zscore(fit: jax.Array) -> jax.Array:
		return (fit - fit.mean()) / (fit.std() + 1e-10)

	fitness_norm = jax.vmap(jax.jit(zscore), in_axes=(1,), out_axes=1)(fitness_tasks)
	return fitness_norm


def pmap_zscore(fitness_tasks: jax.Array) -> jax.Array:
	"""Zscore different task performance across all population members."""

	def score_single_task(fit_task: jax.Array) -> jax.Array:
		def zscore(fit: jax.Array) -> jax.Array:
			all_mean = jax.lax.pmean(fit, axis_name="p").mean()
			diff = fit - all_mean
			std = jnp.sqrt(jax.lax.pmean(diff**2, axis_name="p").mean())
			return diff / (std + 1e-10)

		out = jax.pmap(zscore, axis_name="p")(fit_task)
		return out

	z_out = jax.vmap(score_single_task, in_axes=-1, out_axes=-1)(fitness_tasks)
	return z_out


norm_ops = {"zscore": zscore_tasks}


class MetaObjective:
	def __init__(
		self,
		op_pop: str = "min",
		op_task: str = "median",
		op_norm: str = "zscore",
		select_key: str = "fitness",
		n_devices: int | None = None,
	):
		"""Construction of meta-objective."""
		assert op_pop in ["min", "max", "median", "mean", "sum"]
		assert op_task in ["median", "mean"]
		assert op_norm in ["zscore"]
		self.op_pop = reduce_ops[op_pop]
		self.op_tasks = reduce_ops[op_task]
		self.op_norm = norm_ops[op_norm]
		self.fit_shaper = FitnessShaper()
		self.select_key = select_key

		if n_devices is None:
			self.n_devices = jax.local_device_count()
		else:
			self.n_devices = n_devices

	@partial(jax.jit, static_argnames=("self",))
	def apply(self, x: jax.Array, metrics: jax.Array) -> jax.Array:
		"""Shape meta-fitness objective."""
		objective_norm = self.select_and_norm(metrics)
		meta_fitness = self.fit_shaper.apply(x, objective_norm)
		return meta_fitness

	def select_and_norm(self, metrics: jax.Array) -> jax.Array:
		"""Shape meta-fitness objective."""
		objective = self.select(metrics[self.select_key])
		objective_norm = self.normalize(objective)
		return objective_norm

	def select(self, metric: jax.Array) -> jax.Array:
		"""[meta_popsize, #tasks, popsize] -> [meta_popsize, #tasks]."""
		# Population level aggregation
		objective = self.op_pop(metric, axis=-1)
		return objective

	def normalize(self, objective: jax.Array) -> jax.Array:
		"""[meta_popsize, #tasks] -> [meta_popsize,]"""
		# 1. Normalize performance across tasks
		fitness_norm = self.op_norm(objective, self.n_devices)
		# 2. Aggregate performance across tasks
		fitness = self.op_tasks(fitness_norm, axis=-1)
		return fitness.reshape(-1)
