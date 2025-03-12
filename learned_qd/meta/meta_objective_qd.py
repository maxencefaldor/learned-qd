from functools import partial
from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from evosax import FitnessShaper
from qdax.custom_types import (
	Centroid,
	Descriptor,
	Fitness,
	RNGKey,
)

from learned_qd.evo.populations.grid_population import get_centroid_indices, get_centroids, segment_argmax

reduce_ops = {
	"min": jnp.nanmin,
	"mean": jnp.nanmean,
	"max": jnp.nanmax,
	"median": partial(jnp.nanpercentile, q=50),
	"sum": jnp.nansum,
}


def zscore(fitness_tasks: jax.Array):
	"""Zscoring of task performances across popmembers."""
	return jax.nn.standardize(fitness_tasks, axis=-2, epsilon=1e-10)


norm_ops = {"zscore": zscore}


class MetaObjective:
	def __init__(
		self,
		op_task: str = "median",
		op_norm: str = "zscore",
		select_key: list[str] = ["fitness", "descriptor"],
		objective: str = "qd_grid",
		descriptor_size: int = 2,
		population_size: int = 128,
		n_devices: int | None = None,
	):
		"""Construction of meta-objective."""
		assert op_task in ["median", "mean"]
		assert op_norm in ["zscore"]
		self.op_tasks = reduce_ops[op_task]
		self.op_norm = norm_ops[op_norm]
		self.fit_shaper = FitnessShaper()

		if n_devices is None:
			self.n_devices = jax.local_device_count()
		else:
			self.n_devices = n_devices

		if objective == "qd_grid":
			self.grid = Grid.init(
				key=jax.random.key(0),
				max_size=2 * population_size,
				descriptor_size=descriptor_size,
				descriptor_min=0,
				descriptor_max=1,
				num_init_cvt_samples=100_000,
			)
			self.objective = self.qd_grid
		elif objective == "qd_mean":
			self.objective = self.qd_mean
		elif objective == "qd_max":
			self.objective = self.qd_max
		elif objective == "qd_sum":
			self.objective = self.qd_sum
		elif objective == "qd_dns":
			self.objective = self.qd_dns
		else:
			raise ValueError

	@partial(jax.jit, static_argnames=("self",))
	def apply(self, x: jax.Array, metrics: jax.Array) -> jax.Array:
		"""Shape meta-fitness objective."""
		objective_norm = self.select_and_norm(metrics)
		meta_fitness = self.fit_shaper.apply(x, objective_norm)
		return meta_fitness

	def select_and_norm(self, metrics: jax.Array) -> jax.Array:
		"""Shape meta-fitness objective."""
		objective = self.select(metrics)
		objective_norm = self.normalize(objective)
		return objective_norm

	def select(self, metrics: jax.Array) -> jax.Array:
		"""[meta_popsize, #tasks, popsize] -> [meta_popsize, #tasks]."""
		if self.n_devices > 1:
			metrics = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), metrics)

		objective = self.objective(metrics)
		return objective

	def normalize(self, objective: jax.Array) -> jax.Array:
		"""[meta_popsize, #tasks] -> [meta_popsize,]"""
		# 1. Normalize performance across tasks
		fitness_norm = self.op_norm(objective)
		# 2. Aggregate performance across tasks
		fitness = self.op_tasks(fitness_norm, axis=-1)
		return fitness

	def qd_grid(self, metrics):
		fitness_min = jnp.expand_dims(jnp.min(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_norm = metrics["fitness"] - fitness_min

		descriptor_min = jnp.expand_dims(jnp.min(metrics["descriptor"], axis=(0, 2)), axis=1)
		descriptor_max = jnp.expand_dims(jnp.max(metrics["descriptor"], axis=(0, 2)), axis=1)
		descriptor_norm = (metrics["descriptor"] - descriptor_min) / (descriptor_max - descriptor_min)

		grid = jax.vmap(jax.vmap(self.grid.commit))(fitness_norm, descriptor_norm)
		return jnp.sum(grid.fitness, axis=-1, where=grid.fitness != -jnp.inf)

	def qd_mean(self, metrics):
		fitness_min = jnp.expand_dims(jnp.min(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_max = jnp.expand_dims(jnp.max(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_norm = (metrics["fitness"] - fitness_min) / (fitness_max - fitness_min)

		novelty_min = jnp.expand_dims(jnp.min(metrics["novelty"], axis=(0, 2)), axis=1)
		novelty_max = jnp.expand_dims(jnp.max(metrics["novelty"], axis=(0, 2)), axis=1)
		novelty_norm = (metrics["novelty"] - novelty_min) / (novelty_max - novelty_min)

		# Clip to ensure [0,1] bounds
		fitness_norm = jnp.clip(fitness_norm, 0, 1)
		novelty_norm = jnp.clip(novelty_norm, 0, 1)

		return jnp.mean(fitness_norm, axis=-1) * jnp.mean(novelty_norm, axis=-1)

	def qd_max(self, metrics):
		fitness_min = jnp.expand_dims(jnp.min(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_max = jnp.expand_dims(jnp.max(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_norm = (metrics["fitness"] - fitness_min) / (fitness_max - fitness_min)

		novelty_min = jnp.expand_dims(jnp.min(metrics["novelty"], axis=(0, 2)), axis=1)
		novelty_max = jnp.expand_dims(jnp.max(metrics["novelty"], axis=(0, 2)), axis=1)
		novelty_norm = (metrics["novelty"] - novelty_min) / (novelty_max - novelty_min)

		# Clip to ensure [0,1] bounds
		fitness_norm = jnp.clip(fitness_norm, 0, 1)
		novelty_norm = jnp.clip(novelty_norm, 0, 1)

		return jnp.max(fitness_norm, axis=-1) * jnp.mean(novelty_norm, axis=-1)

	def qd_sum(self, metrics):
		fitness_min = jnp.expand_dims(jnp.min(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_max = jnp.expand_dims(jnp.max(metrics["fitness"], axis=(0, 2)), axis=1)
		fitness_norm = (metrics["fitness"] - fitness_min) / (fitness_max - fitness_min)

		novelty_min = jnp.expand_dims(jnp.min(metrics["novelty"], axis=(0, 2)), axis=1)
		novelty_max = jnp.expand_dims(jnp.max(metrics["novelty"], axis=(0, 2)), axis=1)
		novelty_norm = (metrics["novelty"] - novelty_min) / (novelty_max - novelty_min)

		# Clip to ensure [0,1] bounds
		fitness_norm = jnp.clip(fitness_norm, 0, 1)
		novelty_norm = jnp.clip(novelty_norm, 0, 1)

		return jnp.mean(fitness_norm, axis=-1) + jnp.mean(novelty_norm, axis=-1)

	def qd_dns(self, metrics):
		return jnp.max(metrics["fitness"], axis=-1) / jnp.mean(metrics["dominated_novelty"], axis=-1)


class Grid(flax.struct.PyTreeNode):
	"""
	Grid.

	Args:
		fitness: Fitness of the individuals in the population.
		descriptor: Descriptor of the individuals in the population.
		centroid: Centroids of the centroidal Voronoi tesselation.
	"""

	fitness: Fitness
	descriptor: Descriptor
	centroid: Centroid

	@property
	def max_size(self) -> int:
		return self.fitness.shape[0]

	@classmethod
	def init(
		cls,
		key: RNGKey,
		max_size: int,
		descriptor_size: int,
		descriptor_min: float | list[float],
		descriptor_max: float | list[float],
		num_init_cvt_samples: int,
	) -> Self:
		fitness = jnp.full((max_size,), fill_value=-jnp.inf)
		descriptor = jnp.full((max_size, descriptor_size), fill_value=jnp.nan)
		centroid = get_centroids(
			num_centroids=max_size,
			descriptor_size=descriptor_size,
			descriptor_min=descriptor_min,
			descriptor_max=descriptor_max,
			num_init_cvt_samples=num_init_cvt_samples,
			key=key,
		)

		population = Grid(
			fitness=fitness,
			descriptor=descriptor,
			centroid=centroid,
		)

		return population

	@jax.jit
	def commit(self, fitness: Fitness, descriptor: Descriptor) -> Self:
		# Concatenate
		fitness = jnp.concatenate([self.fitness, fitness], axis=0)
		descriptor = jnp.concatenate([self.descriptor, descriptor], axis=0)

		# Get centroid assignments
		centroid_indices = get_centroid_indices(descriptor, self.centroid)

		# Compute meta-fitness based on centroid assignments
		num_centroids = self.centroid.shape[0]
		best_index_per_centroid = segment_argmax(fitness, centroid_indices, num_centroids)
		centroid_assigned = jnp.isin(jnp.arange(num_centroids), centroid_indices)
		best_index_per_centroid = jnp.where(
			centroid_assigned,
			best_index_per_centroid,
			fitness.shape[0],  # if centroid not used, put the best index out of bounds
		)

		best_index = jnp.zeros_like(fitness, dtype=bool).at[best_index_per_centroid].set(True)
		meta_fitness = jnp.where(best_index, fitness, -jnp.inf)

		# Sort by meta-fitness
		indices = jnp.argsort(meta_fitness, descending=True)
		indices = indices[: self.max_size]

		# Keep best
		fitness = meta_fitness[indices]
		descriptor = descriptor[indices]

		return Grid(
			fitness=fitness,
			descriptor=descriptor,
			centroid=self.centroid,
		)
