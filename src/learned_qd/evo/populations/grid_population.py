"""Grid population module."""

from typing import Self

import jax
import jax.numpy as jnp
from numpy.random import RandomState
from qdax.custom_types import (
	Centroid,
	Descriptor,
	Fitness,
	Genotype,
	RNGKey,
)
from sklearn.cluster import KMeans

from learned_qd.evo.populations.population import Population


class GridPopulation(Population):
	"""Grid population.

	Args:
		genotype: Genotype of the individuals in the population.
		fitness: Fitness of the individuals in the population.
		descriptor: Descriptor of the individuals in the population.
		centroid: Centroids of the centroidal Voronoi tesselation.

	"""

	centroid: Centroid

	@classmethod
	def init(
		cls,
		genotype: Genotype,
		key: RNGKey,
		max_size: int,
		descriptor_size: int,
		descriptor_min: float | list[float],
		descriptor_max: float | list[float],
		num_init_cvt_samples: int,
	) -> Self:
		genotype = jax.tree.map(
			lambda x: jnp.full((max_size,) + x.shape, fill_value=jnp.nan),
			genotype,
		)
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

		population = GridPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
			centroid=centroid,
		)

		return population

	@jax.jit
	def commit(self, genotype: Genotype, fitness: Fitness, descriptor: Descriptor) -> Self:
		# Concatenate
		genotype = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.genotype,
			genotype,
		)
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
		genotype = jax.tree.map(lambda x: x[indices], genotype)
		fitness = meta_fitness[indices]
		descriptor = descriptor[indices]

		return GridPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
			centroid=self.centroid,
		)


def get_centroid_indices(descriptors: Descriptor, centroids: Centroid) -> jax.Array:
	"""Assign descriptors to their closest centroid and return the indices of the centroids.

	Args:
		descriptors: a batch of descriptors
		centroids: centroids array of shape (num_centroids, descriptor_size)

	Returns:
		For each descriptor, the index of the closest centroid.

	"""

	def _get_centroid_indices(descriptor: Descriptor) -> jax.Array:
		return jnp.argmin(jnp.linalg.norm(descriptor - centroids, axis=-1))

	indices = jax.vmap(_get_centroid_indices)(descriptors)
	return indices


def get_centroids(
	num_centroids: int,
	descriptor_size: int,
	descriptor_min: float | list[float],
	descriptor_max: float | list[float],
	num_init_cvt_samples: int,
	key: RNGKey,
) -> jax.Array:
	key_x, key_kmeans = jax.random.split(key)
	descriptor_min = jnp.array(descriptor_min)
	descriptor_max = jnp.array(descriptor_max)

	# Sample x uniformly in [0, 1]
	x = jax.random.uniform(key_x, (num_init_cvt_samples, descriptor_size))

	# Compute k-means
	kmeans = KMeans(
		init="k-means++",
		n_clusters=num_centroids,
		n_init=1,
		random_state=RandomState(jax.random.key_data(key_kmeans)),
	)
	kmeans.fit(x)
	centroid = kmeans.cluster_centers_

	# Rescale
	return descriptor_min + (descriptor_max - descriptor_min) * jnp.asarray(centroid)


def segment_argmax(data, segment_ids, num_segments):
	return jnp.argmax(
		jax.vmap(lambda i: jnp.where(i == segment_ids, data, -jnp.inf))(jnp.arange(num_segments)),
		axis=1,
	)
