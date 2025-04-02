"""Novelty population module."""

from functools import partial
from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, Fitness, Genotype, RNGKey

from learned_qd.evo.populations.population import Population


class NoveltyPopulation(Population):
	"""
	Novelty population.

	Args:
		genotype: Genotype of the individuals in the population.
		fitness: Fitness of the individuals in the population.
		descriptor: Descriptor of the individuals in the population.
	"""

	k: int = flax.struct.field(pytree_node=False)

	@classmethod
	def init(
		cls,
		genotype: Genotype,
		key: RNGKey,
		max_size: int,
		descriptor_size: int,
		k: int,
	) -> Self:
		genotype = jax.tree.map(
			lambda x: jnp.full((max_size,) + x.shape, fill_value=jnp.nan),
			genotype,
		)
		fitness = jnp.full((max_size,), fill_value=-jnp.inf)
		descriptor = jnp.full((max_size, descriptor_size), fill_value=jnp.nan)

		population = NoveltyPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
			k=k,
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

		valid = fitness != -jnp.inf

		# Distance to k-nearest neighbors
		distance = jnp.linalg.norm(descriptor[:, None, :] - descriptor[None, :, :], axis=-1)
		values, indices = jax.vmap(partial(jax.lax.top_k, k=self.k))(-distance)
		distance = jnp.mean(-values, axis=-1)
		distance = jnp.where(jnp.isnan(distance), jnp.inf, distance)  # if no individual is fitter, set distance to inf
		meta_fitness = jnp.where(valid, distance, -jnp.inf)  # empty cells have distance -inf

		# Sort by meta-fitness
		indices = jnp.argsort(meta_fitness, descending=True)
		indices = indices[: self.max_size]

		# Keep best
		genotype = jax.tree.map(lambda x: x[indices], genotype)
		fitness = fitness[indices]
		descriptor = descriptor[indices]

		return NoveltyPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
			k=self.k,
		)
