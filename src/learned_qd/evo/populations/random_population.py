"""Random population module."""

from typing import Self

import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, Fitness, Genotype, RNGKey

from learned_qd.evo.populations.population import Population


class RandomPopulation(Population):
	"""Population.

	Args:
		genotype: Genotype of the individuals in the population.
		fitness: Fitness of the individuals in the population.
		descriptor: Descriptor of the individuals in the population.

	"""

	genotype: Genotype
	fitness: Fitness
	descriptor: Descriptor

	@classmethod
	def init(
		cls,
		genotype: Genotype,
		key: RNGKey,
		max_size: int,
		descriptor_size: int,
	) -> Self:
		genotype = jax.tree.map(
			lambda x: jnp.full((max_size,) + x.shape, fill_value=jnp.nan),
			genotype,
		)
		fitness = jnp.full((max_size,), fill_value=-jnp.inf)
		descriptor = jnp.full((max_size, descriptor_size), fill_value=jnp.nan)

		population = RandomPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
		)

		return population

	@jax.jit
	def commit(self, genotype: Genotype, fitness: Fitness, descriptor: Descriptor) -> Self:
		key = jax.random.key(0)
		key = jax.random.fold_in(key, fitness[0])

		# Concatenate
		genotype = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.genotype,
			genotype,
		)
		fitness = jnp.concatenate([self.fitness, fitness], axis=0)
		descriptor = jnp.concatenate([self.descriptor, descriptor], axis=0)

		# Sort randomly
		indices = jax.random.permutation(key, fitness.shape[0])
		indices = indices[: self.max_size]

		# Keep best
		genotype = jax.tree.map(lambda x: x[indices], genotype)
		fitness = fitness[indices]
		descriptor = descriptor[indices]

		return RandomPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
		)
