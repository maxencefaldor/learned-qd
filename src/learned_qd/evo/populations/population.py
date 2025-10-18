"""Population module."""

from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, Fitness, Genotype, RNGKey


class Population(flax.struct.PyTreeNode):
	"""Population.

	Args:
		genotype: Genotype of the individuals in the population.
		fitness: Fitness of the individuals in the population.
		descriptor: Descriptor of the individuals in the population.

	"""

	genotype: Genotype
	fitness: Fitness
	descriptor: Descriptor

	@property
	def max_size(self) -> int:
		return self.fitness.shape[0]

	@property
	def size(self) -> int:
		valid = self.fitness != -jnp.inf
		return jnp.sum(valid)

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

		population = Population(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
		)

		return population

	def commit(self, genotype: Genotype, fitness: Fitness, descriptor: Descriptor) -> Self:
		# Concatenate
		genotype = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.genotype,
			genotype,
		)
		fitness = jnp.concatenate([self.fitness, fitness], axis=0)
		descriptor = jnp.concatenate([self.descriptor, descriptor], axis=0)

		# Sort by fitness
		indices = jnp.argsort(fitness, descending=True)
		indices = indices[: self.max_size]

		# Keep best
		genotype = jax.tree.map(lambda x: x[indices], genotype)
		fitness = fitness[indices]
		descriptor = descriptor[indices]

		return Population(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
		)

	def sample(self, key: RNGKey, num_samples: int) -> Genotype:
		valid = self.fitness != -jnp.inf
		p = valid / jnp.sum(valid)

		genotype = jax.tree.map(
			lambda x: jax.random.choice(key, x, shape=(num_samples,), p=p),
			self.genotype,
		)

		return genotype
