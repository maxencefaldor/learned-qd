"""Learned population module."""

from typing import Self

import flax.struct
import hydra
import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, Fitness, Genotype, Params, RNGKey

from learned_qd.evo.populations.population import Population


class LearnedPopulation(Population):
	"""Learned Quality-Diversity population.

	Args:
		genotype: Genotype of the individuals in the population.
		fitness: Fitness of the individuals in the population.
		descriptor: Descriptor of the individuals in the population.
		params: Parameters of the attention network.

	"""

	params: Params
	config: dict = flax.struct.field(pytree_node=False)

	@classmethod
	def init(
		cls,
		genotype: Genotype,
		key: RNGKey,
		max_size: int,
		descriptor_size: int,
		learned_fitness: dict,
	) -> Self:
		genotype = jax.tree.map(
			lambda x: jnp.full((max_size,) + x.shape, fill_value=jnp.nan),
			genotype,
		)
		fitness = jnp.full((max_size,), fill_value=-jnp.inf)
		descriptor = jnp.full((max_size, descriptor_size), fill_value=jnp.nan)

		config = learned_fitness
		learned_fitness = hydra.utils.instantiate(config)
		params = learned_fitness.init(key, fitness, descriptor)

		population = LearnedPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
			params=params,
			config=config,
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

		# Compute meta-fitness
		learned_fitness = hydra.utils.instantiate(self.config)
		meta_fitness = learned_fitness.apply(self.params, fitness, descriptor)

		# Sort by meta-fitness
		indices = jnp.argsort(meta_fitness, descending=True)
		indices = indices[: self.max_size]

		# Keep best
		genotype = jax.tree.map(lambda x: x[indices], genotype)
		fitness = fitness[indices]
		descriptor = descriptor[indices]

		return LearnedPopulation(
			genotype=genotype,
			fitness=fitness,
			descriptor=descriptor,
			params=self.params,
			config=self.config,
		)
