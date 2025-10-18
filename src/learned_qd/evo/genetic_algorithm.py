"""Genetic Algorithm module."""

from collections.abc import Callable
from functools import partial

import jax
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
	Descriptor,
	ExtraScores,
	Fitness,
	Genotype,
	Metrics,
	RNGKey,
)

from learned_qd.evo.populations.population import Population


class GeneticAlgorithm:
	"""Genetic Algorithm.

	Args:
		emitter: an emitter used to reproduce a batch of parents to generate offpsring
		metrics_fn: a function that takes a population and returns metrics

	"""

	def __init__(self, emitter: Emitter, metrics_fn: Callable[[Population], Metrics]) -> None:
		self._emitter = emitter
		self._metrics_fn = metrics_fn

	@partial(jax.jit, static_argnames=("self",))
	def init(
		self,
		population: Population,
		genotype: Genotype,
		fitness: Fitness,
		descriptor: Descriptor,
		extra_scores: ExtraScores,
		key: RNGKey,
	) -> tuple[Population, EmitterState]:
		# Commit individuals to the population
		population = population.commit(genotype, fitness, descriptor)

		# Initialize emitter state
		emitter_state = self._emitter.init(
			key=key,
			repertoire=population,
			genotypes=genotype,
			fitnesses=fitness,
			descriptors=descriptor,
			extra_scores=extra_scores,
		)

		# Compute metrics
		metrics = self._metrics_fn(population)

		return population, emitter_state, metrics

	def ask(
		self,
		population: Population,
		emitter_state: EmitterState,
		key: RNGKey,
	) -> tuple[Genotype, ExtraScores]:
		# Reproduce parents to generate offsprings
		genotype, extra_info = self._emitter.emit(population, emitter_state, key)
		return genotype, extra_info

	def tell(
		self,
		population: Population,
		emitter_state: EmitterState,
		genotype: Genotype,
		fitness: Fitness,
		descriptor: Descriptor,
		extra_scores: ExtraScores,
	) -> tuple[Population, EmitterState, Metrics]:
		# Commit individuals to the population
		population = population.commit(genotype, fitness, descriptor)

		# Update emitter state
		emitter_state = self._emitter.state_update(
			emitter_state=emitter_state,
			repertoire=population,
			genotypes=genotype,
			fitnesses=fitness,
			descriptors=descriptor,
			extra_scores=extra_scores,
		)

		# Compute metrics
		metrics = self._metrics_fn(population)

		return population, emitter_state, metrics
