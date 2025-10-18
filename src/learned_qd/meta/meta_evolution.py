from functools import partial

import jax

from learned_qd.evo.evolution import Evolution, EvolutionState
from learned_qd.evo.genetic_algorithm import GeneticAlgorithm
from learned_qd.evo.populations.learned_population import LearnedPopulation
from learned_qd.tasks import Task


class MetaEvolution(Evolution):
	def __init__(
		self,
		ga: GeneticAlgorithm,
		task: Task,
		population: LearnedPopulation,
		num_generations: int,
	):
		super().__init__(ga, task)
		self.population = population
		self.num_generations = num_generations

	def init_vmap(self, params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize evolution state for each task."""

		def init_fn(params, key):
			return super(MetaEvolution, self).init(params, key)

		population = self.population.replace(params=params)
		return jax.vmap(init_fn, in_axes=(None, 0))(population, batch_keys)

	def init_vmap2(self, batch_params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize evolution state for each params/task (Pegasus trick)."""
		evo_state, metrics = jax.vmap(self.init_vmap, in_axes=(0, None))(batch_params, batch_keys)
		return evo_state, metrics

	@partial(jax.jit, static_argnames=("self",))
	def init(self, batch_params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize meta-evolution."""
		evo_state, metrics = self.init_vmap2(batch_params, batch_keys)
		return evo_state, metrics

	def evolve_vmap(
		self, evo_state: EvolutionState, metrics
	) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve state for each task."""

		def evolve_fn(evo_state, metrics):
			return super(MetaEvolution, self).evolve(
				evo_state, metrics, num_generations=self.num_generations, all_steps=False
			)

		return jax.vmap(evolve_fn)(evo_state, metrics)

	def evolve_vmap2(
		self, evo_state: EvolutionState, metrics
	) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve state for each params/task (i.e., Pegasus trick)."""
		evo_state, metrics = jax.vmap(self.evolve_vmap)(evo_state, metrics)
		return evo_state, metrics

	@partial(jax.jit, static_argnames=("self",))
	def evolve(
		self, evo_state: EvolutionState, metrics
	) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve meta-evolution."""
		evo_state, metrics = self.evolve_vmap2(evo_state, metrics)
		return evo_state, metrics
