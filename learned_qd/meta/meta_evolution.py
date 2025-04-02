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
		n_devices: int | None = None,
	):
		super().__init__(ga, task)
		self.population = population
		self.num_generations = num_generations

		if n_devices is None:
			self.n_devices = jax.local_device_count()
		else:
			self.n_devices = n_devices

	def mc_init(self, params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize evolution state for each task."""

		def init_fn(params, key):
			return super(MetaEvolution, self).init(params, key)

		population = self.population.replace(params=params)
		return jax.vmap(init_fn, in_axes=(None, 0))(population, batch_keys)

	def vmap_mc_init(self, batch_params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize evolution state for each params/task (i.e., Pegasus trick)."""
		evo_state, metrics = jax.vmap(self.mc_init, in_axes=(0, None))(batch_params, batch_keys)
		return evo_state, metrics

	def pmap_mc_init(self, batch_params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize evolution state for each params/task with pmap (i.e., Pegasus trick)."""
		evo_state, metrics = jax.pmap(self.vmap_mc_init, in_axes=(0, None))(
			batch_params, jax.random.key_data(batch_keys)
		)
		return evo_state, metrics

	@partial(jax.jit, static_argnames=("self",))
	def init(self, batch_params: jax.Array, batch_keys: jax.Array) -> EvolutionState:
		"""Initialize meta-evolution."""
		init_fn = self.vmap_mc_init if self.n_devices == 1 else self.pmap_mc_init
		evo_state, metrics = init_fn(batch_params, batch_keys)
		return evo_state, metrics

	def mc_evolve(self, evo_state: EvolutionState, metrics) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve state for each task."""

		def evolve_fn(evo_state, metrics):
			return super(MetaEvolution, self).evolve(
				evo_state, metrics, num_generations=self.num_generations, all_steps=False
			)

		return jax.vmap(evolve_fn)(evo_state, metrics)

	def vmap_mc_evolve(self, evo_state: EvolutionState, metrics) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve state for each params/task (i.e., Pegasus trick)."""
		evo_state, metrics = jax.vmap(self.mc_evolve)(evo_state, metrics)
		return evo_state, metrics

	def pmap_mc_evolve(self, evo_state: EvolutionState, metrics) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve state for each params/task with pmap (i.e., Pegasus trick)."""
		evo_state, metrics = jax.pmap(self.vmap_mc_evolve)(evo_state, metrics)
		return evo_state, metrics

	@partial(jax.jit, static_argnames=("self",))
	def evolve(self, evo_state: EvolutionState, metrics) -> tuple[EvolutionState, dict[str, jax.Array]]:
		"""Evolve meta-evolution."""
		evolve_fn = self.vmap_mc_evolve if self.n_devices == 1 else self.pmap_mc_evolve
		evo_state, metrics = evolve_fn(evo_state, metrics)
		return evo_state, metrics
