"""Evolutionary Algorithm."""

from functools import partial

import jax
from flax.struct import dataclass

from learned_qd.evo.genetic_algorithm import EmitterState, GeneticAlgorithm
from learned_qd.evo.populations.population import Population
from learned_qd.tasks import Task, TaskParams, TaskState


@dataclass
class EvolutionState:
	key: jax.Array
	population: Population
	emitter_state: EmitterState
	task_params: TaskParams
	task_state: TaskState


class Evolution:
	def __init__(self, ga: GeneticAlgorithm, task: Task):
		self.ga = ga
		self.task = task

	@partial(jax.jit, static_argnames=("self",))
	def init(self, population: Population, key: jax.Array) -> EvolutionState:
		(
			key_x_init,
			key_task_sample,
			key_task_init,
			key_eval,
			key_ga_init,
			key_evolution,
		) = jax.random.split(key, 6)

		# Init population
		keys = jax.random.split(key_x_init, population.max_size)
		x = jax.vmap(self.task.sample_x)(keys)

		# Init task
		task_params = self.task.sample(key_task_sample)
		task_state = self.task.init(key_task_init, task_params)

		# Init GA
		task_state, task_eval = self.task.evaluate(key_eval, x, task_state, task_params)
		population, emitter_state, metrics = self.ga.init(
			population,
			x,
			task_eval.fitness,
			task_eval.descriptor,
			{},
			key_ga_init,
		)

		return EvolutionState(
			key=key_evolution,
			population=population,
			emitter_state=emitter_state,
			task_params=task_params,
			task_state=task_state,
		), metrics

	@partial(jax.jit, static_argnames=("self", "num_generations", "all_steps"))
	def evolve(
		self, evo_state: EvolutionState, metrics, num_generations: int, all_steps=True
	) -> tuple[EvolutionState, dict[str, jax.Array]]:
		def scan_update(carry, x):
			population, emitter_state, task_state, _ = carry
			key = x

			key_ask, key_evaluate = jax.random.split(key)

			# Generate offspring
			x, extra_info = self.ga.ask(population, emitter_state, key_ask)

			# Evaluate offspring
			task_state, task_eval = self.task.evaluate(
				key_evaluate, x, task_state, evo_state.task_params
			)

			# Add offspring to lpop and update emitter state
			population, emitter_state, metrics = self.ga.tell(
				population,
				emitter_state,
				x,
				task_eval.fitness,
				task_eval.descriptor,
				{} | extra_info,
			)

			return (population, emitter_state, task_state, metrics), metrics if all_steps else None

		key, subkey = jax.random.split(evo_state.key)
		keys = jax.random.split(subkey, num_generations)
		(population, emitter_state, task_state, metrics), all_metrics = jax.lax.scan(
			scan_update,
			(evo_state.population, evo_state.emitter_state, evo_state.task_state, metrics),
			keys,
		)

		return EvolutionState(
			key=key,
			population=population,
			emitter_state=emitter_state,
			task_params=evo_state.task_params,
			task_state=task_state,
		), all_metrics if all_steps else metrics
