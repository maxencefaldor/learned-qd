"""OpenAI Evolution Strategy (Salimans et al. 2017).

[1] https://arxiv.org/abs/1703.03864
[2] https://github.com/hardmaru/estool/blob/master/es.py

see: https://github.com/louiskirsch/vsml-neurips2021/blob/5b696c968cf156c7aa3010e53abe83da0fb88fed/experiment.py#L303
see: https://github.com/louiskirsch/vsml-neurips2021/blob/5b696c968cf156c7aa3010e53abe83da0fb88fed/configs/default.yaml
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from evosax.types import Fitness, Population, Solution
from flax import struct

from learned_qd.es.core.fitness_shaping import centered_rank_fitness_shaping_fn

from .base import (
	DistributionBasedAlgorithm,
	metrics_fn,
)
from .base import (
	Params as BaseParams,
)
from .base import (
	State as BaseState,
)


@struct.dataclass
class State(BaseState):
	mean: jax.Array
	std: jax.Array
	opt_state: optax.OptState


@struct.dataclass
class Params(BaseParams):
	pass


class Open_ES(DistributionBasedAlgorithm):
	"""OpenAI Evolution Strategy (OpenAI-ES)."""

	def __init__(
		self,
		population_size: int,
		solution: Solution,
		optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
		std_schedule: Callable = optax.constant_schedule(1.0),
		fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
		metrics_fn: Callable = metrics_fn,
	):
		"""Initialize OpenAI-ES."""
		assert population_size % 2 == 0, "Population size must be even."
		super().__init__(population_size, solution, fitness_shaping_fn, metrics_fn)

		# Optimizer
		self.optimizer = optimizer

		# std schedule
		self.std_schedule = std_schedule

	@classmethod
	def build(
		cls,
		population_size: int,
		solution: Solution,
		optimizer: optax.GradientTransformation = optax.sgd(learning_rate=1e-3),
		std_schedule: Callable = optax.constant_schedule(1.0),
		fitness_shaping_fn: Callable = centered_rank_fitness_shaping_fn,
		metrics_fn: Callable = metrics_fn,
	) -> "Open_ES":
		es = Open_ES(
			population_size=population_size,
			solution=solution,
			optimizer=optimizer,
			std_schedule=std_schedule,
			fitness_shaping_fn=fitness_shaping_fn,
			metrics_fn=metrics_fn,
		)

		params = es.default_params
		return es, params

	@property
	def _default_params(self) -> Params:
		return Params()

	def _init(self, key: jax.Array, params: Params) -> State:
		state = State(
			mean=jnp.full((self.num_dims,), jnp.nan),
			std=self.std_schedule(0),
			opt_state=self.optimizer.init(jnp.zeros(self.num_dims)),
			best_solution=jnp.full((self.num_dims,), jnp.nan),
			best_fitness=jnp.inf,
			generation_counter=0,
		)
		return state

	def _ask(
		self,
		key: jax.Array,
		state: State,
		params: Params,
	) -> tuple[Population, State]:
		z_plus = jax.random.normal(key, (self.population_size // 2, self.num_dims))
		z = jnp.concatenate([z_plus, -z_plus])
		population = state.mean + state.std * z
		return population, state

	def _tell(
		self,
		key: jax.Array,
		population: Population,
		fitness: Fitness,
		state: State,
		params: Params,
	) -> State:
		z_scaled = population[: self.population_size // 2] - state.mean
		fitness_plus = fitness[: self.population_size // 2]
		fitness_minus = fitness[self.population_size // 2 :]

		# Compute grad
		grad_mean = jnp.dot(fitness_plus - fitness_minus, z_scaled / state.std) / (
			self.population_size * state.std
		)

		# Update mean
		updates, opt_state = self.optimizer.update(-grad_mean, state.opt_state)
		mean = optax.apply_updates(state.mean, updates)

		return state.replace(
			mean=mean,
			std=self.std_schedule(state.generation_counter),
			opt_state=opt_state,
		)
