import jax
from evosax.types import Fitness, Params, Population, State


def centered_rank_fitness_shaping_fn(
	population: Population, fitness: Fitness, state: State, params: Params
) -> Fitness:
	"""Return centered ranks in [-0.5, 0.5] according to fitness."""
	ranks = jax.scipy.stats.rankdata(-fitness, axis=-1) - 1.0
	return ranks / (fitness.shape[-1] - 1) - 0.5
