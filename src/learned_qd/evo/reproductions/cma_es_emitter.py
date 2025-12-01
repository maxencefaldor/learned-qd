"""CMA-ES Emitter."""

import jax
from flax import struct
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from learned_qd.es.algorithms.distribution_based.cma_es import CMA_ES
from learned_qd.evo.populations.population import Population


@struct.dataclass
class CMAESEmitterState(EmitterState):
	"""State of the CMA-ES emitter.

	Args:
		es_state: State of the underlying CMA-ES algorithm.
		params: Parameters of the CMA-ES algorithm.
		key: JAX random key.
	"""

	es_state: struct.PyTreeNode
	params: struct.PyTreeNode
	key: RNGKey


class CMAESEmitter(Emitter):
	"""Emitter that uses Sep-CMA-ES to generate offspring."""

	def __init__(
		self,
		population_size: int,
	):
		self._population_size = population_size

	@property
	def batch_size(self) -> int:
		"""Batch size of the emitter."""
		return self._population_size

	def _get_solution_template(self, genotype: Genotype) -> Genotype:
		return jax.tree.map(lambda x: x[0], genotype)

	def init(
		self,
		key: RNGKey,
		repertoire: Population,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		extra_scores: ExtraScores,
	) -> CMAESEmitterState:
		"""Initialise the emitter."""
		solution = self._get_solution_template(genotypes)
		es = CMA_ES(
			population_size=self._population_size,
			solution=solution,
		)

		key, es_key = jax.random.split(key)
		params = es.default_params
		es_state = es.init(es_key, solution, params)
		es_state = es_state.replace(
			best_solution=es.solution_flat,
			best_fitness=-fitnesses[0],
		)
		return CMAESEmitterState(es_state=es_state, params=params, key=key)

	def emit(
		self,
		repertoire: Population,
		emitter_state: CMAESEmitterState,
		key: RNGKey,
	) -> tuple[Genotype, ExtraScores]:
		"""Generate offspring."""
		solution = self._get_solution_template(repertoire.genotype)
		es = CMA_ES(
			population_size=self._population_size,
			solution=solution,
		)
		population, _ = es.ask(key, emitter_state.es_state, emitter_state.params)
		return population, {}

	def state_update(
		self,
		emitter_state: CMAESEmitterState,
		repertoire: Population,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		extra_scores: ExtraScores,
	) -> CMAESEmitterState:
		"""Update the emitter state."""
		solution = self._get_solution_template(genotypes)
		es = CMA_ES(
			population_size=self._population_size,
			solution=solution,
		)
		es_state_from_emit = extra_scores.get("es_state", emitter_state.es_state)
		tell_key, new_key = jax.random.split(emitter_state.key)
		es_state, _ = es.tell(
			tell_key,
			genotypes,
			-fitnesses,
			es_state_from_emit,
			emitter_state.params,
		)
		return emitter_state.replace(es_state=es_state, key=new_key)


def get_cma_es_emitter(
	batch_size: int,
	minval: float | None = None,
	maxval: float | None = None,
) -> CMAESEmitter:
	"""Returns a CMAESEmitter."""
	return CMAESEmitter(population_size=batch_size)
