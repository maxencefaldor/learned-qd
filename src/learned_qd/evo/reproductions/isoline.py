from functools import partial

from qdax.core.emitters.mutation_operators import isoline_variation

from learned_qd.evo.reproductions.standard_emitter import StandardEmitter


def get_isoline_emitter(
	batch_size: int,
	iso_sigma: float,
	line_sigma: float,
	minval: float = None,
	maxval: float = None,
) -> StandardEmitter:
	variation_fn = partial(
		isoline_variation,
		iso_sigma=iso_sigma,
		line_sigma=line_sigma,
		minval=minval,
		maxval=maxval,
	)
	emitter = StandardEmitter(
		mutation_fn=None,
		variation_fn=variation_fn,
		variation_percentage=1.0,
		batch_size=batch_size,
	)
	return emitter
