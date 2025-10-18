from functools import partial

import jax
import jax.numpy as jnp


def novelty_and_dominated_novelty(fitness, descriptor, novelty_k=3, dominated_novelty_k=3):
	valid = fitness != -jnp.inf

	# Neighbors
	neighbor = valid[:, None] & valid[None, :]
	neighbor = jnp.fill_diagonal(neighbor, False, inplace=False)

	# Fitter
	fitter = fitness[:, None] <= fitness[None, :]
	fitter = jnp.where(neighbor, fitter, False)

	# Distance to neighbors
	distance = jnp.linalg.norm(descriptor[:, None, :] - descriptor[None, :, :], axis=-1)
	distance = jnp.where(neighbor, distance, jnp.inf)

	# Distance to fitter neighbors
	distance_fitter = jnp.where(fitter, distance, jnp.inf)

	# Novelty - distance to k-nearest neighbors
	values, indices = jax.vmap(partial(jax.lax.top_k, k=novelty_k))(-distance)
	novelty = jnp.mean(-values, axis=-1, where=jnp.take_along_axis(neighbor, indices, axis=-1))

	# Dominated Novelty - distance to k-fitter-nearest neighbors
	values, indices = jax.vmap(partial(jax.lax.top_k, k=dominated_novelty_k))(-distance_fitter)
	dominated_novelty = jnp.mean(
		-values, axis=-1, where=jnp.take_along_axis(fitter, indices, axis=-1)
	)  # only max fitness individual should be nan

	return novelty, dominated_novelty


def metrics_fn(population):
	k = 3
	novelty, dominated_novelty = novelty_and_dominated_novelty(
		population.fitness,
		population.descriptor,
		novelty_k=k,
		dominated_novelty_k=k,
	)
	dominated_novelty = jnp.where(jnp.isposinf(dominated_novelty), jnp.nan, dominated_novelty)

	return {
		"fitness": population.fitness,
		"descriptor": population.descriptor,
		"novelty": novelty,
		"dominated_novelty": dominated_novelty,
	}


@jax.jit
def metrics_agg_fn(metrics: dict) -> dict:
	valid = metrics["fitness"] != -jnp.inf

	descriptor_mean = jnp.mean(metrics["descriptor"], axis=-2, where=valid[..., None])
	distance_to_mean = jnp.linalg.norm(
		metrics["descriptor"] - descriptor_mean[..., None, :], axis=-1
	)
	descriptor_std = jnp.std(distance_to_mean, axis=-1, where=valid)

	return {
		"fitness_max": jnp.max(metrics["fitness"], axis=-1, initial=-jnp.inf, where=valid),
		"fitness_min": jnp.min(metrics["fitness"], axis=-1, initial=jnp.inf, where=valid),
		"fitness_mean": jnp.mean(metrics["fitness"], axis=-1, where=valid),
		"fitness_sum": jnp.sum(metrics["fitness"], axis=-1, where=valid),
		"descriptor_std": descriptor_std,
		"novelty_max": jnp.max(metrics["novelty"], axis=-1, initial=-jnp.inf, where=valid),
		"novelty_min": jnp.min(metrics["novelty"], axis=-1, initial=jnp.inf, where=valid),
		"novelty_mean": jnp.mean(metrics["novelty"], axis=-1, where=valid),
		"novelty_sum": jnp.sum(metrics["novelty"], axis=-1, where=valid),
		"dominated_novelty_max": jnp.nanmax(
			metrics["dominated_novelty"], axis=-1, initial=-jnp.inf, where=valid
		),
		"dominated_novelty_min": jnp.nanmin(
			metrics["dominated_novelty"], axis=-1, initial=jnp.inf, where=valid
		),
		"dominated_novelty_mean": jnp.nanmean(metrics["dominated_novelty"], axis=-1, where=valid),
		"dominated_novelty_sum": jnp.nansum(metrics["dominated_novelty"], axis=-1, where=valid),
		"population_size": jnp.sum(valid, axis=-1),
	}
