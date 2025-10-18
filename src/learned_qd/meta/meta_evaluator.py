import os
from functools import partial

import hydra
import jax
import jax.numpy as jnp
from omegaconf import DictConfig

from learned_qd.evo.evolution import Evolution
from learned_qd.evo.genetic_algorithm import GeneticAlgorithm
from learned_qd.evo.metrics import metrics_agg_fn, metrics_fn


class MetaEvaluator:
	"""Meta-evaluator class for learned QD."""

	def __init__(self, config: DictConfig, parallel_sharding: jax.Array) -> None:
		self.parallel_sharding = parallel_sharding

		# Get step evaluation functions
		self.eval_step = {}
		config_dir = hydra.utils.to_absolute_path("configs/meta_evaluator/step")
		for config_file in sorted([f for f in os.listdir(config_dir) if f.endswith(".yaml")]):
			config_name = os.path.splitext(config_file)[0]
			config_eval = hydra.compose(
				config_name=f"meta_evaluator/step/{config_name}"
			).meta_evaluator.step

			if config_eval.evo.name == "lqd":
				config_eval.evo.population.learned_fitness = config.evo.population.learned_fitness

			self.eval_step[config_name] = self.make_eval(config_eval)

		# Get final evaluation functions
		self.eval_final = {}
		config_dir = hydra.utils.to_absolute_path("configs/meta_evaluator/final")
		for config_file in sorted([f for f in os.listdir(config_dir) if f.endswith(".yaml")]):
			config_name = os.path.splitext(config_file)[0]
			config_eval = hydra.compose(
				config_name=f"meta_evaluator/final/{config_name}"
			).meta_evaluator.final

			if config_eval.evo.name == "lqd":
				config_eval.evo.population.learned_fitness = config.evo.population.learned_fitness

			self.eval_final[config_name] = self.make_eval(config_eval)

	def make_eval(self, config: DictConfig) -> Evolution:
		@jax.jit
		def eval(params, key: jax.Array) -> dict[str, jax.Array]:
			# Task
			key, subkey = jax.random.split(key)
			task = hydra.utils.instantiate(config.task)
			x = task.sample_x(subkey)

			# Population
			key, subkey = jax.random.split(key)
			population = hydra.utils.instantiate(config.evo.population)(
				x,
				subkey,
				descriptor_size=task.descriptor_size,
			)

			if config.evo.name == "lqd":
				population = population.replace(params=params)

			# Reproduction
			emitter = hydra.utils.instantiate(config.evo.reproduction)(
				minval=task.x_range[0],
				maxval=task.x_range[1],
			)

			# Genetic Algorithm
			ga = GeneticAlgorithm(
				emitter=emitter,
				metrics_fn=metrics_fn,
			)

			# Evolution
			evo = Evolution(ga, task)

			# Run
			key, subkey = jax.random.split(key)
			keys = jax.random.split(subkey, config.num_evaluations)
			evo_state, metrics_init = jax.vmap(evo.init, in_axes=(None, 0))(population, keys)
			evo_state, metrics = jax.vmap(
				partial(evo.evolve, num_generations=config.num_generations, all_steps=True)
			)(evo_state, metrics_init)

			metrics_init = jax.lax.with_sharding_constraint(metrics_init, self.parallel_sharding)
			metrics = jax.lax.with_sharding_constraint(metrics, self.parallel_sharding)

			return metrics_init, metrics

		return eval

	def evaluate_step(self, params, key: jax.Array) -> dict[str, jax.Array]:
		all_metrics = {}

		for name, eval in self.eval_step.items():
			key, subkey = jax.random.split(key)
			_, metrics = eval(params, subkey)

			# Compute aggregated metrics
			metrics = metrics_agg_fn(metrics)

			# Take mean across evaluations for last generation
			metrics = jax.tree.map(lambda x: jnp.mean(x[:, -1], axis=0), metrics)

			# Add task name prefix to each metric key
			for metric_name, metric_value in metrics.items():
				all_metrics[f"step/{name}_{metric_name}"] = metric_value

		return all_metrics

	def evaluate_final(self, params, key: jax.Array) -> dict[str, jax.Array]:
		all_metrics = {}

		for name, eval in self.eval_final.items():
			key, subkey = jax.random.split(key)
			metrics_init, metrics = eval(params, subkey)

			# Concatenate metrics
			metrics = jax.tree.map(
				lambda x, y: jnp.concatenate([x[:, None], y], axis=1), metrics_init, metrics
			)

			# Compute aggregated metrics
			metrics = metrics_agg_fn(metrics)

			# Take mean across evaluations
			metrics = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)

			# Add task name prefix to each metric key
			for metric_name, metric_value in metrics.items():
				all_metrics[f"final/{name}_{metric_name}"] = metric_value

		return all_metrics
