import os
import pickle
import time

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import DictConfig, OmegaConf

from learned_qd.evo.genetic_algorithm import GeneticAlgorithm
from learned_qd.evo.metrics import metrics_fn
from learned_qd.meta.meta_evaluator import MetaEvaluator
from learned_qd.meta.meta_evolution import MetaEvolution
from learned_qd.meta.meta_objective import MetaObjective
from learned_qd.utils.helpers import get_config_and_model_path


@hydra.main(config_path="configs", config_name="learn", version_base=None)
def main(config: DictConfig) -> None:
	wandb.init(
		project="Learned-QD",
		name=f"learn-{os.getcwd().split('/')[-1]}",
		tags=["learn"] + config.tags,
		config=OmegaConf.to_container(config, resolve=True),
		mode="online" if config.wandb else "disabled",
	)

	key = jax.random.key(config.seed)

	# Init devices and mesh
	num_devices = config.num_devices or jax.device_count()
	assert num_devices <= jax.device_count(), (
		f"Requested {num_devices} devices, but only {jax.device_count()} available."
	)
	devices = jax.devices()[:num_devices]
	mesh = Mesh(devices, ("devices",))

	# Define sharding specifications
	replicate_sharding = NamedSharding(mesh, PartitionSpec())
	parallel_sharding = NamedSharding(mesh, PartitionSpec("devices"))

	# 1. Meta-task
	key, subkey = jax.random.split(key)
	meta_task = hydra.utils.instantiate(config.task)
	task = hydra.utils.instantiate(config.task)
	x = task.sample_x(subkey)

	genotype_count = sum(x.size for x in jax.tree.leaves(x))
	wandb.log({"genotype_count": genotype_count})

	# 2. Inner loop
	# Population
	if config.evo.run_path is not None:
		config_run, model_path = get_config_and_model_path(config.evo.run_path)
		config.evo.population.learned_fitness = config_run.evo.population.learned_fitness
		wandb.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)

	key, subkey = jax.random.split(key)
	population = hydra.utils.instantiate(config.evo.population)(
		x,
		subkey,
		descriptor_size=task.descriptor_size,
	)

	if config.evo.run_path is not None:
		# Load params
		with open(os.path.join(model_path, "params.pickle"), "rb") as f:
			params = pickle.load(f)

		population = population.replace(params=params)

	params_count = sum(x.size for x in jax.tree.leaves(population.params))
	wandb.log({"params_count": params_count})

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

	# 3. Meta-evolution strategy
	meta_strategy, meta_params = hydra.utils.instantiate(config.meta_strategy)(
		solution=population.params
	)

	meta_objective_fn = hydra.utils.instantiate(config.meta_objective_fn)
	meta_objective = MetaObjective(meta_objective_fn=meta_objective_fn)
	meta_evolution = MetaEvolution(
		ga=ga,
		task=meta_task,
		population=population,
		num_generations=config.num_generations,
	)
	meta_evaluator = MetaEvaluator(config, parallel_sharding)

	# 4. Run meta-training loop
	meta_params = jax.device_put(meta_params, replicate_sharding)

	key, subkey = jax.random.split(key)
	meta_state = jax.jit(meta_strategy.init, out_shardings=replicate_sharding)(
		subkey, population.params, meta_params
	)

	if config.evo.run_path is not None:
		mean = meta_strategy._ravel_solution(params)
		meta_state = meta_state.replace(mean=mean)

	# Run first initialization at meta-training start
	key, subkey = jax.random.split(key)
	params = meta_strategy.get_mean(meta_state)
	metrics = meta_evaluator.evaluate_step(params, subkey)

	# Log
	metrics = {
		"meta_generation": 0,
		**metrics,
		"params_norm": jnp.linalg.norm(meta_state.mean),
		"time": 0.0,
		"time_eval": 0.0,
		"iterations_per_second": 0.0,
	}
	metrics_df = pd.DataFrame([metrics])
	metrics_df.to_csv("./metrics.csv", index=False)
	wandb.log(metrics)

	@jax.jit
	def train_step(meta_state, key):
		key_ask, key_eval, key_tell = jax.random.split(key, 3)

		# Ask
		params, meta_state = jax.jit(
			meta_strategy.ask, out_shardings=(parallel_sharding, replicate_sharding)
		)(key_ask, meta_state, meta_params)

		# Eval
		keys = jax.random.split(key_eval, config.meta_batch_size)
		evo_state, metrics = meta_evolution.init(params, keys)
		evo_state, metrics = meta_evolution.evolve(evo_state, metrics)

		# Standardize
		meta_fitness = meta_objective.apply(metrics)

		# Tell
		meta_state, _ = meta_strategy.tell(key_tell, params, -meta_fitness, meta_state, meta_params)

		return meta_state

	@jax.jit
	def eval_step(meta_state, key):
		params = meta_strategy.get_mean(meta_state)
		metrics = meta_evaluator.evaluate_step(params, key)
		return metrics

	@jax.jit
	def eval_final(meta_state, key):
		params = meta_strategy.get_mean(meta_state)
		metrics = meta_evaluator.evaluate_final(params, key)
		return metrics

	for i in range(1, config.num_meta_generations + 1):
		start_time = time.time()

		key, subkey = jax.random.split(key)
		meta_state = train_step(meta_state, subkey)

		time_elapsed = time.time() - start_time

		# Log
		if i % config.log_every == 0:
			time_eval = time.time()

			key, subkey = jax.random.split(key)
			metrics = eval_step(meta_state, subkey)

			time_eval_elapsed = time.time() - time_eval

			metrics = {
				"meta_generation": i,
				**metrics,
				"params_norm": jnp.linalg.norm(meta_state.mean),
				"time": time_elapsed,
				"time_eval": time_eval_elapsed,
				"iterations_per_second": 1 / time_elapsed,
			}
			metrics_df = pd.DataFrame([metrics])
			metrics_df.to_csv("./metrics.csv", mode="a", header=False, index=False)
			wandb.log(metrics)

	with open("./params.pickle", "wb") as f:
		params = meta_strategy.get_mean(meta_state)
		pickle.dump(params, f)

	artifact = wandb.Artifact("params", type="model")
	artifact.add_file("./params.pickle")
	wandb.log_artifact(artifact)

	# Evaluate
	metrics = eval_final(meta_state, key)

	for metric_name, metric_values in metrics.items():
		num_generations = jnp.arange(len(metric_values))
		data = [[gen, val] for gen, val in zip(num_generations, metric_values)]
		table = wandb.Table(data=data, columns=["num_generations", metric_name])
		wandb.log(
			{metric_name: wandb.plot.line(table, "num_generations", metric_name, title=metric_name)}
		)


if __name__ == "__main__":
	main()
