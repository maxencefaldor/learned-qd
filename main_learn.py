import os
import pickle
import time

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from learned_qd.evo.genetic_algorithm import GeneticAlgorithm
from learned_qd.evo.metrics import metrics_fn
from learned_qd.meta.meta_evaluator import MetaEvaluator
from learned_qd.meta.meta_evolution import MetaEvolution
from learned_qd.utils.helpers import get_config_and_model_path


@hydra.main(config_path="configs", config_name="learn", version_base=None)
def main(config: DictConfig) -> None:
	wandb.init(
		project="Learned-QD",
		name=f"learn-{os.getcwd().split("/")[-1]}",
		tags=["learn"] + config.tags,
		config=OmegaConf.to_container(config, resolve=True),
		mode="online" if config.wandb else "disabled",
	)

	key = jax.random.key(config.seed)

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
		metrics_fn=lambda population: {
			k: v for k, v in metrics_fn(population).items() if k in config.meta_objective.select_key
		},
	)

	# 3. Meta-evolution strategy
	meta_strategy = hydra.utils.instantiate(config.meta_strategy)(
		pholder_params=population.params,
	)
	meta_params = meta_strategy.default_params

	meta_objective = hydra.utils.instantiate(config.meta_objective)
	meta_evolution = MetaEvolution(
		ga=ga,
		task=meta_task,
		population=population,
		num_generations=config.num_generations,
	)
	meta_evaluator = MetaEvaluator(config)

	# 4. Run meta-training loop
	key, subkey = jax.random.split(key)
	meta_state = meta_strategy.initialize(subkey, meta_params)

	if config.evo.run_path is not None:
		meta_state = meta_strategy.set_mean(meta_state, params)

	# Run first initialization at meta-training start
	key, subkey = jax.random.split(key)
	params = meta_strategy.get_eval_params(meta_state)
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

	for i in range(1, config.num_meta_generations + 1):
		start_time = time.time()

		key, subkey = jax.random.split(key)
		params, meta_state = meta_strategy.ask(subkey, meta_state, meta_params)

		# Run inner loop
		key, subkey = jax.random.split(key)
		keys = jax.random.split(subkey, config.meta_batch_size)
		evo_state, metrics = meta_evolution.init(params, keys)
		evo_state, metrics = meta_evolution.evolve(evo_state, metrics)

		# Normalize fitness
		meta_fitness = meta_objective.apply(params, metrics)

		# Update meta-state
		meta_state = meta_strategy.tell(params, meta_fitness, meta_state, meta_params)

		time_elapsed = time.time() - start_time

		# Log
		if i % config.log_every == 0:
			time_eval = time.time()

			key, subkey = jax.random.split(key)
			params = meta_strategy.get_eval_params(meta_state)
			metrics = meta_evaluator.evaluate_step(params, subkey)

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
		params = meta_strategy.get_eval_params(meta_state)
		pickle.dump(params, f)

	artifact = wandb.Artifact("params", type="model")
	artifact.add_file("./params.pickle")
	wandb.log_artifact(artifact)

	# Evaluate
	metrics = meta_evaluator.evaluate_final(params, subkey)
	wandb.log(metrics)


if __name__ == "__main__":
	main()
