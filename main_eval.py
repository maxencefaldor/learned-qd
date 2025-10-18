import os
import pickle
import time
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import pandas as pd
import wandb
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import DictConfig, OmegaConf

from learned_qd.evo.evolution import Evolution
from learned_qd.evo.genetic_algorithm import GeneticAlgorithm
from learned_qd.evo.metrics import metrics_agg_fn, metrics_fn
from learned_qd.utils.helpers import get_config_and_model_path
from learned_qd.utils.plot import plot_evolution


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(config: DictConfig) -> None:
	wandb.init(
		project="Learned-QD",
		name=f"eval-{os.getcwd().split('/')[-1]}",
		tags=["eval"] + config.tags,
		config=OmegaConf.to_container(config, resolve=True),
		mode="online" if config.wandb else "disabled",
	)

	key = jax.random.key(config.seed)

	num_devices = config.num_devices or jax.device_count()
	assert num_devices <= jax.device_count(), (
		f"Requested {num_devices} devices, but only {jax.device_count()} available."
	)
	devices = jax.devices()[:num_devices]
	mesh = Mesh(devices, ("devices",))

	# Define sharding specifications
	parallel_sharding = NamedSharding(mesh, PartitionSpec("devices"))

	# Task
	key, subkey = jax.random.split(key)
	task = hydra.utils.instantiate(config.task)
	x = task.sample_x(subkey)

	genotype_count = sum(x.size for x in jax.tree.leaves(x))
	wandb.log({"genotype_count": genotype_count})

	# Population
	if config.evo.name == "lqd":
		config_run, model_path = get_config_and_model_path(config.evo.run_path)
		config.evo.population.learned_fitness = config_run.evo.population.learned_fitness
		wandb.config.update(OmegaConf.to_container(config, resolve=True), allow_val_change=True)

		key, subkey = jax.random.split(key)
		population = hydra.utils.instantiate(config.evo.population)(
			x,
			subkey,
			descriptor_size=task.descriptor_size,
		)

		# Load params
		with open(os.path.join(model_path, "params.pickle"), "rb") as f:
			params = pickle.load(f)

		population = population.replace(params=params)

		params_count = sum(x.size for x in jax.tree.leaves(params))
	else:
		key, subkey = jax.random.split(key)
		population = hydra.utils.instantiate(config.evo.population)(
			x,
			subkey,
			descriptor_size=task.descriptor_size,
		)

		params_count = 0
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

	# Evolution
	evo = Evolution(ga, task)

	# Run
	key, subkey = jax.random.split(key)
	keys = jax.random.split(subkey, config.num_evaluations)
	keys = jax.device_put(keys, parallel_sharding)
	evo_state, metrics_init = jax.vmap(evo.init, in_axes=(None, 0))(population, keys)

	# Log
	fitness_all = [jnp.expand_dims(metrics_init["fitness"][0], axis=0)]
	descriptor_all = [jnp.expand_dims(metrics_init["descriptor"][0], axis=0)]

	current_metrics_df = pd.DataFrame(
		{
			"batch_id": range(config.num_evaluations),
			"generation": [0] * config.num_evaluations,
			**metrics_agg_fn(metrics_init),
			"time": [0.0] * config.num_evaluations,
			"iterations_per_second": [0.0] * config.num_evaluations,
		}
	)

	metrics_df = current_metrics_df.copy()

	metrics_agg = (
		current_metrics_df.drop(columns="batch_id").groupby("generation").mean().reset_index()
	)
	metrics_agg.to_csv("./metrics.csv", index=False)
	wandb.log(metrics_agg.to_dict("records")[0])

	for i in range(config.num_generations // config.log_every):
		start_time = time.time()

		evo_state, metrics = jax.vmap(partial(evo.evolve, num_generations=config.log_every))(
			evo_state, metrics_init
		)

		time_elapsed = time.time() - start_time

		# Log
		fitness_all.append(metrics["fitness"][0])
		descriptor_all.append(metrics["descriptor"][0])

		current_metrics_df = pd.DataFrame(
			{
				"batch_id": [
					i for i in range(config.num_evaluations) for _ in range(config.log_every)
				],
				"generation": [
					gen
					for _ in range(config.num_evaluations)
					for gen in range(1 + i * config.log_every, 1 + (i + 1) * config.log_every)
				],
				**{k: v.reshape(-1) for k, v in metrics_agg_fn(metrics).items()},
				"time": [time_elapsed] * (config.num_evaluations * config.log_every),
				"iterations_per_second": [config.log_every / time_elapsed]
				* (config.num_evaluations * config.log_every),
			}
		)

		metrics_df = pd.concat([metrics_df, current_metrics_df])

		metrics_agg = (
			current_metrics_df.drop(columns="batch_id").groupby("generation").mean().reset_index()
		)
		metrics_agg.to_csv("./metrics.csv", mode="a", header=False, index=False)
		wandb.log(metrics_agg.to_dict("records")[-1])

	with open("./metrics.pickle", "wb") as f:
		pickle.dump(metrics_df, f)

	with open("./population.pickle", "wb") as f:
		pickle.dump(evo_state.population, f)

	fitness_all = jnp.concatenate(fitness_all)
	descriptor_all = jnp.concatenate(descriptor_all)

	if config.log_evolution and task.descriptor_size == 2:
		anim = plot_evolution(fitness_all, descriptor_all)
		anim.save("./evolution.mp4", writer="ffmpeg")
		wandb.log({"evolution": wandb.Video("./evolution.mp4")})

	if config.log_fitness_descriptor:
		with open("./fitness_all.pickle", "wb") as f:
			pickle.dump(fitness_all, f)

		with open("./descriptor_all.pickle", "wb") as f:
			pickle.dump(descriptor_all, f)


if __name__ == "__main__":
	main()
