import os
import pickle
import time
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
import pandas as pd
import wandb
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf

from learned_qd.evo.evolution import Evolution
from learned_qd.evo.genetic_algorithm import GeneticAlgorithm
from learned_qd.evo.metrics import metrics_fn


@hydra.main(config_path="configs", config_name="distill", version_base=None)
def main(config: DictConfig) -> None:
	wandb.init(
		project="Learned-QD",
		name=f"distill-{os.getcwd().split("/")[-1]}",
		tags=["distill"] + config.tags,
		config=OmegaConf.to_container(config, resolve=True),
		mode="online" if config.wandb else "disabled",
	)

	key = jax.random.key(config.seed)

	# Task
	key, subkey = jax.random.split(key)
	task = hydra.utils.instantiate(config.task)
	x = task.sample_x(subkey)

	genotype_count = sum(x.size for x in jax.tree.leaves(x))
	wandb.log({"genotype_count": genotype_count})

	key, subkey = jax.random.split(key)
	population = hydra.utils.instantiate(config.teacher.population)(
		x,
		subkey,
		descriptor_size=task.descriptor_size,
	)

	# Reproduction
	emitter = hydra.utils.instantiate(config.evo.reproduction)(
		minval=task.x_range[0],
		maxval=task.x_range[1],
	)

	# Genetic Algorithm
	ga = GeneticAlgorithm(
		emitter=emitter,
		metrics_fn=lambda population: {
			k: v for k, v in metrics_fn(population).items() if k in ["fitness", "descriptor", "dominated_novelty"]
		},
	)

	# Evolution
	evo = Evolution(ga, task)

	# Init params
	key, subkey = jax.random.split(key)
	learned_fitness = hydra.utils.instantiate(config.evo.population.learned_fitness)
	fitness_dummy = jnp.zeros((config.evo.population.max_size,))
	descriptor_dummy = jnp.zeros((config.evo.population.max_size, task.descriptor_size))
	params = learned_fitness.init(key, fitness_dummy, descriptor_dummy)

	params_count = sum(x.size for x in jax.tree.leaves(params))
	wandb.log({"params_count": params_count})

	# Create train state
	dataset_size = config.num_tasks * (config.num_generations + 1)
	warmup_steps = int(0.1 * config.num_epochs * (dataset_size // config.batch_size))  # 10% of total steps for warmup
	total_steps = config.num_epochs * (dataset_size // config.batch_size)

	schedule = optax.join_schedules(
		schedules=[
			optax.linear_schedule(  # Warm up
				init_value=0.0,
				end_value=config.learning_rate,
				transition_steps=warmup_steps,
			),
			optax.cosine_decay_schedule(  # Cosine decay
				init_value=config.learning_rate,
				decay_steps=total_steps - warmup_steps,
			),
		],
		boundaries=[warmup_steps],
	)

	tx = optax.chain(
		optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
		optax.adamw(learning_rate=schedule),
	)
	train_state = TrainState.create(apply_fn=learned_fitness.apply, params=params, tx=tx)

	@jax.jit
	def generate_dataset(key):
		# Generate
		key, subkey = jax.random.split(key)
		keys = jax.random.split(subkey, config.num_tasks)
		evo_state, metrics_init = jax.vmap(evo.init, in_axes=(None, 0))(population, keys)
		evo_state, metrics = jax.vmap(partial(evo.evolve, num_generations=config.num_generations))(
			evo_state, metrics_init
		)
		dataset = jax.tree.map(
			lambda x, y: jnp.concatenate([jnp.expand_dims(x, axis=1), y], axis=1), metrics_init, metrics
		)
		dataset = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), dataset)

		# Shuffle
		dataset_size = dataset["fitness"].shape[0]
		permutation = jax.random.permutation(key, dataset_size)
		dataset = jax.tree.map(lambda x: x[permutation], dataset)

		return dataset

	def loss_fn(params, batch):
		# Prediction
		meta_fitness = jax.vmap(train_state.apply_fn, in_axes=(None, 0, 0))(
			params, batch["fitness"], batch["descriptor"]
		)

		# Target
		dominated_novelty = jnp.where(
			jnp.isnan(batch["dominated_novelty"]),
			jnp.max(
				batch["dominated_novelty"],
				axis=-1,
				keepdims=True,
				initial=-jnp.inf,
				where=~jnp.isnan(batch["dominated_novelty"]),
			),
			batch["dominated_novelty"],
		)

		# Normalize to zero mean, unit variance
		dominated_novelty = jax.nn.standardize(dominated_novelty, axis=-1)
		meta_fitness = jax.nn.standardize(meta_fitness, axis=-1)

		labels = jax.nn.softmax(dominated_novelty / config.temperature, axis=-1)
		loss = optax.softmax_cross_entropy(logits=meta_fitness / config.temperature, labels=labels)
		return jnp.mean(loss)
		# return jnp.mean((meta_fitness - dominated_novelty) ** 2)

	@jax.jit
	def train_step(train_state, batch):
		loss, grad = jax.value_and_grad(loss_fn)(train_state.params, batch)
		train_state = train_state.apply_gradients(grads=grad)
		return train_state, loss

	# Log
	key, subkey = jax.random.split(key)
	dataset = generate_dataset(subkey)
	batch = jax.tree.map(lambda x: x[: config.batch_size], dataset)
	loss = loss_fn(train_state.params, batch)

	metrics = {
		"epoch": 0,
		"num_train_steps": 0,
		"loss": float(loss),
		"params_norm": jnp.linalg.norm(jax.flatten_util.ravel_pytree(train_state.params)[0]),
		"time": 0.0,
		"learning_rate": 0.0,
	}
	metrics_df = pd.DataFrame([metrics])
	metrics_df.to_csv("./metrics.csv", index=False)
	wandb.log(metrics)

	for epoch in range(1, config.num_epochs + 1):
		start_time = time.time()

		# Generate dataset
		key, subkey = jax.random.split(key)
		dataset = generate_dataset(subkey)

		num_batches = dataset_size // config.batch_size
		for batch_idx in range(num_batches):
			batch = jax.tree.map(
				lambda x: x[batch_idx * config.batch_size : (batch_idx + 1) * config.batch_size], dataset
			)
			train_state, loss = train_step(train_state, batch)

		time_elapsed = time.time() - start_time

		# Log
		metrics = {
			"epoch": epoch,
			"num_train_steps": epoch * dataset_size // config.batch_size,
			"loss": float(loss),
			"params_norm": jnp.linalg.norm(jax.flatten_util.ravel_pytree(train_state.params)[0]),
			"time": time_elapsed,
			"learning_rate": schedule(epoch * dataset_size // config.batch_size),
		}
		metrics_df = pd.DataFrame([metrics])
		metrics_df.to_csv("./metrics.csv", mode="a", header=False, index=False)
		wandb.log(metrics)

	with open("./params.pickle", "wb") as f:
		pickle.dump(train_state.params, f)

	artifact = wandb.Artifact("params", type="model")
	artifact.add_file("./params.pickle")
	wandb.log_artifact(artifact)


if __name__ == "__main__":
	main()
