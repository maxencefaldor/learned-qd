import os
import pickle

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from learned_qd.utils.helpers import get_config_and_model_path
from learned_qd.utils.plot import run_paths

eval_paths = {
	"lqd_q": "<path>",
	"lqd_d": "<path>",
	"lqd_qd": "<path>",
}
run_titles = ["LQD (Q)", "LQD (D)", "LQD (QD)"]
generation = 16

# Create figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for run_idx, evo in enumerate(run_paths):
	eval_path = eval_paths[evo]
	with open(os.path.join(eval_path, "fitness_all.pickle"), "rb") as f:
		fitness_all = pickle.load(f)

	with open(os.path.join(eval_path, "descriptor_all.pickle"), "rb") as f:
		descriptor_all = pickle.load(f)

	# Get run path and load config/model
	config_run, model_path = get_config_and_model_path(run_paths[evo])

	# Load params
	with open(os.path.join(model_path, "params.pickle"), "rb") as f:
		params = pickle.load(f)

	# Init learned fitness
	learned_fitness = hydra.utils.instantiate(config_run.evo.population.learned_fitness)

	# Compute descriptor bounds
	desc_min = jnp.min(descriptor_all)
	desc_max = jnp.max(descriptor_all)
	x = jnp.linspace(desc_min, desc_max, 128)
	y = jnp.linspace(desc_min, desc_max, 128)
	X, Y = jnp.meshgrid(x, y)
	grid = jnp.stack([X.ravel(), Y.ravel()], axis=1)

	def compute_learned_fitness_at_point(fitness, descriptor, point):
		idx = jnp.argsort(fitness)[len(fitness) // 2]
		descriptor = descriptor.at[..., idx, :].set(point)
		learned_fitness_ = learned_fitness.apply(params, fitness, descriptor)
		return learned_fitness_[..., idx]

	fitness, descriptor = fitness_all[generation], descriptor_all[generation]
	learned_fitness_grid = jax.vmap(compute_learned_fitness_at_point, in_axes=(None, None, 0))(
		fitness, descriptor, grid
	)

	# Reshape learned fitness values to match grid shape for heatmap
	Z = learned_fitness_grid.reshape(X.shape)

	# Plot heatmap on current subplot
	im = axes[run_idx].pcolormesh(X, Y, Z, shading="auto", cmap="viridis")

	axes[run_idx].scatter(
		descriptor[:, 0],
		descriptor[:, 1],
		s=64,
		c=fitness,
		cmap="viridis",
		edgecolors="black",
	)

	# Add labels and title for each subplot
	axes[run_idx].set_title(run_titles[run_idx])

	# Set equal aspect ratio and same limits for all plots
	axes[run_idx].set_aspect("equal")
	axes[run_idx].set_xlim([desc_min, desc_max])
	axes[run_idx].set_ylim([desc_min, desc_max])

plt.tight_layout()
plt.savefig("output/plot_bbo_heatmap.pdf")
plt.close()
