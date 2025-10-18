import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt

from learned_qd.utils.helpers import get_config
from learned_qd.utils.plot import evos_dict, run_paths

env_name = "ant"
descriptor_name = "velocity"
task_name = env_name + "_" + descriptor_name

task_names = {
	"hopper_feet_contact_velocity": "Hopper - Feet Contact Velocity",
	"walker2d_feet_contact": "Walker2d - Feet Contact",
	"halfcheetah_feet_contact": "Halfcheetah - Feet Contact",
	"ant_feet_contact": "Ant - Feet Contact",
	"ant_velocity": "Ant - Velocity",
	"arm_position": "Arm - Position",
}

batch_index = 0

evo_order = [
	"lqd_q",
	"lqd_d",
	"lqd_qd",
	"dns",
	"ga",
	"ns",
	"me",
	"random",
]

results_dir = "<path>"

# Collect populations from all algorithms
populations = {}

for evo_dir in os.listdir(results_dir):
	evo_path = os.path.join(results_dir, evo_dir)
	if not os.path.isdir(evo_path):
		continue

	# Get all run directories for evo algorithm
	for run_dir in os.listdir(evo_path):
		run_dir_path = os.path.join(evo_path, run_dir)
		if not os.path.isdir(run_dir_path):
			continue

		config = get_config(run_dir_path)

		# Check if this run matches our target environment and descriptor
		if (
			config.task._target_ == "learned_qd.tasks.brax.BraxTask"
			and config.task.env_name == env_name
			and config.task.descriptor == descriptor_name
		) or (
			config.task._target_ == "learned_qd.tasks.arm.ArmTask"
			and env_name == "arm"
			and config.task.descriptor == "position"
		):
			# Load population
			population_path = os.path.join(run_dir_path, "population.pickle")
			if not os.path.exists(population_path):
				continue

			with open(population_path, "rb") as f:
				population = pickle.load(f)

			# Determine the algorithm name
			evo_name = config.evo.name
			if evo_name == "lqd":
				# Map run_path to specific LQD variant
				for lqd_variant, path in run_paths.items():
					if config.evo.run_path == path:
						evo_name = lqd_variant
						break

			populations[evo_name] = population

for batch_index in range(32):
	# Fitness bounds
	fitness = jnp.concatenate(
		[population.fitness[batch_index] for population in populations.values()]
	)
	finite_fitness = fitness[jnp.isfinite(fitness)]
	global_vmin = jnp.percentile(finite_fitness, 25)  # First quartile
	global_vmax = jnp.percentile(finite_fitness, 75)  # Third quartile

	# Descriptor bounds
	descriptor_bounds = {
		("hopper", "feet_contact_velocity"): ([0.0, -5.0], [1.0, 5.0]),
		("walker2d", "feet_contact"): ([0.0, 0.0], [1.0, 1.0]),
		("halfcheetah", "feet_contact"): ([0.0, 0.0], [1.0, 1.0]),
		("ant", "velocity"): ([-5.0, -5.0], [5.0, 5.0]),
		("ant", "feet_contact"): ([0.0, 0.0], [1.0, 1.0]),
		("arm", "position"): ([0.0, 0.0], [1.0, 1.0]),
	}
	desc_min, desc_max = descriptor_bounds.get((env_name, descriptor_name), (None, None))

	# Split algorithms into two rows
	first_row = ["lqd_q", "lqd_d", "lqd_qd", "dns"]
	second_row = ["ga", "ns", "me", "random"]

	if len(evo_order) > 0:
		# Create figure with two rows and 4 columns
		fig, axes = plt.subplots(2, 4, figsize=(16, 8), squeeze=False)

		# Keep track of the last scatter plot for the colorbar
		last_scatter = None

		# Plot first row
		for i, evo in enumerate(first_row):
			if evo in populations:
				population = populations[evo]

				# Get first population in batch
				fitness = population.fitness[batch_index]
				descriptor = population.descriptor[batch_index]

				# Create scatter plot with global color scale
				scatter = axes[0, i].scatter(
					descriptor[:, 0],
					descriptor[:, 1],
					c=fitness,
					cmap="viridis",
					alpha=0.7,
					s=50,
					vmin=global_vmin,
					vmax=global_vmax,
				)

				# Keep reference to scatter plot for colorbar
				last_scatter = scatter

				# Customize the plot
				algo_name = evos_dict.get(evo, evo)
				axes[0, i].set_title(f"{algo_name}")
				axes[0, i].grid(True, alpha=0.3)

				# Set consistent bounds for all subplots
				if desc_min is not None and desc_max is not None:
					axes[0, i].set_xlim(desc_min[0], desc_max[0])
					axes[0, i].set_ylim(desc_min[1], desc_max[1])

		# Plot second row
		for i, evo in enumerate(second_row):
			if evo in populations:
				population = populations[evo]

				# Get first population in batch
				fitness = population.fitness[batch_index]
				descriptor = population.descriptor[batch_index]

				# Create scatter plot with global color scale
				scatter = axes[1, i].scatter(
					descriptor[:, 0],
					descriptor[:, 1],
					c=fitness,
					cmap="viridis",
					alpha=0.7,
					s=50,
					vmin=global_vmin,
					vmax=global_vmax,
				)

				# Keep reference to scatter plot for colorbar
				last_scatter = scatter

				# Customize the plot
				algo_name = evos_dict.get(evo, evo)
				axes[1, i].set_title(f"{algo_name}")
				axes[1, i].grid(True, alpha=0.3)

				# Set consistent bounds for all subplots
				if desc_min is not None and desc_max is not None:
					axes[1, i].set_xlim(desc_min[0], desc_max[0])
					axes[1, i].set_ylim(desc_min[1], desc_max[1])

		# Add global title
		fig.suptitle(task_names[task_name], fontsize=16, y=0.95)

		# Adjust layout first to make room for colorbar and title
		plt.subplots_adjust(right=0.92, top=0.88)

		# Add a single colorbar on the far right
		if last_scatter is not None:
			cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
			cbar = fig.colorbar(last_scatter, cax=cbar_ax)
			cbar.set_label("Fitness", rotation=90, labelpad=15)

		plt.savefig(
			f"output/tmp/plot_robot_population_{task_name}_{batch_index}.pdf",
			dpi=300,
			bbox_inches="tight",
		)
		plt.close()
	else:
		print(f"No populations found for {env_name} with {descriptor_name} descriptor")
