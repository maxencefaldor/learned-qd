import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import (
	customize_axis,
	evos_colors,
	evos_dict,
	metrics_dict,
	robot_task,
	run_paths,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", size=12)

results_dir = "<path>"

metric = "dominated_novelty_mean"
lqd = "lqd_qd"

run_path = run_paths[lqd]

x_label = "Evaluations"
y_label = metrics_dict[metric]

num_evaluations = 8192 * 32
population_size = 128
reproduction_batch_size = 32

# Get metrics for all runs
metrics_list = []
for evo_dir in os.listdir(results_dir):
	evo_path = os.path.join(results_dir, evo_dir)
	if not os.path.isdir(evo_path):
		continue

	# Get all run directories for evo algorithm
	for run_dir in os.listdir(evo_path):
		run_dir = os.path.join(evo_path, run_dir)
		if not os.path.isdir(run_dir):
			continue

		config = get_config(run_dir)
		metrics = get_metrics(run_dir)

		if config.evo.population.max_size != population_size:
			continue

		if config.evo.reproduction.batch_size != reproduction_batch_size:
			continue

		if "descriptor" in config.task and config.task.descriptor in ["random", "zero"]:
			continue

		metrics["evo"] = config.evo.name
		if config.evo.name == "lqd":
			if config.evo.run_path != run_path:
				continue
			metrics["run_path"] = config.evo.run_path
		else:
			metrics["run_path"] = None

		if config.task._target_ == "learned_qd.tasks.brax.BraxTask":
			metrics["env_name"] = config.task.env_name
			metrics["descriptor"] = config.task.descriptor
		elif config.task._target_ == "learned_qd.tasks.arm.ArmTask":
			metrics["env_name"] = "arm"
			if "descriptor" in config.task:
				metrics["descriptor"] = config.task.descriptor
			else:
				metrics["descriptor"] = "position"
			if config.task.num_joints != 8:
				continue

		metrics = metrics[
			[
				"evo",
				"batch_id",
				"env_name",
				"descriptor",
				"generation",
				"population_size",
				"fitness_max",
				"novelty_mean",
				"dominated_novelty_mean",
			]
		]
		metrics_list.append(metrics)

metrics = pd.concat(metrics_list, ignore_index=True)
metrics["evaluations"] = metrics["generation"] * reproduction_batch_size
metrics = metrics[metrics.evaluations <= num_evaluations]
metrics = metrics[~metrics.evo.isin(["ns"])]
metrics = metrics[~metrics.descriptor.isin(["random", "zero"])]

# Normalize metrics by population size
metrics["novelty_mean"] = metrics["novelty_mean"] * metrics["population_size"] / population_size
metrics["dominated_novelty_mean"] = (
	metrics["dominated_novelty_mean"] * metrics["population_size"] / population_size
)

# Create figure with subplots
num_tasks = len(robot_task)
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), squeeze=True)
axes = axes.flatten()

# Plot each task in order from robot_task
lines = []
labels = []
for plot_idx, task in enumerate(robot_task.keys()):
	# Customize subplot
	customize_axis(axes[plot_idx])
	axes[plot_idx].set_xlabel(x_label)
	if plot_idx % 3 == 0:  # Only set y label for leftmost subplots
		axes[plot_idx].set_ylabel(y_label)
	else:  # Remove y ticks labels for other subplots
		axes[plot_idx].set_ylabel("")
	axes[plot_idx].set_title(f"{robot_task[task]:^32}")
	# Use scientific notation for x-axis
	axes[plot_idx].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

	# Get data for this task
	if task == "arm":
		task_metrics = metrics[metrics.env_name == "arm"]
	else:
		# Split on last underscore to handle multiple underscores in env name
		env_name = task.split("_")[0]
		descriptor = "_".join(task.split("_")[1:])
		task_metrics = metrics[(metrics.env_name == env_name) & (metrics.descriptor == descriptor)]

	# Plot each algorithm in order from evos_dict
	for evo in reversed(evos_dict):
		if evo not in task_metrics.evo.unique() or evo == "random":
			continue

		evo_metrics = task_metrics[task_metrics.evo == evo]

		# For arm task, limit to 512 generations
		if task == "arm":
			evo_metrics = evo_metrics[evo_metrics.evaluations <= 80_000]

		# Calculate statistics
		grouped = evo_metrics.groupby("evaluations")[metric]
		mean = grouped.mean()
		std = grouped.std()
		n = grouped.count()
		se = std / np.sqrt(n)
		ci_lower = mean - (1.96 * se)
		ci_upper = mean + (1.96 * se)

		# Plot mean line and shaded area with higher zorder using colors from evos_colors
		line = axes[plot_idx].plot(mean.index, mean.values, color=evos_colors[evo], zorder=2)
		if plot_idx == 0:
			lines.append(line[0])
			labels.append(evos_dict[evo])
		axes[plot_idx].fill_between(
			mean.index,
			ci_lower.values,
			ci_upper.values,
			color=evos_colors[evo],
			alpha=0.2,
			zorder=1,
		)
		# Use scientific notation for y-axis
		axes[plot_idx].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Add common legend below the plots with reversed labels
fig.legend(lines[::-1], labels[::-1], loc="center", bbox_to_anchor=(0.5, 0.0), ncol=len(evos_dict))

plt.tight_layout()

plt.savefig(f"output/plot_robot_line_{lqd}_{metric}.pdf", dpi=300, bbox_inches="tight")
plt.close()
