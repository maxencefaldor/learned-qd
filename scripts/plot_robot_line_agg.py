import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import customize_axis, evos_colors, evos_dict, metrics_dict, run_paths

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", size=12)

results_dir = "<path>"

metric = "novelty_mean"
lqd = "lqd_q"

run_path = run_paths[lqd]

x_label = "Evaluations"
y_label = metrics_dict[metric] + " Normalized\nRelative to MAP-Elites"

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
			metrics["task"] = config.task.env_name + "_" + config.task.descriptor
		elif config.task._target_ == "learned_qd.tasks.arm.ArmTask":
			continue
			if config.task.num_joints != 8:
				continue

			metrics["env_name"] = "arm"
			if "descriptor" in config.task:
				metrics["descriptor"] = config.task.descriptor
				metrics["task"] = "arm" + "_" + config.task.descriptor
			else:
				metrics["descriptor"] = "position"
				metrics["task"] = "arm" + "_" + "position"

		metrics = metrics[
			[
				"evo",
				"batch_id",
				"env_name",
				"descriptor",
				"task",
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

# Normalize fitness scores per task by ME mean
for task in metrics.task.unique():
	task_metrics = metrics[metrics.task == task]
	me_metrics = task_metrics[task_metrics.evo == "me"]
	me_mean = me_metrics.groupby("evaluations")[metric].mean()
	me_final_mean = me_mean.iloc[-1]
	if "arm" in task and metric == "fitness_max":
		metrics.loc[metrics.task == task, metric] = (
			me_final_mean / metrics[metrics.task == task][metric]
		)
	else:
		metrics.loc[metrics.task == task, metric] = (
			metrics[metrics.task == task][metric] / me_final_mean
		)

# Create figure
fig, ax = plt.subplots(figsize=(5.5, 4))

# Customize plot
customize_axis(ax)
ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_title("Performance Aggregated Across Robot Control Tasks", fontsize=12)
ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

# # Update the legend label for LQD
# evos_dict = {
# 	**evos_dict,  # Keep existing dictionary entries
# 	"lqd": "LQD (ours)",  # Override the LQD entry
# }

# Plot each algorithm
lines = []
labels = []
for evo in evos_dict:
	if evo not in metrics.evo.unique() or evo == "random":
		continue

	evo_metrics = metrics[metrics.evo == evo]

	# Calculate statistics across all tasks
	grouped = evo_metrics.groupby("evaluations")[metric]
	mean = grouped.mean()
	std = grouped.std()
	n = grouped.count()
	se = std / np.sqrt(n)
	ci_lower = mean - (1.96 * se)
	ci_upper = mean + (1.96 * se)

	# Plot mean line and shaded area with higher zorder using colors from evos_colors
	line = ax.plot(mean.index, mean.values, color=evos_colors[evo], zorder=2)
	lines.append(line[0])
	labels.append(evos_dict[evo])
	ax.fill_between(
		mean.index, ci_lower.values, ci_upper.values, color=evos_colors[evo], alpha=0.2, zorder=1
	)

# Add legend
ax.legend(lines, labels)

plt.tight_layout()

plt.savefig(f"output/plot_robot_line_agg_{lqd}_{metric}.svg", dpi=300, bbox_inches="tight")
plt.close()
