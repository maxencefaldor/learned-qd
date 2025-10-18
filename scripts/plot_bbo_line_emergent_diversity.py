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
	fn_names_short_dict,
	metrics_dict,
	run_paths,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rc("font", size=12)

results_dir = "<path>"

run_path = run_paths["lqd_q"]

x_label = "Evaluations"
y_label_fitness = metrics_dict["fitness_max"]
y_label_novelty = metrics_dict["novelty_mean"]

num_evaluations = 8192 * 32
population_size = 128
reproduction_batch_size = 32

num_dims = 16
fn_names = [
	# Part 5: Multi-modal functions with weak global structure
	"gallagher_101_me",
	"gallagher_21_hi",
	# Extra
	"ackley",
	"dixon_price",
	"griewank",
	"levy",
]

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

		if config.task.fn_names[0] not in fn_names:
			continue
		if config.task.max_num_dims != num_dims:
			continue

		metrics = get_metrics(run_dir)

		metrics["evo"] = config.evo.name
		if config.evo.name == "lqd":
			if config.evo.run_path != run_path:
				continue
			metrics["run_path"] = config.evo.run_path
		else:
			metrics["run_path"] = None
		metrics["fn_name"] = config.task.fn_names[0]

		metrics = metrics[
			[
				"evo",
				"batch_id",
				"fn_name",
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

# Create figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(24, 8), squeeze=True)

# Plot each function
lines = []
labels = []
for plot_idx, fn_name in enumerate(fn_names):
	fn_metrics = metrics[metrics.fn_name == fn_name]

	for row in range(2):
		# Customize subplot
		customize_axis(axes[row, plot_idx])
		axes[row, plot_idx].set_xlabel(x_label)
		if plot_idx == 0:  # Only set y label for leftmost subplots
			axes[row, plot_idx].set_ylabel(y_label_fitness if row == 0 else y_label_novelty)
		else:  # Remove y ticks labels for other subplots
			axes[row, plot_idx].set_ylabel("")
		if row == 0:  # Only set title for top row
			axes[row, plot_idx].set_title(fn_names_short_dict[fn_name])
		axes[row, plot_idx].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

		# Plot each algorithm
		for evo in reversed(evos_dict):
			if (
				evo not in fn_metrics.evo.unique()
				or evo == "random"
				or evo == "dns"
				or evo == "ns"
				or evo == "me"
			):
				continue

			evo_metrics = fn_metrics[fn_metrics.evo == evo]

			# Calculate statistics
			metric = "fitness_max" if row == 0 else "novelty_mean"
			grouped = evo_metrics.groupby("evaluations")[metric]
			mean = grouped.mean()
			std = grouped.std()
			n = grouped.count()
			se = std / np.sqrt(n)
			ci_lower = mean - (1.96 * se)
			ci_upper = mean + (1.96 * se)

			# Plot mean line and shaded area
			line = axes[row, plot_idx].plot(
				mean.index, mean.values, color=evos_colors[evo], zorder=2
			)[0]
			axes[row, plot_idx].fill_between(
				mean.index,
				ci_lower.values,
				ci_upper.values,
				color=evos_colors[evo],
				alpha=0.2,
				zorder=1,
			)

			if plot_idx == 0 and row == 0:
				lines.append(line)
				labels.append(evos_dict[evo])

			axes[row, plot_idx].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

# Add common legend below the plots with reversed labels
fig.legend(lines[::-1], labels[::-1], loc="center", bbox_to_anchor=(0.5, 0.0), ncol=len(evos_dict))

plt.tight_layout()

plt.savefig("output/plot_bbo_line_emergent_diversity.pdf", dpi=300, bbox_inches="tight")
plt.close()
