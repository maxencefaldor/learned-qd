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

metric = "fitness_max"
lqd = "lqd_q"

run_path = run_paths[lqd]

x_label = "Evaluations"
y_label = metrics_dict[metric]

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
	# "salomon",
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
metrics = metrics[~metrics.evo.isin(["ns"])]

# Normalize metrics by population size
metrics["novelty_mean"] = metrics["novelty_mean"] * metrics["population_size"] / population_size
metrics["dominated_novelty_mean"] = (
	metrics["dominated_novelty_mean"] * metrics["population_size"] / population_size
)

# Filter to only include functions that exist in the data and maintain the order
fn_names_plot = [fn for fn in fn_names if fn in metrics["fn_name"].unique()]
num_fns = len(fn_names_plot)

# Create figure with subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), squeeze=True)
axes = axes.flatten()

# Plot each function
lines = []
labels = []
for plot_idx, fn_name in enumerate(fn_names_plot):
	# Customize subplot
	customize_axis(axes[plot_idx])
	axes[plot_idx].set_xlabel(x_label)
	if plot_idx % 3 == 0:  # Only set y label for leftmost subplots
		axes[plot_idx].set_ylabel(y_label)
	else:  # Remove y ticks labels for other subplots
		axes[plot_idx].set_ylabel("")
	axes[plot_idx].set_title(fn_names_short_dict[fn_name])
	# Use scientific notation for x-axis and y-axis
	axes[plot_idx].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
	axes[plot_idx].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

	fn_metrics = metrics[metrics.fn_name == fn_name]

	# Plot each algorithm
	for evo in evos_dict:
		if evo in ["random"] or evo not in fn_metrics.evo.unique():
			continue

		evo_metrics = fn_metrics[fn_metrics.evo == evo]

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

# Add common legend below the plots with reversed labels
fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.5, 0.0), ncol=len(evos_dict))

plt.tight_layout()

plt.savefig(f"output/plot_bbo_line_{lqd}_{metric}.pdf", dpi=300, bbox_inches="tight")
plt.close()
