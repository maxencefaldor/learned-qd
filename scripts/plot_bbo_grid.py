import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import fn_names_dict

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

results_dir = "<path>"

x_label = "Population Size"
y_label = "Number of Dimensions"

num_evaluations = 8192 * 32
population_size = 128
reproduction_batch_size = 32

fn_names = [
	# Part 1: Separable functions
	# "sphere",
	# "ellipsoidal",
	# "rastrigin",
	# "bueche_rastrigin",
	# "linear_slope",
	# Part 2: Functions with low or moderate conditions
	# "attractive_sector",
	# "step_ellipsoidal",
	# "rosenbrock",
	# "rosenbrock_rotated",
	# Part 3: Functions with high conditioning and unimodal
	# "ellipsoidal_rotated",
	# "discus",
	# "bent_cigar",
	# "sharp_ridge",
	# "different_powers",
	# Part 4: Multi-modal functions with adequate global structure
	# "rastrigin_rotated",
	# "weierstrass",
	"schaffers_f7",
	# "schaffers_f7_ill_conditioned",
	# "griewank_rosenbrock",
	# Part 5: Multi-modal functions with weak global structure
	# "schwefel",
	"katsuura",
	# "lunacek",
]

# Get metrics for all runs
metrics_list = []
for fn_dir in os.listdir(results_dir):
	fn_dir = os.path.join(results_dir, fn_dir)
	if not os.path.isdir(fn_dir):
		continue

	for evo_dir in os.listdir(fn_dir):
		evo_dir = os.path.join(fn_dir, evo_dir)
		if not os.path.isdir(evo_dir):
			continue

		for run_dir in os.listdir(evo_dir):
			run_dir = os.path.join(evo_dir, run_dir)
			if not os.path.isdir(run_dir):
				continue

			config = get_config(run_dir)

			if config.task.fn_names[0] not in fn_names:
				continue

			metrics = get_metrics(run_dir)

			metrics["evo"] = config.evo.name
			metrics["population_size"] = config.evo.population.max_size
			metrics["fn_name"] = config.task.fn_names[0]
			metrics["num_dims"] = config.task.min_num_dims

			metrics = metrics[
				[
					"evo",
					"batch_id",
					"fn_name",
					"num_dims",
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

# Get last generation for each run
last_gen = metrics.groupby(["evo", "population_size", "fn_name", "num_dims"])[
	"generation"
].transform("max")
last_gen_metrics = metrics[metrics.generation == last_gen]

# Aggregate with median
metrics = (
	last_gen_metrics.groupby(["evo", "population_size", "fn_name", "num_dims"])
	.median()
	.reset_index()
)

# Create figure with 1x2 subplots
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes = axes.flatten()

for idx, fn_name in enumerate(fn_names):
	fn_metrics = metrics[metrics.fn_name == fn_name]

	# Pivot data to create population_size x num_dims grid of fitness differences
	lqd_metrics = fn_metrics[fn_metrics.evo == "lqd"].pivot(
		index="population_size", columns="num_dims", values="fitness_max"
	)
	me_metrics = fn_metrics[fn_metrics.evo == "me"].pivot(
		index="population_size", columns="num_dims", values="fitness_max"
	)

	fitness_diff = lqd_metrics - me_metrics

	# Create heatmap in subplot
	# Find global min/max across all functions for consistent scale
	vmax = fitness_diff.max().max()
	im = axes[idx].imshow(
		fitness_diff.T[::-1],
		aspect="auto",
		cmap="RdBu",  # Red-Blue diverging colormap
		vmin=-vmax,  # Center around 0
		vmax=vmax,
	)

	# Configure axes
	axes[idx].set_xticks(range(len(fitness_diff.index)))
	axes[idx].set_xticklabels(fitness_diff.index, rotation=0)
	axes[idx].set_yticks(range(len(fitness_diff.columns)))
	axes[idx].set_yticklabels(fitness_diff.columns[::-1])

	axes[idx].set_xlabel(x_label)
	if idx == 0:  # Left plot only
		axes[idx].set_ylabel(y_label)

	axes[idx].set_title(fn_names_dict[fn_name])

	# Add colorbar
	if idx == 1:  # Right plot
		plt.colorbar(im, ax=axes[idx], label="LQD - ME Fitness Difference")
	else:
		plt.colorbar(im, ax=axes[idx])

plt.tight_layout()
plt.savefig("output/plot_bbo_grid.pdf", dpi=300, bbox_inches="tight")
plt.close()
