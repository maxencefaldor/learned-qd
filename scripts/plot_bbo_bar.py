import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import (
	evos_colors,
	evos_dict,
	fn_names_short_dict,
	metrics_dict,
	run_paths,
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

results_dir = (
	"/home/maxencefaldor_sakana_ai/projects/learned-qd-private/output/eval_bbo/2025-09-23_215330"
)

num_evaluations = 8192 * 32
population_size = 128
reproduction_batch_size = 32

num_dims = 16
fn_names = [
	# Part 1: Separable functions
	"sphere",
	"ellipsoidal",
	"rastrigin",
	"bueche_rastrigin",
	"linear_slope",
	# Part 2: Functions with low or moderate conditions
	"attractive_sector",
	"step_ellipsoidal",
	"rosenbrock",
	"rosenbrock_rotated",
	# Part 3: Functions with high conditioning and unimodal
	"ellipsoidal_rotated",
	"discus",
	"bent_cigar",
	"sharp_ridge",
	"different_powers",
	# Part 4: Multi-modal functions with adequate global structure
	"rastrigin_rotated",
	"weierstrass",
	"schaffers_f7",
	"schaffers_f7_ill_conditioned",
	"griewank_rosenbrock",
	# Part 5: Multi-modal functions with weak global structure
	"schwefel",
	"katsuura",
	"lunacek",
]
# fn_names = [
# 	# Part 5: Multi-modal functions with weak global structure
# 	"gallagher_101_me",
# 	"gallagher_21_hi",
# 	# Extra
# 	"ackley",
# 	"dixon_price",
# 	"salomon",
# 	"levy",
# ]

# Get metrics for all runs (load once for all subplots)
metrics_list = []
all_run_paths = [run_paths["lqd_q"], run_paths["lqd_d"], run_paths["lqd_qd"]]

for evo_dir in os.listdir(results_dir):
	if evo_dir not in evos_dict:
		continue

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
			if config.evo.run_path not in all_run_paths:
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
				"run_path",
			]
		]
		metrics_list.append(metrics)

all_metrics = pd.concat(metrics_list, ignore_index=True)
all_metrics["evaluations"] = all_metrics["generation"] * reproduction_batch_size
all_metrics = all_metrics[all_metrics.evaluations <= num_evaluations]

# Get unique function names that exist in the data
fn_names_plot = [fn for fn in fn_names if fn in all_metrics["fn_name"].unique()]
num_fns = len(fn_names_plot)

# Calculate figure dimensions based on number of functions
# Base width for readability, with additional width for more functions
base_width = 6
width_per_fn = 0.8
fig_width = max(base_width, min(16, base_width + num_fns * width_per_fn))

# Create figure with 3 subplots and shared x axis
fig, axes = plt.subplots(3, 1, figsize=(fig_width, 8), sharex=True)

for subplot_idx, (metric, run_path) in enumerate(
	[
		("fitness_max", run_paths["lqd_q"]),
		("novelty_mean", run_paths["lqd_d"]),
		("dominated_novelty_mean", run_paths["lqd_qd"]),
	]
):
	# Filter metrics for this specific subplot
	metrics = all_metrics.copy()
	if run_path is not None:
		# Filter LQD runs to only include the specific run_path for this metric
		lqd_mask = (metrics["evo"] == "lqd") & (metrics["run_path"] == run_path)
		non_lqd_mask = metrics["evo"] != "lqd"
		metrics = metrics[lqd_mask | non_lqd_mask]

	# Filter last generation
	generation_last = metrics.groupby(["evo", "fn_name"])["generation"].transform("max")
	metrics = metrics[metrics.generation == generation_last][
		[
			"evo",
			"batch_id",
			"fn_name",
			"population_size",
			"fitness_max",
			"novelty_mean",
			"dominated_novelty_mean",
		]
	]

	# Normalize metrics by population size
	metrics["novelty_mean"] = metrics["novelty_mean"] * metrics["population_size"] / population_size
	metrics["dominated_novelty_mean"] = (
		metrics["dominated_novelty_mean"] * metrics["population_size"] / population_size
	)

	# Get random scores for normalization
	random_scores = metrics[metrics.evo == "random"].groupby("fn_name")[metric].median()

	# Normalize scores by random (higher is better)
	if metric in ["novelty_mean", "dominated_novelty_mean"]:
		metrics[metric] = metrics.apply(lambda x: x[metric] / random_scores[x["fn_name"]], axis=1)
	else:
		metrics[metric] = metrics.apply(
			lambda x: random_scores[x["fn_name"]] / (x[metric] + 1e-10), axis=1
		)

	# Compute stats
	stats = metrics.groupby(["evo", "fn_name"])[metric].agg(
		[
			("quartile_1", lambda x: x.quantile(0.25)),
			("quartile_2", lambda x: x.quantile(0.5)),
			("quartile_3", lambda x: x.quantile(0.75)),
		]
	)

	# Get unique evos excluding random, in order of evos_dict
	evos = [evo for evo in evos_dict if evo not in ["random"] and evo in metrics.evo.unique()]
	num_evos = len(evos)

	ax = axes[subplot_idx]

	# Set log scale for y-axis
	ax.set_yscale("log")

	# Adaptive bar width and spacing based on number of functions and evos
	# More functions = narrower bars, fewer functions = wider bars
	max_bar_width = 0.8 / num_evos  # Maximum width per bar
	min_bar_width = 0.05  # Minimum width to maintain readability
	bar_width = max(min_bar_width, min(max_bar_width, 0.15))

	# Adaptive spacing
	spacing = max(0.01, bar_width * 0.1)

	# Calculate positions for each group of bars
	indices = range(num_fns)
	positions = {
		evo: [i + j * (bar_width + spacing) for i in indices] for j, evo in enumerate(evos)
	}

	# Plot bars for each evo algorithm
	for evo in evos:
		# Get data from pre-computed stats
		evo_stats = stats.loc[evo].reindex(fn_names_plot)

		medians = evo_stats["quartile_2"].values
		q1s = evo_stats["quartile_1"].values
		q3s = evo_stats["quartile_3"].values

		# Plot bars with error bars using colors from evos_colors
		ax.bar(
			positions[evo],
			medians,
			bar_width,
			label=evos_dict[evo],
			color=evos_colors[evo],
			yerr=[
				[m - q1 for m, q1 in zip(medians, q1s)],  # Lower errors
				[q3 - m for m, q3 in zip(medians, q3s)],  # Upper errors
			],
			capsize=max(2, min(5, bar_width * 30)),  # Adaptive capsize
		)

	# Add horizontal line at y=1 to show random baseline
	ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Random")

	# Customize plot
	ax.set_ylabel(f"{metrics_dict[metric]} Normalized\nRelative to Random")
	ax.set_xticks([i + (num_evos - 1) * (bar_width + spacing) / 2 for i in indices])

	# Only show x labels on bottom subplot
	if subplot_idx == 2:
		# Adaptive rotation based on number of functions and label length
		avg_label_length = (
			sum(len(fn_names_short_dict.get(fn, fn)) for fn in fn_names_plot) / num_fns
		)
		rotation = 45 if num_fns > 8 or avg_label_length > 8 else 0
		ha = "right" if rotation > 0 else "center"

		ax.set_xticklabels(
			[fn_names_short_dict.get(fn, fn) for fn in fn_names_plot], rotation=rotation, ha=ha
		)
	else:
		ax.set_xticklabels([])

	# Remove top and right spines
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)

	# Add legend in top right (only for first subplot)
	if subplot_idx == 0:
		# Adaptive legend positioning and font size
		legend_fontsize = max(8, min(12, 120 / num_evos))
		ax.legend(loc="upper right", fontsize=legend_fontsize)

plt.tight_layout()
plt.savefig("output/plot_bbo_bar.pdf", dpi=300, bbox_inches="tight")
plt.close()
