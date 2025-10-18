import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import evos_colors, evos_dict, run_paths

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

results_dir = "<path>"

num_evaluations = 1_000_000
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
# 	# "griewank",
# 	"salomon",
# 	"levy",
# ]

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

		if config.evo.name == "lqd":
			if config.evo.run_path == run_paths["lqd_q"]:
				metrics["evo"] = config.evo.name + "_" + "q"
			elif config.evo.run_path == run_paths["lqd_d"]:
				metrics["evo"] = config.evo.name + "_" + "d"
			elif config.evo.run_path == run_paths["lqd_qd"]:
				metrics["evo"] = config.evo.name + "_" + "qd"
			else:
				continue
		elif config.evo.name == "random":
			continue
		else:
			metrics["evo"] = config.evo.name
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

# Filter last generation
generation_last = metrics.groupby(["evo", "fn_name"])["generation"].transform("max")
metrics = metrics[metrics.generation == generation_last][
	["evo", "batch_id", "fn_name", "population_size", "fitness_max", "novelty_mean"]
]

# Get ME fitness_max for normalization
me_scores = metrics[metrics.evo == "me"].groupby("fn_name")["fitness_max"].median()

# Normalize fitness_max by ME and invert
metrics["fitness_max"] = metrics.apply(
	lambda x: me_scores[x["fn_name"]] / (x["fitness_max"] + 1e-8), axis=1
)

# Normalize novelty_mean by population size
metrics["novelty_mean_normalized"] = (
	metrics["novelty_mean"] * metrics["population_size"] / population_size
)

# Get ME novelty_mean for normalization
me_scores = metrics[metrics.evo == "me"].groupby("fn_name")["novelty_mean_normalized"].median()

# Normalize novelty_mean by ME
metrics["novelty_mean"] = metrics.apply(
	lambda x: x["novelty_mean_normalized"] / me_scores[x["fn_name"]], axis=1
)

# Compute median fitness and novelty for each evo
medians = (
	metrics.groupby("evo").agg({"fitness_max": "median", "novelty_mean": "median"}).reset_index()
)

# Create figure
fig, ax = plt.subplots(figsize=(6, 4))

# Add reference lines for ME point
me_row = medians[medians.evo == "me"].iloc[0]

# Get min/max values for x and y
x_min = medians["fitness_max"].min()
y_min = medians["novelty_mean"].min()

# Draw reference lines behind points
ax.plot(
	[x_min, me_row["fitness_max"]],
	[me_row["novelty_mean"], me_row["novelty_mean"]],
	color="gray",
	linestyle="--",
	alpha=0.5,
	zorder=1,
)

# Draw vertical line from min y to ME point
ax.plot(
	[me_row["fitness_max"], me_row["fitness_max"]],
	[y_min, me_row["novelty_mean"]],
	color="gray",
	linestyle="--",
	alpha=0.5,
	zorder=1,
)

# Plot points for each evo algorithm
for _, row in medians.iterrows():
	evo = row["evo"]
	color = evos_colors["lqd"] if evo.startswith("lqd") else evos_colors[evo]
	ax.scatter(
		row["fitness_max"], row["novelty_mean"], label=evos_dict[evo], color=color, s=100, zorder=2
	)

# Define custom offsets for each algorithm
label_offsets = {
	"me": (5, 5),
	"lqd_q": (5, 5),
	"lqd_d": (5, 5),
	"lqd_qd": (5, 5),
	"dns": (5, 5),
	"ga": (5, 5),
	"ns": (5, 5),
}

# Add labels for each point
for _, row in medians.iterrows():
	evo = row["evo"]
	offset = label_offsets.get(evo, (5, 5))  # Default offset if not specified
	ax.annotate(
		evos_dict[row["evo"]],
		(row["fitness_max"], row["novelty_mean"]),
		xytext=offset,
		textcoords="offset points",
	)

# Set log scale for both axes
ax.set_xscale("log")
ax.set_yscale("log")

# Customize plot
ax.set_xlabel("Fitness Normalized\nRelative to ME")
ax.set_ylabel("Novelty Normalized\nRelative to ME")

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("output/plot_bbo_pareto.pdf", dpi=300, bbox_inches="tight")
plt.close()
