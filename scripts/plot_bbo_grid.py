import math
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import fn_names_dict, fn_names_short_dict, metrics_dict, run_paths

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

results_dir = (
	"/home/maxencefaldor_sakana_ai/projects/learned-qd-private/output/eval_bbo_grid_lqd_qd"
)

metric = "dominated_novelty_mean"
lqd = "lqd_qd"

run_path = run_paths[lqd]

x_label = "Population Size"
y_label = "Number of Dimensions"

num_evaluations = 8_192 * 32
reproduction_batch_size = 32

# Get metrics for all runs
metrics_list = []
for fn_dir_name in os.listdir(results_dir):
	fn_dir = os.path.join(results_dir, fn_dir_name)
	if not os.path.isdir(fn_dir):
		continue

	for evo_dir in os.listdir(fn_dir):
		evo_dir_path = os.path.join(fn_dir, evo_dir)
		if not os.path.isdir(evo_dir_path):
			continue

		for run_dir in os.listdir(evo_dir_path):
			run_dir_path = os.path.join(evo_dir_path, run_dir)
			if not os.path.isdir(run_dir_path):
				continue

			config = get_config(run_dir_path)

			# Filter by run_path for lqd
			if config.evo.name == "lqd" and config.evo.run_path != run_path:
				continue

			metrics = get_metrics(run_dir_path)

			metrics["evo"] = config.evo.name
			metrics["population_size_max"] = config.evo.population.max_size
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
					"population_size_max",
					"fitness_max",
					"novelty_mean",
					"dominated_novelty_mean",
				]
			]
			metrics_list.append(metrics)

if not metrics_list:
	print("No metrics found.")
	exit()

metrics = pd.concat(metrics_list, ignore_index=True)
metrics["evaluations"] = metrics["generation"] * reproduction_batch_size
metrics = metrics[metrics.evaluations <= num_evaluations]

# Normalize metrics by population size
metrics["novelty_mean"] = (
	metrics["novelty_mean"] * metrics["population_size"] / metrics["population_size_max"]
)
metrics["dominated_novelty_mean"] = (
	metrics["dominated_novelty_mean"] * metrics["population_size"] / metrics["population_size_max"]
)

# Get last generation for each run
last_gen = metrics.groupby(["evo", "population_size_max", "fn_name", "num_dims"])[
	"generation"
].transform("max")
last_gen_metrics = metrics[metrics.generation == last_gen]

# Aggregate with median
metrics = (
	last_gen_metrics.groupby(["evo", "population_size_max", "fn_name", "num_dims"])
	.median(numeric_only=True)
	.reset_index()
)

# Identify functions present and sort by predefined order in fn_names_dict
present_fns = metrics["fn_name"].unique()
fn_names = [fn for fn in fn_names_dict if fn in present_fns]
# Add any extra functions that might not be in the dict (at the end)
extra_fns = [fn for fn in present_fns if fn not in fn_names]
fn_names.extend(sorted(extra_fns))
num_fns = len(fn_names)

# Calculate grid size for subplots
if num_fns <= 3:
	nrows = 1
	ncols = num_fns
else:
	ncols = 3
	nrows = math.ceil(num_fns / ncols)

# Create figure
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
axes = axes.flatten()

for idx, fn_name in enumerate(fn_names):
	fn_metrics = metrics[metrics.fn_name == fn_name]

	# Pivot data to create population_size x num_dims grid of metric values
	# Check if we have data for both lqd and me
	lqd_df = fn_metrics[fn_metrics.evo == "lqd"]
	me_df = fn_metrics[fn_metrics.evo == "me"]

	if lqd_df.empty or me_df.empty:
		print(f"Skipping {fn_name}: missing data for LQD or ME")
		continue

	lqd_metrics = lqd_df.pivot(index="population_size_max", columns="num_dims", values=metric)
	me_metrics = me_df.pivot(index="population_size_max", columns="num_dims", values=metric)

	diff = (lqd_metrics - me_metrics) / (me_metrics.abs() + 1e-8)

	# Create heatmap in subplot
	# Find range for colormap
	v_min = diff.min().min()
	v_max = diff.max().max()
	v_max = max(abs(v_min), abs(v_max))
	v_min = -v_max

	ticks = None
	if v_min < 0 < v_max:
		norm = TwoSlopeNorm(vmin=v_min, vcenter=0.0, vmax=v_max)
		kwargs = {"norm": norm, "cmap": "RdBu"}

		# Explicitly define ticks to ensure they appear on both sides of 0
		locator = ticker.MaxNLocator(nbins=4)
		neg_ticks = locator.tick_values(v_min, 0)
		neg_ticks = neg_ticks[(neg_ticks >= v_min) & (neg_ticks <= 0)]
		pos_ticks = locator.tick_values(0, v_max)
		pos_ticks = pos_ticks[(pos_ticks >= 0) & (pos_ticks <= v_max)]
		ticks = np.unique(np.concatenate([neg_ticks, pos_ticks]))
	elif v_max <= 0:
		# All negative or zero: map 0 to white (center of RdBu)
		# We use the negative half of RdBu: [0, 0.5]
		# But RdBu 0.5 is white. So we want [v_min, 0] -> [0.0, 0.5]??
		# Actually, if we set vmin=v_min, vmax=-v_min, then 0 is center (white).
		kwargs = {"vmin": v_min, "vmax": -v_min, "cmap": "RdBu"}
	else:
		# All positive or zero: map 0 to white
		# We use the positive half of RdBu: [0.5, 1.0]
		# If we set vmin=-v_max, vmax=v_max, then 0 is center (white).
		kwargs = {"vmin": -v_max, "vmax": v_max, "cmap": "RdBu"}

	im = axes[idx].imshow(
		diff.T[::-1],
		aspect="auto",
		**kwargs,
	)

	# Configure axes
	axes[idx].set_xticks(range(len(diff.index)))
	axes[idx].set_xticklabels(diff.index, rotation=0)
	axes[idx].set_yticks(range(len(diff.columns)))
	axes[idx].set_yticklabels(diff.columns[::-1])

	axes[idx].set_xlabel(x_label)
	if idx % ncols == 0:  # Leftmost column
		axes[idx].set_ylabel(y_label)

	# Use short name if available
	title = fn_names_short_dict.get(fn_name, fn_names_dict.get(fn_name, fn_name))
	axes[idx].set_title(title)

	# Add colorbar
	metric_name = metrics_dict.get(metric, metric)
	plt.colorbar(im, ax=axes[idx], label=f"LQD - ME {metric_name} Relative Diff", ticks=ticks)

# Hide empty subplots
for idx in range(num_fns, len(axes)):
	axes[idx].axis("off")

plt.tight_layout()
plt.savefig(f"output/plot_bbo_grid_{lqd}_{metric}.pdf", dpi=300, bbox_inches="tight")
plt.close()
