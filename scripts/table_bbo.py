import os

import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import evos_dict, fn_names_short_dict, run_paths

results_dir = "<path>"

metrics = ["fitness_max", "novelty_mean", "dominated_novelty_mean"]

num_generations = 4096
num_dims = 16
population_size = 128

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
] + [
	# Part 5: Multi-modal functions with weak global structure
	"gallagher_101_me",
	"gallagher_21_hi",
	# Extra
	"ackley",
	"dixon_price",
	"griewank",
	"salomon",
	"levy",
]

# Get metrics for all runs
metrics_list = []
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

		run_metrics = get_metrics(run_dir)

		# Determine the method name
		if config.evo.name == "lqd":
			if config.evo.run_path == run_paths["lqd_q"]:
				method_name = "LQD (Q)"
			elif config.evo.run_path == run_paths["lqd_d"]:
				method_name = "LQD (D)"
			elif config.evo.run_path == run_paths["lqd_qd"]:
				method_name = "LQD (QD)"
			else:
				continue  # Skip unknown LQD variants
		else:
			method_name = evos_dict[config.evo.name]

		run_metrics["method"] = method_name
		run_metrics["fn_name"] = config.task.fn_names[0]

		run_metrics = run_metrics[run_metrics["generation"] <= num_generations][
			[
				"method",
				"batch_id",
				"fn_name",
				"generation",
				"population_size",
				"fitness_max",
				"novelty_mean",
				"dominated_novelty_mean",
			]
		]
		metrics_list.append(run_metrics)

if not metrics_list:
	print("No data found!")
	exit()

all_metrics = pd.concat(metrics_list, ignore_index=True)

# Filter last generation
generation_last = all_metrics.groupby(["method", "fn_name"])["generation"].transform("max")
all_metrics = all_metrics[all_metrics.generation == generation_last]

# Normalize novelty metrics by population size
all_metrics["novelty_mean"] = (
	all_metrics["novelty_mean"] * all_metrics["population_size"] / population_size
)
all_metrics["dominated_novelty_mean"] = (
	all_metrics["dominated_novelty_mean"] * all_metrics["population_size"] / population_size
)

# Get unique function names in the order they appear in fn_names
fn_names_plot = [fn for fn in fn_names if fn in all_metrics["fn_name"].unique()]

# Define method order
method_order = ["LQD (Q)", "LQD (D)", "LQD (QD)", "ME", "DNS", "GA", "NS", "Random"]

# Create tables for each metric
for metric in metrics:
	# Compute stats for this metric
	stats = all_metrics.groupby(["method", "fn_name"])[metric].agg(
		[
			("median", lambda x: x.quantile(0.5)),
			("q1", lambda x: x.quantile(0.25)),
			("q3", lambda x: x.quantile(0.75)),
		]
	)

	# Create table
	table_data = []
	for fn_name in fn_names_plot:
		row = {"Function": fn_names_short_dict.get(fn_name, fn_name)}
		for method in method_order:
			if method in all_metrics["method"].unique() and (method, fn_name) in stats.index:
				median = stats.loc[(method, fn_name), "median"]
				q1 = stats.loc[(method, fn_name), "q1"]
				q3 = stats.loc[(method, fn_name), "q3"]
				row[method] = f"{median:.3f} ({q1:.3f}-{q3:.3f})"
			else:
				row[method] = "N/A"
		table_data.append(row)

	# Convert to DataFrame and display
	table_df = pd.DataFrame(table_data)

	# Save to CSV
	table_df.to_csv(f"output/table_bbo_{metric}.csv", index=False)
