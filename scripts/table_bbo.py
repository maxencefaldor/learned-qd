import os

import numpy as np
import pandas as pd

from learned_qd.utils.helpers import get_config, get_metrics
from learned_qd.utils.plot import evos_dict, fn_names_short_dict, run_paths

results_dir = (
	"/home/maxencefaldor_sakana_ai/projects/learned-qd-private/output/eval_bbo/2025-09-23_215330"
)

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
	# "salomon",
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
base_method_order = ["ME", "DNS", "GA", "NS"]
lqd_variants = {
	"fitness_max": ("LQD (Q)", r"\lqdQ{}"),
	"novelty_mean": ("LQD (D)", r"\lqdD{}"),
	"dominated_novelty_mean": ("LQD (QD)", r"\lqdQD{}"),
}

method_headers = {
	"ME": "ME",
	"DNS": "DNS",
	"GA": "GA",
	"NS": "NS",
}


def format_latex_cell(median, q1, q3, is_best, force_exp=None):
	if force_exp is not None:
		exp = force_exp
	elif median == 0:
		exp = 0
	else:
		exp = int(np.floor(np.log10(abs(median))))

	if abs(exp) < 2:
		# Standard formatting
		if force_exp is not None:
			# If forced, we need to respect the power if it's being used for alignment,
			# but if exp < 2 (e.g. 0 or 1), we usually just show the number.
			# However, if we are anchoring to a small number (e.g. exp=-1) and this number is exp=1,
			# force_exp=-1 would make it exp < 2? No, exp is -1.
			pass

		m_str = f"{median:.2f}"
		if is_best:
			m_str = f"\\textbf{{{m_str}}}"
		res = f"${m_str}\\ ({q1:.2f}, {q3:.2f})$"
	else:
		# Scientific notation
		norm_median = median / (10.0**exp)
		norm_q1 = q1 / (10.0**exp)
		norm_q3 = q3 / (10.0**exp)

		m_str = f"{norm_median:.2f}"
		if is_best:
			m_str = f"\\textbf{{{m_str}}}"

		res = f"${m_str}\\ ({norm_q1:.2f}, {norm_q3:.2f}) \\cdot 10^{{{exp}}}$"

	return res


# Create tables for each metric
for metric in metrics:
	# Determine method order for this metric
	lqd_name, lqd_header = lqd_variants.get(metric, ("LQD (Q)", r"\lqdQ{}"))
	method_order = [lqd_name] + base_method_order

	# Update headers locally
	current_method_headers = method_headers.copy()
	current_method_headers[lqd_name] = lqd_header

	# Compute stats for this metric
	stats = all_metrics.groupby(["method", "fn_name"])[metric].agg(
		[
			("median", lambda x: x.quantile(0.5)),
			("q1", lambda x: x.quantile(0.25)),
			("q3", lambda x: x.quantile(0.75)),
		]
	)

	# Start LaTeX string
	latex_str = (
		r"\begin{sidewaystable}"
		+ "\n"
		+ r"\scriptsize"
		+ "\n"
		+ r"\centering"
		+ "\n"
		+ r"\caption{Comparison of median "
		+ metric.replace("_", " ")
		+ r" values on BBOB functions. Higher values indicate better performance. The best-performing method for each function is highlighted in bold.}"
		+ "\n"
		+ r"\label{tab:"
		+ metric
		+ r"}"
		+ "\n"
		+ r"\begin{tabular}{@{} l"
		+ "l" * len(method_order)
		+ r" @{}}"
		+ "\n"
		+ r"\toprule"
		+ "\n"
		+ r"Function & "
		+ " & ".join([current_method_headers.get(m, m) for m in method_order])
		+ r" \\"
		+ "\n"
		+ r"\midrule"
		+ "\n"
	)

	for fn_name in fn_names_plot:
		row_str = fn_names_short_dict.get(fn_name, fn_name)

		# Find best median for this function among methods in method_order
		best_median = -np.inf
		for method in method_order:
			if (method, fn_name) in stats.index:
				val = stats.loc[(method, fn_name), "median"]
				if val > best_median:
					best_median = val

		# Calculate exponent of best method
		if best_median == 0 or best_median == -np.inf:
			best_exp = 0
		else:
			best_exp = int(np.floor(np.log10(abs(best_median))))

		for method in method_order:
			row_str += " & "
			if (method, fn_name) in stats.index:
				median = stats.loc[(method, fn_name), "median"]
				q1 = stats.loc[(method, fn_name), "q1"]
				q3 = stats.loc[(method, fn_name), "q3"]
				is_best = median == best_median

				# Determine if we should force the exponent
				if median == 0:
					current_exp = 0
				else:
					current_exp = int(np.floor(np.log10(abs(median))))

				# Anchor to best if within 2 orders of magnitude
				if abs(current_exp - best_exp) <= 2:
					force_exp = best_exp
				else:
					force_exp = None

				row_str += format_latex_cell(median, q1, q3, is_best, force_exp=force_exp)
			else:
				row_str += "N/A"

		row_str += r" \\" + "\n"

		# Add midrule after Lunacek (Part 5 end)
		if fn_name == "lunacek":
			row_str += r"\midrule" + "\n"

		latex_str += row_str

	latex_str += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{sidewaystable}" + "\n"

	# Save to TeX
	with open(f"output/table_bbo_{metric}.tex", "w") as f:
		f.write(latex_str)

	print(f"Saved output/table_bbo_{metric}.tex")

	# Create table (CSV legacy support)
	table_data = []
	for fn_name in fn_names_plot:
		row = {"Function": fn_names_short_dict.get(fn_name, fn_name)}
		for method in method_order:
			if (method, fn_name) in stats.index:
				median = stats.loc[(method, fn_name), "median"]
				q1 = stats.loc[(method, fn_name), "q1"]
				q3 = stats.loc[(method, fn_name), "q3"]
				row[method] = f"{median:.2f} ({q1:.2f}-{q3:.2f})"
			else:
				row[method] = "N/A"
		table_data.append(row)

	# Convert to DataFrame and display
	table_df = pd.DataFrame(table_data)

	# Save to CSV
	table_df.to_csv(f"output/table_bbo_{metric}.csv", index=False)
