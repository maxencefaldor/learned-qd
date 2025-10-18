import datetime
import subprocess

from learned_qd.utils.plot import run_paths

evos = ["random", "ga", "ns", "me", "dns", "lqd_q", "lqd_d", "lqd_qd"]
population_size = 128

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

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
hydra_run_dir = f"output/eval_bbo/{timestamp}/${{evo.name}}/${{now:%Y-%m-%d_%H%M%S_%f}}"

for evo in evos:
	for seed, fn_name in enumerate(fn_names):
		args = [
			"python",
			"main_eval.py",
			f"hydra.run.dir={hydra_run_dir}",
			"tags=[eval_bbo]",
			f"seed={seed}",
			f"evo={evo}" if "lqd" not in evo else "evo.name=lqd",
			f"evo.run_path={run_paths[evo]}" if "lqd" in evo else f"evo.name={evo}",
			f"evo.population.max_size={population_size}",
			f"evo.population.descriptor_min={-10 * num_dims**0.5}" if evo == "me" else "",
			f"evo.population.descriptor_max={10 * num_dims**0.5}" if evo == "me" else "",
			f"task.fn_names=[{fn_name}]",
			f"task.min_num_dims={num_dims}",
			f"task.max_num_dims={num_dims}",
			"num_generations=32768",
			"log_every=32768",
			"log_evolution=false",
			"wandb=false",
		]
		# Filter out empty strings to avoid subprocess errors
		args = [arg for arg in args if arg]
		subprocess.run(args, check=True)
