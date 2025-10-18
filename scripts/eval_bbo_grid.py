import datetime
import subprocess

from learned_qd.utils.plot import run_paths

run_path = run_paths["lqd_q"]

population_sizes = [64, 128, 256, 512, 1024]
fn_name = "katsuura"
num_dims_ = [2, 4, 8, 16, 32]

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
hydra_run_dir = (
	f"output/eval_bbo_grid/{fn_name}_{timestamp}/${{evo.name}}/${{now:%Y-%m-%d_%H%M%S_%f}}"
)

# LQD
for population_size in population_sizes:
	for num_dims in num_dims_:
		subprocess.run(
			[
				"python",
				"main_eval.py",
				f"hydra.run.dir={hydra_run_dir}",
				"tags=[eval_bbo_grid]",
				"seed=0",
				"evo=lqd",
				f"evo.run_path={run_path}",
				f"evo.population.max_size={population_size}",
				f"task.fn_names=[{fn_name}]",
				f"task.min_num_dims={num_dims}",
				f"task.max_num_dims={num_dims}",
				"num_generations=32768",
				"log_every=4096",
				"log_evolution=false",
				"wandb=false",
			],
			check=True,
		)

# ME
for population_size in population_sizes:
	for num_dims in num_dims_:
		subprocess.run(
			[
				"python",
				"main_eval.py",
				f"hydra.run.dir={hydra_run_dir}",
				"tags=[eval_bbo_grid]",
				"seed=0",
				"evo=me",
				f"evo.population.max_size={population_size}",
				f"evo.population.descriptor_min={-10 * num_dims**0.5}",
				f"evo.population.descriptor_max={10 * num_dims**0.5}",
				f"task.fn_names=[{fn_name}]",
				f"task.min_num_dims={num_dims}",
				f"task.max_num_dims={num_dims}",
				"num_generations=32768",
				"log_every=4096",
				"log_evolution=false",
				"wandb=false",
			],
			check=True,
		)
