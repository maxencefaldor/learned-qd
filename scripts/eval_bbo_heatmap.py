import datetime
import subprocess

from learned_qd.utils.plot import run_paths

evos = ["lqd_q", "lqd_d", "lqd_qd"]
population_size = 128
fn_names = ["rastrigin"]
num_dims = 16

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
hydra_run_dir = f"output/eval_bbo_heatmap/{timestamp}/${{evo.name}}/${{now:%Y-%m-%d_%H%M%S_%f}}"

for evo in evos:
	for fn_name in fn_names:
		args = [
			"python",
			"main_eval.py",
			f"hydra.run.dir={hydra_run_dir}",
			"tags=[eval_bbo_heatmap]",
			"seed=0",
			"evo.name=lqd",
			f"evo.run_path={run_paths[evo]}",
			f"evo.population.max_size={population_size}",
			f"task.fn_names=[{fn_name}]",
			f"task.min_num_dims={num_dims}",
			f"task.max_num_dims={num_dims}",
			"task.x_opt_range=[0.0,0.0]",
			"task.noise_config.noise_model_names=[noiseless]",
			"num_generations=512",
			"log_every=16",
			"log_evolution=false",
			"log_fitness_descriptor=true",
			"wandb=false",
		]
		# Filter out empty strings to avoid subprocess errors
		args = [arg for arg in args if arg]
		subprocess.run(args, check=True)
