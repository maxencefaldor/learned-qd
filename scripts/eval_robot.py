import datetime
import subprocess

from learned_qd.utils.plot import run_paths

evos = ["random", "ga", "ns", "me", "dns", "lqd_q", "lqd_d", "lqd_qd"]
num_generations = 4096
population_size = 1024
reproduction_batch_size = 256

# Define environments and their descriptors
env_configs = [
	("hopper", "feet_contact_velocity", "[0.,-5.]", "[1.,5.]"),
	("walker2d", "feet_contact", "0.", "1."),
	("halfcheetah", "feet_contact", "0.", "1."),
	("ant", "velocity", "-5.", "5."),
	("ant", "feet_contact", "0.", "1."),
]

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
hydra_run_dir = f"output/eval_robot/{timestamp}/${{evo.name}}/${{now:%Y-%m-%d_%H%M%S_%f}}"

for evo in evos:
	for seed, (env_name, descriptor, desc_min, desc_max) in enumerate(env_configs):
		args = [
			"python",
			"main_eval.py",
			f"hydra.run.dir={hydra_run_dir}",
			"tags=[eval_robot]",
			f"seed={seed}",
			"task=brax",
			f"task.env_name={env_name}",
			f"task.descriptor={descriptor}",
			f"evo={evo}",
			f"evo.population.max_size={population_size}",
			f"evo.reproduction.batch_size={reproduction_batch_size}",
			f"evo.population.descriptor_min={desc_min}" if evo == "me" else "",
			f"evo.population.descriptor_max={desc_max}" if evo == "me" else "",
			f"evo.run_path={run_paths[evo]}" if evo.startswith("lqd") else "",
			f"num_generations={num_generations}",
			f"log_every={num_generations}",
			"log_evolution=false",
			"wandb=false",
		]
		# Filter out empty strings to avoid subprocess errors
		args = [arg for arg in args if arg]
		subprocess.run(args, check=True)

# Add arm task
for evo in evos:
	args = [
		"python",
		"main_eval.py",
		f"hydra.run.dir={hydra_run_dir}",
		"tags=[eval_robot]",
		"seed=0",
		"task=arm",
		f"evo={evo}",
		f"evo.population.max_size={population_size}",
		f"evo.reproduction.batch_size={reproduction_batch_size}",
		"evo.population.descriptor_min=0." if evo == "me" else "",
		"evo.population.descriptor_max=1." if evo == "me" else "",
		f"evo.run_path={run_paths[evo]}" if evo.startswith("lqd") else "",
		f"num_generations={num_generations}",
		f"log_every={num_generations}",
		"log_evolution=false",
		"wandb=false",
	]
	# Filter out empty strings to avoid subprocess errors
	args = [arg for arg in args if arg]
	subprocess.run(args, check=True)
