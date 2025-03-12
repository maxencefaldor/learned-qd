import os
import pickle

import wandb
from omegaconf import OmegaConf


def get_config(run_path):
	if os.path.isdir(run_path):
		# Local
		config = OmegaConf.load(os.path.join(run_path, ".hydra", "config.yaml"))
	else:
		# WandB
		api = wandb.Api()
		run = api.run(run_path)
		config = OmegaConf.create(run.config)

	return config


def get_model_path(run_path):
	if os.path.isdir(run_path):
		# Local
		model_path = run_path
	else:
		# WandB
		api = wandb.Api()
		run = api.run(run_path)

		artifacts = run.logged_artifacts()
		model = [artifact for artifact in artifacts if artifact.type == "model"][0]
		model_path = model.download()

	return model_path


def get_config_and_model_path(run_path):
	config = get_config(run_path)
	model_path = get_model_path(run_path)
	return config, model_path


def get_metrics(run_path):
	with open(os.path.join(run_path, "metrics.pickle"), "rb") as metrics_file:
		metrics = pickle.load(metrics_file)
	return metrics
