# Learned Quality-Diversity

[![Paper](http://img.shields.io/badge/paper-arxiv.2502.02190-B31B1B.svg)](https://arxiv.org/abs/2502.02190)
[![X](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white&style=flat)](https://x.com/maxencefaldor/status/1907390364249649172)

This repository contains the reference implementation for **Discovering Quality-Diversity Algorithms via Meta-Black-Box Optimization** paper, introducing Learned Quality-Diversity (LQD) a family of meta-optimized evolutionary algorithms designed to efficiently collect stepping stones for open-ended discovery. 🧑‍🔬

## Overview 🔎

LQD introduces a novel approach to Quality-Diversity (QD) optimization by using meta-learning to discover sophisticated competition rules. Unlike traditional QD algorithms that rely on heuristic-based mechanisms (e.g., grid-based competition in MAP-Elites), LQD leverages attention-based neural architectures to parameterize and learn local competition strategies. These strategies are optimized across diverse black-box optimization tasks, resulting in algorithms that excel at balancing fitness, novelty, and diversity.

Key highlights:
- Outperforms or matches established baselines like MAP-Elites, Dominated Novelty Search, Novelty Search, and Genetic Algorithms.
- Demonstrates strong generalization to higher dimensions, larger populations, and out-of-distribution domains like robot control.
- Naturally maintains diverse populations, even when optimized solely for fitness, rediscovering diversity as a key to effective optimization.

## Getting Started 🚦

To explore the code and reproduce the results:

1. **Clone the repository**:
	```
	git clone https://github.com/maxencefaldor/learned-qd.git
	cd learned-qd
	```

2. **Install dependencies**:
	Requires Python 3.10+ and a working JAX installation. Install the required packages:
	```
	pip install -r requirements.txt
	```

3. **Run an experiment**:
	You can run the meta-optimization process on BBOB tasks with the script `main_learn.py`:
	```
	python main_learn.py
	```

4. **Evaluate a method**:
	You can evaluate LQD, or any other QD algorithm with the script `main_eval.py`:

	For example to evaluate MAP-Elites on a BBOB task, you can run:
	```
	python main_eval.py evo=me
	```

	To evaluate a LQD run, you can use:
	```
	python main_eval.py evo=lqd evo.run_path=<path>
	```

To see the full list of options, please look inside the configs folder. For example, you can explore different evolutionary algorithms (in `configs/evo/`) or tasks (in `configs/task/`).

## Repository Structure 📂

Here is an overview of the key directories and files:

```
.
├── README.md
├── learned_qd/                   # Core package containing the LQD implementation.
│   ├── __init__.py
│   ├── evo/                      # Evolutionary algorithm components.
│   │   ├── __init__.py
│   │   ├── evolution.py          # Evolution loop for LQD.
│   │   ├── genetic_algorithm.py  # Base genetic algorithm implementation adapted for QD.
│   │   ├── metrics.py            # Utilities for tracking performance metrics.
│   │   ├── populations/          # Submodules for managing populations
│   │   └── reproductions/        # Submodules for managing reproductions
│   ├── meta/                     # Meta-optimization framework.
│   │   ├── __init__.py
│   │   ├── meta_evaluator.py     # Evaluates LQD variants across tasks.
│   │   ├── meta_evolution.py     # Meta-Evolution loop for LQD (e.g., using Sep-CMA-ES).
│   │   ├── meta_objective.py     # Defines meta-objectives (Fitness, Novelty, QD score).
│   │   └── meta_objective_qd.py
│   ├── nn/                       # Neural network components, including the transformer-based competition function.
│   ├── tasks/                    # Definitions of black-box optimization and robot control tasks.
│   └── utils/                    # Helper functions and utilities.
├── main_eval.py                  # Script to evaluate pre-trained LQD models or QD algorithms on benchmark tasks.
├── main_learn.py                 # Script to run the meta-optimization process and train LQD algorithms.
└── requirements.txt
```

## Using checkpoints

We provide a checkpoint for LQD parameters for each objective (i.e., fitness, novelty, quality-diversity). You can learn about using checkpoints in the notebook `lqd_checkpoints.ipynb`.

## Citation 📝

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{lqd,
	title = {Discovering Quality-Diversity Algorithms via Meta-Black-Box Optimization},
	url = {http://arxiv.org/abs/2502.02190},
	doi = {10.48550/arXiv.2502.02190},
	number = {{arXiv}:2502.02190},
	publisher = {{arXiv}},
	author = {Faldor, Maxence and Lange, Robert Tjarko and Cully, Antoine},
	year = {2025},
	keywords = {Genetic Algorithms, Meta-Learning, Quality-Diversity},
}
```

## Acknowledgments 🙏

This work was conducted in collaboration with [Robert Tjarko Lange](https://github.com/RobertTLange) from [Sakana AI](https://sakana.ai/) and [Antoine Cully](https://github.com/CullyAntoine).

For questions or issues, feel free to open an issue on this repository!
