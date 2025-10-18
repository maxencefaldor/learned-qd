# Learned Quality-Diversity

<div align="center">

[![Paper](http://img.shields.io/badge/paper-arxiv.2502.02190-B31B1B.svg)](https://arxiv.org/abs/2502.02190)
[![X](https://img.shields.io/badge/X-%23000000.svg?style=for-the-badge&logo=X&logoColor=white&style=flat)](https://x.com/maxencefaldor/status/1907390364249649172)

<img width="50%" alt="Banner" src="https://github.com/user-attachments/assets/6d533945-3fad-45f5-ad69-ef78abea0a72" />

</div>

This repository contains the reference implementation for **Discovering Quality-Diversity Algorithms via Meta-Black-Box Optimization** paper, introducing Learned Quality-Diversity (LQD) a family of meta-optimized evolutionary algorithms designed to efficiently collect stepping stones for open-ended discovery. 🧑‍🔬

## Overview 🔎

Quality-Diversity has emerged as a powerful family of evolutionary algorithms that generate diverse populations of high-performing solutions. While these algorithms successfully foster diversity, their mechanisms often rely on heuristics, such as grid-based competition in MAP-Elites. 

This work introduces a fundamentally different approach: using meta-learning to automatically discover novel Quality-Diversity algorithms. By parameterizing the competition rules using attention-based neural architectures, we evolve new algorithms that capture complex relationships between individuals. 

Key highlights of the discovered algorithms:
- They demonstrate competitive or superior performance compared to established Quality-Diversity baselines.
- They exhibit strong generalization to higher dimensions, larger populations, and out-of-distribution domains like robot control.
- Even when optimized solely for fitness, they naturally maintain diverse populations, suggesting that meta-learning rediscovers that diversity is fundamental to effective optimization.

## Getting Started 🚦

To explore the code and reproduce the results:

1. **Clone**:
	```
	git clone https://github.com/maxencefaldor/learned-qd.git
	cd learned-qd
	```

2. **Install**:
	Requires Python 3.13+ and a working JAX installation.

	Create a new virtual environment:
	```
	uv venv
	source .venv/bin/activate
	```

	Install JAX following the [official installation guide](https://github.com/jax-ml/jax?tab=readme-ov-file#installation).

	Install the package:
	```
	uv pip install -e .
	```

3. **Learn**:
	You can run the meta-optimization process on BBOB tasks with the script `main_learn.py`:
	```
	python main_learn.py
	```

4. **Eval**:
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
├── CITATION.cff                  # Citation file for the repository.
├── pyproject.toml                # Project configuration and dependencies.
├── configs/                      # Configuration files.
├── src/learned_qd/               # Core package containing the LQD implementation.
│   ├── es/                       # Evolution Strategies.
│   ├── evo/                      # Evolutionary algorithms.
│   │   ├── evolution.py          # Evolution loop for LQD.
│   │   ├── genetic_algorithm.py  # Base genetic algorithm implementation.
│   │   ├── metrics.py            # Utilities for tracking performance metrics.
│   │   ├── populations/          # Population classes.
│   │   └── reproductions/        # Reproduction classes.
│   ├── meta/                     # Meta-optimization.
│   │   ├── meta_evaluator.py     # Evaluates LQD variants across tasks.
│   │   ├── meta_evolution.py     # Meta-Evolution loop for LQD (e.g., using Sep-CMA-ES or SNES).
│   │   └── meta_objective.py     # Defines meta-objectives (Fitness, Novelty, QD score).
│   ├── nn/                       # Neural network components.
│   ├── tasks/                    # Definitions of black-box optimization and robot control tasks.
│   └── utils/                    # Helper functions and utilities.
├── scripts/                      # Evaluation, analysis scripts.
├── main_eval.py                  # Script to evaluate algorithms on tasks.
└── main_learn.py                 # Script to run the meta-optimization process and train LQD algorithms.
```

## Citation 📝

If you use this software in your research, please cite the paper below.

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
