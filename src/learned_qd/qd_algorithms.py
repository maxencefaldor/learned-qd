"""Quality-Diversity Algorithms."""

from functools import partial
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
from numpy.random import RandomState
from sklearn.cluster import KMeans

# Types
type Genotype = Any
type Fitness = jax.Array
type Descriptor = jax.Array
type RNGKey = jax.Array
type Centroid = jax.Array


# --- Metrics ---
def novelty_and_dominated_novelty(fitness, descriptor, novelty_k=3, dominated_novelty_k=3):
	valid = fitness != -jnp.inf

	# Neighbors
	neighbor = valid[:, None] & valid[None, :]
	neighbor = jnp.fill_diagonal(neighbor, False, inplace=False)

	# Fitter
	fitter = fitness[:, None] <= fitness[None, :]
	fitter = jnp.where(neighbor, fitter, False)

	# Distance to neighbors
	distance = jnp.linalg.norm(descriptor[:, None, :] - descriptor[None, :, :], axis=-1)
	distance = jnp.where(neighbor, distance, jnp.inf)

	# Distance to fitter neighbors
	distance_fitter = jnp.where(fitter, distance, jnp.inf)

	# Novelty - distance to k-nearest neighbors
	values, indices = jax.vmap(partial(jax.lax.top_k, k=novelty_k))(-distance)
	novelty = jnp.mean(-values, axis=-1, where=jnp.take_along_axis(neighbor, indices, axis=-1))

	# Dominated Novelty - distance to k-fitter-nearest neighbors
	values, indices = jax.vmap(partial(jax.lax.top_k, k=dominated_novelty_k))(-distance_fitter)
	dominated_novelty = jnp.mean(
		-values, axis=-1, where=jnp.take_along_axis(fitter, indices, axis=-1)
	)  # only max fitness individual should be nan

	return novelty, dominated_novelty


def metrics_fn(
	key: RNGKey,
	population: Genotype,
	fitness: Fitness,
	descriptor: Descriptor,
	state: "QDState",
	params: "QDParams",
) -> dict:
	"""Compute QD metrics."""
	k = 3
	novelty, dominated_novelty = novelty_and_dominated_novelty(
		fitness,
		descriptor,
		novelty_k=k,
		dominated_novelty_k=k,
	)
	dominated_novelty = jnp.where(jnp.isposinf(dominated_novelty), jnp.nan, dominated_novelty)

	return {
		"fitness": fitness,
		"descriptor": descriptor,
		"novelty": novelty,
		"dominated_novelty": dominated_novelty,
	}


@jax.jit
def metrics_agg_fn(metrics: dict) -> dict:
	"""Aggregate QD metrics."""
	valid = metrics["fitness"] != -jnp.inf

	descriptor_mean = jnp.mean(metrics["descriptor"], axis=-2, where=valid[..., None])
	distance_to_mean = jnp.linalg.norm(
		metrics["descriptor"] - descriptor_mean[..., None, :], axis=-1
	)
	descriptor_std = jnp.std(distance_to_mean, axis=-1, where=valid)

	return {
		"fitness_max": jnp.max(metrics["fitness"], axis=-1, initial=-jnp.inf, where=valid),
		"fitness_min": jnp.min(metrics["fitness"], axis=-1, initial=jnp.inf, where=valid),
		"fitness_mean": jnp.mean(metrics["fitness"], axis=-1, where=valid),
		"fitness_sum": jnp.sum(metrics["fitness"], axis=-1, where=valid),
		"descriptor_std": descriptor_std,
		"novelty_max": jnp.max(metrics["novelty"], axis=-1, initial=-jnp.inf, where=valid),
		"novelty_min": jnp.min(metrics["novelty"], axis=-1, initial=jnp.inf, where=valid),
		"novelty_mean": jnp.mean(metrics["novelty"], axis=-1, where=valid),
		"novelty_sum": jnp.sum(metrics["novelty"], axis=-1, where=valid),
		"dominated_novelty_max": jnp.nanmax(
			metrics["dominated_novelty"], axis=-1, initial=-jnp.inf, where=valid
		),
		"dominated_novelty_min": jnp.nanmin(
			metrics["dominated_novelty"], axis=-1, initial=jnp.inf, where=valid
		),
		"dominated_novelty_mean": jnp.nanmean(metrics["dominated_novelty"], axis=-1, where=valid),
		"dominated_novelty_sum": jnp.nansum(metrics["dominated_novelty"], axis=-1, where=valid),
		"population_size": jnp.sum(valid, axis=-1),
	}


# --- Grid Helpers ---
def get_centroid_indices(descriptors: Descriptor, centroids: Centroid) -> jax.Array:
	"""Assign descriptors to their closest centroid and return the indices of the centroids."""

	def _get_centroid_indices(descriptor: Descriptor) -> jax.Array:
		return jnp.argmin(jnp.linalg.norm(descriptor - centroids, axis=-1))

	indices = jax.vmap(_get_centroid_indices)(descriptors)
	return indices


def get_centroids(
	num_centroids: int,
	descriptor_size: int,
	descriptor_min: float | list[float],
	descriptor_max: float | list[float],
	num_init_cvt_samples: int,
	key: RNGKey,
) -> jax.Array:
	descriptor_min = jnp.array(descriptor_min)
	descriptor_max = jnp.array(descriptor_max)

	# Sample x uniformly in [0, 1]
	key_x, key_kmeans = jax.random.split(key)
	x = jax.random.uniform(key_x, (num_init_cvt_samples, descriptor_size))

	# Generate an integer seed for RandomState
	seed = jax.random.randint(key_kmeans, (), 0, 2**30, dtype=jnp.int32)

	def _kmeans_host_fn(x_np, seed_np):
		rs = RandomState(int(seed_np))
		kmeans = KMeans(
			init="k-means++",
			n_clusters=num_centroids,
			n_init=1,
			random_state=rs,
		)
		kmeans.fit(x_np)
		return kmeans.cluster_centers_.astype(x_np.dtype)

	# Call host function
	centroids = jax.pure_callback(
		_kmeans_host_fn,
		jax.ShapeDtypeStruct((num_centroids, descriptor_size), x.dtype),
		x,
		seed,
	)

	# Rescale
	return descriptor_min + (descriptor_max - descriptor_min) * centroids


def segment_argmax(data, segment_ids, num_segments):
	return jnp.argmax(
		jax.vmap(lambda i: jnp.where(i == segment_ids, data, -jnp.inf))(jnp.arange(num_segments)),
		axis=1,
	)


# --- Base Algorithm ---
@flax.struct.dataclass
class QDState:
	population: Genotype
	fitness: Fitness
	descriptor: Descriptor
	# Best solution found so far (global)
	best_solution: Genotype
	best_fitness: Fitness
	generation_counter: int


@flax.struct.dataclass
class QDParams:
	mutation_sigma: float = 0.1


# --- Fitness Shaping Functions ---
def update_best_solution_and_fitness(
	population, fitness, best_solution_so_far, best_fitness_so_far
):
	"""Update best solution and fitness so far."""
	idx = jnp.argmax(fitness)
	best_solution_in_population = jax.tree.map(lambda x: x[idx], population)
	best_fitness_in_population = fitness[idx]

	condition = best_fitness_in_population < best_fitness_so_far
	best_solution_so_far = jax.tree.map(
		lambda n, o: jnp.where(condition, o, n),
		best_solution_in_population,
		best_solution_so_far,
	)
	best_fitness_so_far = jnp.where(
		condition,
		best_fitness_so_far,
		best_fitness_in_population,
	)
	return best_solution_so_far, best_fitness_so_far


def random_fitness_shaping(
	key: RNGKey,
	fitness: Fitness,
	descriptor: Descriptor,
	state: QDState,
	params: QDParams,
) -> Fitness:
	"""Random Search: assign random fitness to valid individuals."""
	random_fitness = jax.random.uniform(key, fitness.shape)
	valid = fitness != -jnp.inf
	return jnp.where(valid, random_fitness, -jnp.inf)


def identity_fitness_shaping(
	key: RNGKey,
	fitness: Fitness,
	descriptor: Descriptor,
	state: QDState,
	params: QDParams,
) -> Fitness:
	"""Standard GA: use raw fitness."""
	return fitness


def novelty_fitness_shaping(
	key: RNGKey,
	fitness: Fitness,
	descriptor: Descriptor,
	state: QDState,
	params: QDParams,
	novelty_k: int = 3,
) -> Fitness:
	"""Novelty Search: use novelty score."""
	novelty, _ = novelty_and_dominated_novelty(
		fitness,
		descriptor,
		novelty_k=novelty_k,
	)
	valid = fitness != -jnp.inf
	return jnp.where(valid, novelty, -jnp.inf)


def dominated_novelty_fitness_shaping(
	key: RNGKey,
	fitness: Fitness,
	descriptor: Descriptor,
	state: QDState,
	params: QDParams,
	novelty_k: int = 3,
) -> Fitness:
	"""Dominated Novelty Search: use dominated novelty score."""
	_, dominated_novelty = novelty_and_dominated_novelty(
		fitness,
		descriptor,
		dominated_novelty_k=novelty_k,
	)
	valid = fitness != -jnp.inf
	return jnp.where(valid, dominated_novelty, -jnp.inf)


def map_elites_fitness_shaping(
	key: RNGKey,
	fitness: Fitness,
	descriptor: Descriptor,
	state: QDState,
	params: QDParams,
) -> Fitness:
	"""MAP-Elites: use fitness if best in cell, else -inf."""
	centroids = state.centroids

	# Get centroid assignments
	centroid_indices = get_centroid_indices(descriptor, centroids)
	num_centroids = centroids.shape[0]
	best_index_per_centroid = segment_argmax(fitness, centroid_indices, num_centroids)

	# Check which centroids have assigned individuals
	centroid_assigned = jnp.isin(jnp.arange(num_centroids), centroid_indices)

	# Handle empty centroids to avoid collision at index 0
	best_index_per_centroid = jnp.where(
		centroid_assigned,
		best_index_per_centroid,
		fitness.shape[0],  # if centroid not used, put the best index out of bounds
	)

	# Create mask for individuals that are the best in their assigned cell
	best_index = jnp.zeros_like(fitness, dtype=bool).at[best_index_per_centroid].set(True)

	return jnp.where(best_index, fitness, -jnp.inf)


def gaussian_mutation(key: RNGKey, genotype: Genotype, sigma: float) -> Genotype:
	return jax.tree.map(lambda x: x + sigma * jax.random.normal(key, x.shape), genotype)


class QDAlgorithm:
	def __init__(
		self,
		population_size: int,
		solution: Genotype,
		fitness_shaping_fn=identity_fitness_shaping,
	):
		self.population_size = population_size
		self.solution = solution
		self.fitness_shaping_fn = fitness_shaping_fn
		self.metrics_fn = metrics_fn

	@partial(jax.jit, static_argnames=("self",))
	def init(
		self,
		key: jax.Array,
		population: Genotype,
		fitness: Fitness,
		descriptor: Descriptor,
		params: QDParams,
	) -> QDState:
		"""Initialize evolutionary algorithm."""
		state = self._init(key, params)
		state, _ = self.tell(key, population, fitness, descriptor, state, params)
		return state

	@partial(jax.jit, static_argnames=("self",))
	def ask(
		self,
		key: jax.Array,
		state: QDState,
		params: QDParams,
	) -> tuple[Genotype, QDState]:
		"""Ask evolutionary algorithm for new candidate solutions."""
		return self._ask(key, state, params)

	@property
	def default_params(self) -> QDParams:
		return QDParams()

	def tell(
		self,
		key: RNGKey,
		population: Genotype,
		fitness: Fitness,
		descriptor: Descriptor,
		state: QDState,
		params: QDParams,
	) -> tuple[QDState, dict]:
		"""Tell evolutionary algorithm fitness and descriptors for state update."""
		# Update best solution and fitness
		best_solution, best_fitness = update_best_solution_and_fitness(
			population, fitness, state.best_solution, state.best_fitness
		)

		state = state.replace(
			best_solution=best_solution,
			best_fitness=best_fitness,
		)

		# Concatenate
		all_genotype = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			state.population,
			population,
		)
		all_fitness = jnp.concatenate([state.fitness, fitness], axis=0)
		all_descriptor = jnp.concatenate([state.descriptor, descriptor], axis=0)

		# Compute competition fitness
		key_shaping, key_metrics = jax.random.split(key)
		shaped_fitness = self.fitness_shaping_fn(
			key_shaping,
			all_fitness,
			all_descriptor,
			state,
			params,
		)

		# Sort by competition fitness
		indices = jnp.argsort(shaped_fitness, descending=True)
		indices = indices[: self.population_size]

		# Keep best
		new_genotype = jax.tree.map(lambda x: x[indices], all_genotype)
		new_fitness = all_fitness[indices]
		new_descriptor = all_descriptor[indices]

		# Mark invalid individuals as -inf
		is_valid = shaped_fitness[indices] != -jnp.inf
		new_fitness = jnp.where(is_valid, new_fitness, -jnp.inf)
		new_descriptor = jnp.where(is_valid[:, None], new_descriptor, jnp.nan)

		state = state.replace(
			population=new_genotype,
			fitness=new_fitness,
			descriptor=new_descriptor,
			generation_counter=state.generation_counter + 1,
		)

		# Metrics
		metrics = self.metrics_fn(
			key_metrics, state.population, state.fitness, state.descriptor, state, params
		)
		metrics["generation"] = state.generation_counter
		metrics["best_fitness"] = state.best_fitness

		return state, metrics

	def _init(self, key: RNGKey, params: QDParams) -> QDState:
		raise NotImplementedError

	def _ask(self, key: RNGKey, state: QDState, params: QDParams) -> tuple[Genotype, QDState]:
		"""Common ask method for QD algorithms."""
		# Simple Selection -> Mutation
		valid = state.fitness != -jnp.inf

		p = valid / jnp.sum(valid)
		p = jnp.where(jnp.isnan(p), 1.0 / self.population_size, p)

		population = jax.tree.map(
			lambda x: jax.random.choice(key, x, shape=(self.population_size,), p=p),
			state.population,
		)

		population = gaussian_mutation(key, population, params.mutation_sigma)

		return population, state


# --- Algorithms ---
class GeneticAlgorithm(QDAlgorithm):
	def __init__(
		self,
		population_size: int,
		solution: Genotype,
		fitness_shaping_fn=identity_fitness_shaping,
		descriptor_size: int = 2,
	):
		super().__init__(
			population_size,
			solution,
			fitness_shaping_fn=fitness_shaping_fn,
		)
		self.descriptor_size = descriptor_size

	def _init(self, key: RNGKey, params: QDParams) -> QDState:
		genotype = jax.tree.map(
			lambda x: jnp.full((self.population_size,) + x.shape, fill_value=jnp.nan),
			self.solution,
		)
		fitness = jnp.full((self.population_size,), fill_value=-jnp.inf)
		descriptor = jnp.full((self.population_size, self.descriptor_size), fill_value=jnp.nan)

		# Initialize best_solution structure matching solution template
		best_solution = jax.tree.map(lambda x: jnp.full(x.shape, jnp.nan), self.solution)

		state = QDState(
			population=genotype,
			fitness=fitness,
			descriptor=descriptor,
			best_solution=best_solution,
			best_fitness=-jnp.inf,
			generation_counter=0,
		)
		return state


class NoveltySearch(GeneticAlgorithm):
	def __init__(
		self, population_size: int, solution: Genotype, novelty_k: int = 3, descriptor_size: int = 2
	):
		super().__init__(
			population_size,
			solution,
			fitness_shaping_fn=partial(novelty_fitness_shaping, novelty_k=novelty_k),
			descriptor_size=descriptor_size,
		)


class DominatedNoveltySearch(GeneticAlgorithm):
	def __init__(
		self, population_size: int, solution: Genotype, novelty_k: int = 3, descriptor_size: int = 2
	):
		super().__init__(
			population_size,
			solution,
			fitness_shaping_fn=partial(dominated_novelty_fitness_shaping, novelty_k=novelty_k),
			descriptor_size=descriptor_size,
		)


class RandomSearch(GeneticAlgorithm):
	"""Random Search: replaces individuals randomly."""

	def __init__(self, population_size: int, solution: Genotype, descriptor_size: int = 2):
		super().__init__(
			population_size,
			solution,
			fitness_shaping_fn=random_fitness_shaping,
			descriptor_size=descriptor_size,
		)


@flax.struct.dataclass
class MAPElitesState(QDState):
	centroids: Centroid


class MAPElites(QDAlgorithm):
	def __init__(
		self,
		population_size: int,
		solution: Genotype,
		descriptor_size: int,
		descriptor_min: float | list[float],
		descriptor_max: float | list[float],
		num_init_cvt_samples: int = 10000,
	):
		self.descriptor_size = descriptor_size
		self.descriptor_min = descriptor_min
		self.descriptor_max = descriptor_max
		self.num_init_cvt_samples = num_init_cvt_samples
		super().__init__(population_size, solution, fitness_shaping_fn=map_elites_fitness_shaping)

	def _init(self, key: RNGKey, params: QDParams) -> MAPElitesState:
		genotype = jax.tree.map(
			lambda x: jnp.full((self.population_size,) + x.shape, fill_value=jnp.nan),
			self.solution,
		)
		fitness = jnp.full((self.population_size,), fill_value=-jnp.inf)
		descriptor = jnp.full((self.population_size, self.descriptor_size), fill_value=jnp.nan)

		centroids = get_centroids(
			num_centroids=self.population_size,
			descriptor_size=self.descriptor_size,
			descriptor_min=self.descriptor_min,
			descriptor_max=self.descriptor_max,
			num_init_cvt_samples=self.num_init_cvt_samples,
			key=key,
		)

		# Initialize best_solution structure matching solution template
		best_solution = jax.tree.map(lambda x: jnp.full(x.shape, jnp.nan), self.solution)

		state = MAPElitesState(
			population=genotype,
			fitness=fitness,
			descriptor=descriptor,
			centroids=centroids,
			best_solution=best_solution,
			best_fitness=-jnp.inf,
			generation_counter=0,
		)
		return state


if __name__ == "__main__":
	from learned_qd.tasks.bbob import BBOBTask

	# Configuration
	seed = 42
	pop_size = 1024
	num_generations = 100
	dim = 2

	# Setup Task
	task = BBOBTask(
		min_num_dims=dim,
		max_num_dims=dim,
		fn_names=["sphere"],
		descriptor="gaussian_random_projection",
		descriptor_size=2,
	)

	key = jax.random.key(seed)
	key_task, key_init, key_algo, key_pop = jax.random.split(key, 4)

	task_params = task.sample(key_task)
	task_state = task.init(key_init, task_params)

	# Solution template
	solution_template = jnp.zeros((dim,))

	# Sample initial population from task
	keys = jax.random.split(key_pop, pop_size)
	initial_population = jax.vmap(task.sample_x)(keys)

	# Evaluate initial population to get fitness/descriptor for init
	task_state, task_eval = task.evaluate(key_pop, initial_population, task_state, task_params)

	# Algorithms to test
	algorithms = {
		"Random": RandomSearch(pop_size, solution_template),
		"GA": GeneticAlgorithm(pop_size, solution_template),
		"NoveltySearch": NoveltySearch(pop_size, solution_template),
		"DominatedNoveltySearch": DominatedNoveltySearch(pop_size, solution_template),
		"MAP-Elites": MAPElites(
			pop_size,
			solution_template,
			descriptor_size=2,
			descriptor_min=[-3.0, -3.0],
			descriptor_max=[3.0, 3.0],
		),
	}

	print(f"Starting benchmark on Sphere (dim={dim}) for {num_generations} generations...")

	for name, algo in algorithms.items():
		print(f"\n--- {name} ---")

		# Init Algorithm
		params = algo.default_params

		# Initialize with the sampled population
		state = algo.init(
			key_algo,
			population=initial_population,
			fitness=task_eval.fitness,
			descriptor=task_eval.descriptor,
			params=params,
		)

		# Loop
		curr_task_state = task_state

		for gen in range(num_generations):
			key_algo, key_ask, key_eval, key_tell = jax.random.split(key_algo, 4)

			# Ask
			population, state = algo.ask(key_ask, state, params)

			# Evaluate
			curr_task_state, task_eval = task.evaluate(
				key_eval, population, curr_task_state, task_params
			)

			# Tell
			state, metrics = algo.tell(
				key_tell, population, task_eval.fitness, task_eval.descriptor, state, params
			)

			if gen % 20 == 0:
				# Aggregate metrics for display
				agg = metrics_agg_fn(metrics)
				print(
					f"Generation {gen:03d}: "
					f"population_size={agg['population_size']:.0f}, "
					f"fitness_max={agg['fitness_max']:.4f}, "
					f"novelty_mean={agg['novelty_mean']:.4f}, "
					f"dominated_novelty_mean={agg['dominated_novelty_mean']:.4f}"
				)

		print(f"Final Best Fitness: {state.best_fitness:.4f}")
