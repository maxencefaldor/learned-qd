"""Blackbox Optimization Benchmarking Task."""

import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .bbob_fn import bbob_fns
from .bbob_noise import NoiseModel, NoiseParams
from .extra_fn import extra_fns
from .task import Task, TaskEval, TaskParams, TaskState

all_fns = bbob_fns | extra_fns


@dataclass
class BBOBParams(TaskParams):
	fn_id: jax.Array
	num_dims: jax.Array
	x_opt: jax.Array
	f_opt: jax.Array
	descriptor_params: jax.Array
	noise_params: NoiseParams


@dataclass
class BBOBState(TaskState):
	r: jax.Array
	q: jax.Array
	counter: int = 0


@dataclass
class BBOBEval(TaskEval):
	fitness: jax.Array
	descriptor: jax.Array


class BBOBTask(Task):
	"""Blackbox Optimization Benchmarking Task class."""

	def __init__(
		self,
		min_num_dims: int = 2,
		max_num_dims: int = 10,
		descriptor: str = "gaussian_random_projection",
		descriptor_size: int = 2,
		fn_names: list[str] = ["sphere"],
		x_range: list[float] = [-5.0, 5.0],
		x_opt_range: list[float] = [-4.0, 4.0],
		f_opt_range: list[float] = [0.0, 0.0],
		clip_x: bool = False,
		sample_rotation: bool = False,
		noise_config: dict = {
			"noise_models": ["noiseless", "gaussian", "uniform", "cauchy", "additive"],
			"use_stabilization": True,
		},
	):
		self.min_num_dims = min_num_dims
		self.max_num_dims = max_num_dims
		self.x_range = x_range
		self.x_opt_range = x_opt_range
		self.f_opt_range = f_opt_range
		self.clip_x = clip_x
		self.sample_rotation = sample_rotation

		# Collect active BBOB functions
		self.fn_ids, self.fns, counter = [], [], 0
		for fn_name, fn in all_fns.items():
			if fn_name in fn_names:
				self.fn_ids.append(counter)
				self.fns.append(jax.vmap(fn, in_axes=(0, None, None, None, None)))
				counter += 1
		self.fn_ids = jnp.array(self.fn_ids)

		# Descriptor
		self.descriptor = descriptor
		self._descriptor_size = descriptor_size

		# Noise
		self.noise_model = NoiseModel(**noise_config)

	def sample(self, key: jax.Array) -> BBOBParams:
		"""Sample BBOB task parameters."""
		key_fn, key_d, key_x, key_f, key_noise, key_desc = jax.random.split(key, 6)

		fn_id = jax.random.choice(key_fn, self.fn_ids)
		num_dims = jax.random.randint(key_d, (), minval=self.min_num_dims, maxval=self.max_num_dims)

		x_opt = jax.random.uniform(
			key_x,
			shape=(self.max_num_dims,),
			minval=self.x_opt_range[0],
			maxval=self.x_opt_range[1],
		)
		f_opt = jax.random.uniform(
			key_f,
			minval=self.f_opt_range[0],
			maxval=self.f_opt_range[1],
		)

		# Sample noise model parameters
		noise_params = self.noise_model.sample(key_noise)

		# Descriptor
		if self.descriptor == "gaussian_random_projection":
			descriptor_params = self.gaussian_random_projection(key_desc, num_dims)
		elif self.descriptor == "random_index":
			descriptor_params = self.random_index(key_desc, num_dims)
		else:
			raise NotImplementedError

		return BBOBParams(fn_id, num_dims, x_opt, f_opt, descriptor_params, noise_params)

	def init(self, key: jax.Array, task_params: BBOBParams) -> BBOBState:
		if self.sample_rotation:
			key_r, key_q = jax.random.split(key)
			r = self.generate_random_rotation(key_r, self.max_num_dims)
			q = self.generate_random_rotation(key_q, self.max_num_dims)
		else:
			r = jnp.eye(self.max_num_dims)
			q = jnp.eye(self.max_num_dims)
		return BBOBState(counter=0, r=r, q=q)

	def evaluate(
		self,
		key: jax.Array,
		x: jax.Array,
		task_state: BBOBState,
		task_params: BBOBParams,
	) -> tuple[BBOBState, BBOBEval]:
		if self.clip_x:
			x = jnp.clip(x, self.x_range[0], self.x_range[1])

		fn_val, fn_pen = jax.lax.switch(
			task_params.fn_id,
			self.fns,
			x,
			task_params.x_opt,
			task_state.r,
			task_state.q,
			task_params.num_dims,
		)

		# Apply noise
		fn_noise = self.noise_model.apply(key, fn_val, task_params.noise_params)

		# Add boundary handling penalty and optimal function value
		fn_val = fn_noise + fn_pen + task_params.f_opt

		# Descriptor
		descriptor = jax.vmap(lambda x_i: task_params.descriptor_params @ x_i)(x)

		task_eval = BBOBEval(fitness=-fn_val, descriptor=descriptor)
		return task_state.replace(counter=task_state.counter + 1), task_eval

	def sample_x(self, key: jax.Array) -> jax.Array:
		return jax.random.uniform(
			key,
			shape=(self.max_num_dims,),
			minval=self.x_range[0],
			maxval=self.x_range[1],
		)

	@property
	def descriptor_size(self):
		return self._descriptor_size

	def generate_random_rotation(self, key: jax.Array, num_dims: int) -> jax.Array:
		"""Generate a random (n, n) rotation matrix uniformly sampled from SO(n).

		This implementation follows the method described in:
		"How to generate a random unitary matrix" [Maris Ozols 2006]
		http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
		https://github.com/alecjacobson/gptoolbox/blob/master/matrix/rand_rotation.m

		Uses a fixed-size matrix of max_num_dims and masks the extra dimensions to handle
		variable num_dims while remaining jit-compatible.
		"""
		# Generate fixed-size random normal matrix but mask based on num_dims
		random_matrix = jax.random.normal(key, (self.max_num_dims, self.max_num_dims))
		mask = (jnp.arange(self.max_num_dims)[:, None] < num_dims) & (
			jnp.arange(self.max_num_dims)[None, :] < num_dims
		)
		random_matrix = jnp.where(mask, random_matrix, 0.0)

		# Add identity matrix for masked region to ensure valid QR decomposition
		random_matrix = random_matrix + jnp.where(~mask, jnp.eye(self.max_num_dims), 0.0)

		# QR decomposition
		orthogonal_matrix, upper_triangular = jnp.linalg.qr(random_matrix)

		# Extract diagonal and create sign correction matrix
		diagonal = jnp.diag(upper_triangular)
		sign_correction = jnp.diag(diagonal / jnp.abs(diagonal))

		# Apply sign correction
		rotation = orthogonal_matrix @ sign_correction

		# Ensure determinant is 1 by possibly flipping first row
		determinant = jnp.linalg.det(rotation)
		rotation = rotation.at[0].multiply(determinant)

		return rotation

	def gaussian_random_projection(self, key: jax.Array, num_dims: int) -> jax.Array:
		descriptor_params = jax.random.normal(
			key,
			shape=(self.descriptor_size, self.max_num_dims),
		) / jnp.sqrt(self.descriptor_size)
		mask = jnp.arange(self.max_num_dims) < num_dims
		descriptor_params = jnp.where(mask, descriptor_params, 0)
		return descriptor_params
