from abc import ABC

import jax


class TaskParams(ABC):
	"""A template of the task parameters."""


class TaskState(ABC):
	"""A template of the task state."""


class TaskEval(ABC):
	"""A template of the task results."""


class Task(ABC):
	"""Abstract class for vectorized tasks."""

	def sample(self, key: jax.Array) -> TaskParams:
		"""Reset the vectorized task.

		Args:
			key: A PRNG key.

		Returns:
			TaskParams: Task parameters.

		"""
		raise NotImplementedError

	def init(self, key: jax.Array, task_params: TaskParams) -> TaskState:
		"""Initialize the state of a vectorized task.

		Args:
			key: A PRNG key.
			task_params: Task parameters.

		Returns:
			TaskState: Initial task state.

		"""
		raise NotImplementedError

	def evaluate(
		self,
		key: jax.Array,
		x: jax.Array,
		task_state: TaskState,
		task_params: TaskParams,
	) -> tuple[TaskState, TaskEval]:
		"""Evaluate a batch of solutions on the task.

		Args:
			key: A PRNG key.
			x: A batch of solutions to evaluate fitness for.
			task_params: Task parameters.

		Returns:
			TaskState: Task state after evaluation.
			TaskEval: Task evaluation results.

		"""
		raise NotImplementedError

	def sample_x(self, key: jax.Array) -> jax.Array:
		raise NotImplementedError

	@property
	def descriptor_size(self):
		raise NotImplementedError
