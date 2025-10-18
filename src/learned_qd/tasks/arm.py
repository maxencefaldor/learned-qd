import jax
import jax.numpy as jnp
from flax.struct import dataclass
from qdax.custom_types import Descriptor, Fitness, Genotype

from .task import Task, TaskEval, TaskParams, TaskState


def arm(genotype: Genotype) -> tuple[Fitness, Descriptor]:
	genotype = jnp.clip(genotype, 0.0, 1.0)

	# Fitness is the standard deviation of the angles
	fitness = -jnp.std(genotype)

	# Descriptor is the end-effector position
	size = genotype.shape[0]
	cum_angles = jnp.cumsum(2 * jnp.pi * genotype - jnp.pi)
	x_pos = jnp.sum(jnp.cos(cum_angles)) / (2 * size) + 0.5
	y_pos = jnp.sum(jnp.sin(cum_angles)) / (2 * size) + 0.5

	return fitness, jnp.array([x_pos, y_pos])


@dataclass
class ArmParams(TaskParams):
	pass


@dataclass
class ArmState(TaskState):
	counter: int = 0


@dataclass
class ArmEval(TaskEval):
	fitness: jax.Array
	descriptor: jax.Array


class ArmTask(Task):
	"""Task for the n-joint robotic arm."""

	def __init__(self, num_joints: int = 8, descriptor: str = "position"):
		self.num_joints = num_joints
		self.descriptor = descriptor
		self.x_range = (0.0, 1.0)

	def sample(self, key: jax.Array) -> ArmParams:
		return ArmParams()

	def init(self, key: jax.Array, task_params: ArmParams) -> ArmState:
		return ArmState()

	def evaluate(
		self,
		key: jax.Array,
		x: jax.Array,
		task_state: ArmState,
		task_params: ArmParams,
	) -> tuple[ArmState, ArmEval]:
		"""Evaluate a batch of solutions on the arm task."""
		x = jnp.clip(x, self.x_range[0], self.x_range[1])

		fitness, descriptor = jax.vmap(arm)(x)
		if self.descriptor == "position":
			descriptor = descriptor
		elif self.descriptor == "random":
			key, subkey = jax.random.split(key)
			descriptor = jax.random.normal(subkey, descriptor.shape)
		elif self.descriptor == "zero":
			descriptor = jnp.zeros(descriptor.shape)
		else:
			raise ValueError
		task_eval = ArmEval(fitness=fitness, descriptor=descriptor)

		return task_state.replace(counter=task_state.counter + 1), task_eval

	def sample_x(self, key: jax.Array) -> jax.Array:
		return jax.random.uniform(
			key,
			shape=(self.num_joints,),
			minval=self.x_range[0],
			maxval=self.x_range[1],
		)

	@property
	def descriptor_size(self):
		return 2
