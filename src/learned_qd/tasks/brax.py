from collections.abc import Callable
from functools import partial
from typing import Any

import brax
import brax.envs
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from .task import Task, TaskEval, TaskParams, TaskState


@dataclass
class BraxParams(TaskParams):
	pass


@dataclass
class BraxState(TaskState):
	counter: int = 0


@dataclass
class BraxEval(TaskEval):
	fitness: jax.Array
	descriptor: jax.Array
	env_states: brax.State


class BraxTask(Task):
	"""Brax Environment Task."""

	def __init__(
		self,
		env_name: str = "walker2d",
		descriptor: str = "feet_contact",
		backend: str = "spring",
		episode_length: int = 1000,
		layer_sizes: tuple[int, ...] = (8,),
		activation: str = "tanh",
	):
		self.env_name = env_name
		self.episode_length = episode_length
		self.x_range = (None, None)

		# Env
		self.env = brax.envs.create(
			env_name=self.env_name,
			backend=backend,
			episode_length=self.episode_length,
			auto_reset=True,
			debug=True,
		)
		self._reset = jax.jit(self.env.reset)
		self._step = jax.jit(self.env.step)

		# Descriptor
		self._descriptor_size = 2
		if descriptor == "feet_contact":
			feet_idx = jnp.array(
				[
					i
					for i, feet_name in enumerate(self.env.sys.link_names)
					if feet_name in FEET_NAMES[env_name]
				]
			)
			if env_name == "ant":
				feet_idx = jnp.take(feet_idx, jnp.array([0, 3]))
			self.get_descriptor = partial(get_feet_contact, feet_idx=feet_idx)
		elif descriptor == "velocity":
			cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
			self.get_descriptor = partial(get_velocity, cog_idx=cog_idx)
		elif descriptor == "feet_contact_velocity":
			feet_idx = jnp.array(
				[
					i
					for i, feet_name in enumerate(self.env.sys.link_names)
					if feet_name in FEET_NAMES[env_name]
				]
			)
			if env_name == "ant":
				feet_idx = jnp.take(feet_idx, jnp.array([0, 3]))
			cog_idx = self.env.sys.link_names.index(COG_NAMES[env_name])
			self.get_descriptor = partial(
				get_feet_contact_velocity, feet_idx=feet_idx, cog_idx=cog_idx
			)
		elif descriptor == "random":
			self.get_descriptor = partial(get_random_descriptor, size=self._descriptor_size)
		elif descriptor == "zero":
			self.get_descriptor = partial(get_zero_descriptor, size=self._descriptor_size)
		else:
			raise NotImplementedError

		# Policy
		activation = nn.tanh if activation == "tanh" else nn.relu
		self.policy = MLP(
			layer_sizes=layer_sizes + (self.env.action_size,),
			activation=activation,
		)

	def sample(self, key: jax.Array) -> BraxParams:
		return BraxParams()

	def init(self, key: jax.Array, task_params: BraxParams) -> BraxState:
		return BraxState()

	def evaluate(
		self,
		key: jax.Array,
		x: jax.Array,
		task_state: BraxState,
		task_params: BraxParams,
	):
		num_x = jax.tree.leaves(x)[0].shape[0]
		keys = jax.random.split(key, num_x)
		task_eval = jax.vmap(self._evaluate)(keys, x)
		return task_state.replace(counter=task_state.counter + 1), task_eval

	def _evaluate(
		self,
		key: jax.Array,
		x: jax.Array,
	) -> tuple[BraxState, BraxEval]:
		env_state_init = self._reset(key)

		def step(carry, _):
			env_state = carry
			action = self.policy.apply(x, env_state.obs)
			next_state = self._step(env_state, action)
			return next_state, next_state

		_, env_states = jax.lax.scan(
			step,
			env_state_init,
			None,
			length=self.episode_length,
		)

		# Get fitness and descriptor
		fitness = self.get_fitness(env_states)
		descriptor = self.get_descriptor(env_states)

		return BraxEval(fitness=fitness, descriptor=descriptor, env_states=env_states)

	def get_fitness(self, env_states: brax.State) -> jax.Array:
		mask = get_mask(env_states)
		return jnp.sum(env_states.reward, axis=-1, where=mask)

	def sample_x(self, key: jax.Array) -> jax.Array:
		key_1, key_2 = jax.random.split(key)
		env_state = self._reset(key_1)
		return self.policy.init(key_2, env_state.obs)

	@property
	def descriptor_size(self):
		return self._descriptor_size


def get_mask(env_states: brax.State) -> jax.Array:
	env_state_done = jnp.roll(env_states.done, shift=1, axis=-1).at[0].set(0.0)
	mask = 1.0 - jnp.clip(jnp.cumsum(env_state_done, axis=-1), 0.0, 1.0)
	return mask


FEET_NAMES = {
	"hopper": ["foot"],
	"walker2d": ["foot", "foot_left"],
	"halfcheetah": ["bfoot", "ffoot"],
	"ant": ["", "", "", ""],
	"humanoid": ["left_shin", "right_shin"],
}


def get_feet_contact(env_states: brax.State, feet_idx: jax.Array) -> jax.Array:
	mask = get_mask(env_states)
	feet_contacts = jnp.any(
		jax.vmap(
			lambda x: (env_states.pipeline_state.contact.link_idx[1] == x)
			& (env_states.pipeline_state.contact.dist <= 0)
		)(feet_idx),
		axis=-1,
	)
	return jnp.mean(feet_contacts, axis=-1, where=mask)


COG_NAMES = {
	"hopper": "torso",
	"walker2d": "torso",
	"halfcheetah": "torso",
	"ant": "torso",
	"humanoid": "torso",
}


def get_position(env_states: brax.State, cog_idx: jax.Array) -> jax.Array:
	mask = get_mask(env_states)
	indicator = mask * env_states.done
	return jnp.sum(
		env_states.pipeline_state.x.pos[..., cog_idx, :2], axis=-2, where=indicator[..., None]
	)


def get_velocity(env_states: brax.State, cog_idx: jax.Array) -> jax.Array:
	mask = get_mask(env_states)
	return jnp.mean(
		env_states.pipeline_state.xd.vel[..., cog_idx, :2], axis=-2, where=mask[..., None]
	)


def get_feet_contact_velocity(
	env_states: brax.State, feet_idx: jax.Array, cog_idx: jax.Array
) -> jax.Array:
	feet_contacts = get_feet_contact(env_states, feet_idx)
	velocity = get_velocity(env_states, cog_idx)
	return jnp.concatenate([feet_contacts, velocity[:1]], axis=-1)


def get_random_descriptor(env_states: brax.State, size: int) -> jax.Array:
	key = jax.random.key(0)
	key = jax.random.fold_in(key, jnp.sum(env_states.pipeline_state.x.pos))
	return jax.random.normal(key, shape=(size,))


def get_zero_descriptor(env_states: brax.State, size: int) -> jax.Array:
	return jnp.zeros((size,))


class MLP(nn.Module):
	"""MLP class."""

	layer_sizes: tuple[int, ...]
	activation: Callable[[jax.Array], jax.Array] = nn.relu
	kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
	final_activation: Callable[[jax.Array], jax.Array] | None = None
	bias: bool = True
	kernel_init_final: Callable[..., Any] | None = None

	@nn.compact
	def __call__(self, obs: jax.Array) -> jax.Array:
		hidden = obs
		for hidden_size in self.layer_sizes[:-1]:
			hidden = nn.Dense(
				hidden_size,
				kernel_init=self.kernel_init,
				use_bias=self.bias,
			)(hidden)
			hidden = self.activation(hidden)

		# Handle final layer separately
		kernel_init = (
			self.kernel_init_final if self.kernel_init_final is not None else self.kernel_init
		)
		hidden = nn.Dense(
			self.layer_sizes[-1],
			kernel_init=kernel_init,
			use_bias=self.bias,
		)(hidden)

		if self.final_activation is not None:
			hidden = self.final_activation(hidden)

		return hidden
