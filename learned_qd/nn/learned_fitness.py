import jax
import jax.numpy as jnp
from flax import linen as nn


class LearnedFitness(nn.Module):
	num_layers: int
	num_heads: int
	num_features: int
	num_ffn_features: int
	use_layer_norm: bool = True
	use_bias_proj: bool = False
	use_bias_attn: bool = True
	use_bias_ffn: bool = True

	@nn.compact
	def __call__(self, fitness: jax.Array, descriptor: jax.Array) -> jax.Array:
		valid = fitness != -jnp.inf
		mask = valid[:, None] & valid[None, :]  # (N, N)

		x = jnp.concatenate([fitness[:, None], descriptor], axis=-1)

		# Standardize
		x = jax.nn.standardize(x, axis=-2, where=valid[:, None])
		x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

		# Linear Projection
		hidden = nn.Dense(features=self.num_features, use_bias=self.use_bias_proj)(x)

		# Transformer layers
		for i in range(self.num_layers):
			# Self-Attention
			attn = nn.LayerNorm()(hidden) if self.use_layer_norm else hidden
			attn = nn.MultiHeadDotProductAttention(
				num_heads=self.num_heads,
				qkv_features=self.num_features,
				out_features=self.num_features,
				use_bias=self.use_bias_attn,
			)(attn, mask=mask)
			hidden = hidden + attn

			# FFN
			ffn = nn.LayerNorm()(hidden) if self.use_layer_norm else hidden
			ffn = nn.Dense(features=self.num_ffn_features, use_bias=self.use_bias_ffn)(ffn)
			ffn = nn.relu(ffn)
			ffn = nn.Dense(features=self.num_features, use_bias=self.use_bias_ffn)(ffn)
			hidden = hidden + ffn

		# Final layer
		hidden = nn.LayerNorm()(hidden) if self.use_layer_norm else hidden
		hidden = nn.Dense(features=1, use_bias=self.use_bias_ffn)(hidden)

		meta_fitness = jnp.squeeze(hidden, axis=-1)
		meta_fitness = jnp.where(valid, meta_fitness, -jnp.inf)
		return meta_fitness
