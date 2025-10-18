import jax
import jax.numpy as jnp
from flax import linen as nn


class LearnedFitness(nn.Module):
	num_layers: int
	num_heads: int
	num_features: int
	num_ffn_features: int
	use_layer_norm: bool
	use_bias_proj: bool
	use_bias_attn: bool
	use_bias_ffn: bool

	@nn.compact
	def __call__(self, fitness: jax.Array, descriptor: jax.Array) -> jax.Array:
		valid = fitness != -jnp.inf
		mask = valid[:, None] & valid[None, :]  # (N, N)

		# Standardize and project
		descriptor = jax.nn.standardize(descriptor, axis=-2, where=valid[:, None])
		descriptor = jnp.nan_to_num(descriptor, nan=0.0, posinf=0.0, neginf=0.0)
		descriptor_hidden = nn.Dense(features=self.num_features, use_bias=self.use_bias_proj)(
			descriptor
		)

		fitness = jax.nn.standardize(fitness[:, None], axis=-2, where=valid[:, None])
		fitness = jnp.nan_to_num(fitness, nan=0.0, posinf=0.0, neginf=0.0)
		fitness_hidden = nn.Dense(features=self.num_features, use_bias=self.use_bias_proj)(fitness)

		# Transformer layers
		for i in range(self.num_layers):
			# Self-Attention: fitness
			attn = nn.LayerNorm()(fitness_hidden) if self.use_layer_norm else fitness_hidden
			attn = nn.MultiHeadDotProductAttention(
				num_heads=self.num_heads,
				qkv_features=self.num_features,
				out_features=self.num_features,
				use_bias=self.use_bias_attn,
			)(attn, mask=mask)
			fitness_hidden = fitness_hidden + attn

			# FFN
			ffn = nn.LayerNorm()(fitness_hidden) if self.use_layer_norm else fitness_hidden
			ffn = nn.Dense(features=self.num_ffn_features, use_bias=self.use_bias_ffn)(ffn)
			ffn = nn.gelu(ffn)
			ffn = nn.Dense(features=self.num_features, use_bias=self.use_bias_ffn)(ffn)
			fitness_hidden = fitness_hidden + ffn

			# Self-Attention: descriptor
			attn = nn.LayerNorm()(descriptor_hidden) if self.use_layer_norm else descriptor_hidden
			attn = nn.MultiHeadDotProductAttention(
				num_heads=self.num_heads,
				qkv_features=self.num_features,
				out_features=self.num_features,
				use_bias=self.use_bias_attn,
			)(attn, mask=mask)
			descriptor_hidden = descriptor_hidden + attn

			# FFN
			ffn = nn.LayerNorm()(descriptor_hidden) if self.use_layer_norm else descriptor_hidden
			ffn = nn.Dense(features=self.num_ffn_features, use_bias=self.use_bias_ffn)(ffn)
			ffn = nn.gelu(ffn)
			ffn = nn.Dense(features=self.num_features, use_bias=self.use_bias_ffn)(ffn)
			descriptor_hidden = descriptor_hidden + ffn

			# Cross-Attention: fitness (Q) attends to descriptors (K, V)
			q = nn.LayerNorm()(fitness_hidden) if self.use_layer_norm else fitness_hidden
			kv = nn.LayerNorm()(descriptor_hidden) if self.use_layer_norm else descriptor_hidden
			attn = nn.MultiHeadDotProductAttention(
				num_heads=self.num_heads,
				qkv_features=self.num_features,
				out_features=self.num_features,
				use_bias=self.use_bias_attn,
			)(q, kv, mask=mask)
			fitness_hidden = fitness_hidden + attn

			# FFN
			ffn = nn.LayerNorm()(fitness_hidden) if self.use_layer_norm else fitness_hidden
			ffn = nn.Dense(features=self.num_ffn_features, use_bias=self.use_bias_ffn)(ffn)
			ffn = nn.gelu(ffn)
			ffn = nn.Dense(features=self.num_features, use_bias=self.use_bias_ffn)(ffn)
			fitness_hidden = fitness_hidden + ffn

		# Final layer
		fitness_hidden = nn.LayerNorm()(fitness_hidden) if self.use_layer_norm else fitness_hidden
		fitness_hidden = nn.Dense(features=1, use_bias=self.use_bias_ffn)(fitness_hidden)

		meta_fitness = jnp.squeeze(fitness_hidden, axis=-1)
		meta_fitness = jnp.where(valid, meta_fitness, -jnp.inf)
		return meta_fitness
