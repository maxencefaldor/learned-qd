import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def plot_evolution(fitness: jax.Array, descriptor: jax.Array, batch_size: int = 0) -> animation.FuncAnimation:
	assert descriptor.shape[-1] == 2

	# Get fitness and descriptor range for the full evolution
	fitness_range = (
		jnp.nanmin(fitness, axis=(0, 1), initial=jnp.inf, where=fitness != -jnp.inf),
		jnp.nanmax(fitness, axis=(0, 1), initial=-jnp.inf, where=fitness != -jnp.inf),
	)
	descriptor_range = jnp.nanmin(descriptor, axis=(0, 1)), jnp.nanmax(descriptor, axis=(0, 1))
	x_margin = 0.05 * (descriptor_range[1][0] - descriptor_range[0][0])
	y_margin = 0.05 * (descriptor_range[1][1] - descriptor_range[0][1])

	def plot_frame(ax, fitness: jax.Array, descriptor: jax.Array) -> animation.FuncAnimation:
		ax.clear()

		ax.set(
			xlim=(descriptor_range[0][0] - x_margin, descriptor_range[1][0] + x_margin),
			ylim=(descriptor_range[0][1] - y_margin, descriptor_range[1][1] + y_margin),
		)

		valid = fitness != -jnp.inf
		valid_fitness = fitness[valid]
		valid_descriptor = descriptor[valid]

		# Get indices of worst solutions
		if batch_size > 0:
			worst_indices = jnp.argsort(valid_fitness)[:batch_size]
			worst_mask = jnp.zeros_like(valid_fitness, dtype=bool).at[worst_indices].set(True)

			# Plot best solutions with black edges
			scatter = ax.scatter(
				valid_descriptor[~worst_mask][:, 0],
				valid_descriptor[~worst_mask][:, 1],
				s=64,
				c=valid_fitness[~worst_mask],
				cmap="viridis",
				edgecolors="black",
				vmin=fitness_range[0],
				vmax=fitness_range[1],
			)

			# Plot worst solutions with red edges
			ax.scatter(
				valid_descriptor[worst_mask][:, 0],
				valid_descriptor[worst_mask][:, 1],
				s=64,
				c=valid_fitness[worst_mask],
				cmap="viridis",
				edgecolors="red",
				vmin=fitness_range[0],
				vmax=fitness_range[1],
			)

		else:
			scatter = ax.scatter(
				valid_descriptor[:, 0],
				valid_descriptor[:, 1],
				s=64,
				c=valid_fitness,
				cmap="viridis",
				edgecolors="black",
				vmin=fitness_range[0],
				vmax=fitness_range[1],
			)
		return scatter

	# Create figure and axis once
	# fig, ax = plt.subplots(figsize=(5, 5))
	fig, ax = plt.subplots()
	plt.axis("equal")  # This will automatically adjust the scaling

	# Frame 0
	ax.set(title="Generation 0")
	scatter = plot_frame(ax, fitness[0], descriptor[0])
	plt.colorbar(scatter, label="Fitness")

	def update(frame):
		ax.set(title=f"Generation {frame}")
		scatter = plot_frame(ax, fitness[frame], descriptor[frame])
		return [scatter]

	# Create animation
	anim = animation.FuncAnimation(
		fig,
		update,
		frames=fitness.shape[0],
		interval=10,  # 50ms between frames
		blit=True,
	)

	plt.tight_layout()
	return anim
