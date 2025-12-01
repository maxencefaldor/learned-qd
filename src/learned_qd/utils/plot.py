import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

evos_dict = {
	"lqd": "LQD",
	"lqd_q": "LQD (Q)",
	"lqd_d": "LQD (D)",
	"lqd_qd": "LQD (QD)",
	"me": "ME",
	"dns": "DNS",
	"ga": "GA",
	"ns": "NS",
	"cma_es": "CMA-ES",
	"random": "Random",
}

# Replace with actual run IDs
run_paths = {
	"lqd_q": "meta-synth-env/Learned-QD/payvld97",
	"lqd_d": "meta-synth-env/Learned-QD/jrtm98xf",
	"lqd_qd": "meta-synth-env/Learned-QD/9ycq2u3q",
}

metrics_dict = {
	"fitness_max": "Fitness",
	"novelty_mean": "Novelty",
	"dominated_novelty_mean": "QD Score",
}

# Get a colorblind-friendly palette for the number of algorithms
palette = sns.color_palette(
	"colorblind", n_colors=len(["lqd", "me", "dns", "ga", "ns", "cma_es", "random"])
)

# Create dictionary mapping algorithm names to colors
evos_colors = {
	"lqd": palette[0],
	"lqd_q": palette[0],
	"lqd_d": palette[0],
	"lqd_qd": palette[0],
	"me": palette[1],
	"dns": palette[2],
	"ga": palette[3],
	"ns": palette[4],
	"cma_es": palette[6],
	"random": palette[5],
}

fn_names_dict = {
	# Part 1: Separable functions
	"sphere": "Sphere Function",
	"ellipsoidal": "Ellipsoidal Function",
	"rastrigin": "Rastrigin Function",
	"bueche_rastrigin": "Büche-Rastrigin Function",
	"linear_slope": "Linear Slope",
	# Part 2: Functions with low or moderate conditioning
	"attractive_sector": "Attractive Sector Function",
	"step_ellipsoidal": "Step Ellipsoidal Function",
	"rosenbrock": "Rosenbrock Function",
	"rosenbrock_rotated": "Rosenbrock Function, rotated",
	# Part 3: Functions with high conditioning and unimodal
	"ellipsoidal_rotated": "Ellipsoidal Function, rotated",
	"discus": "Discus Function",
	"bent_cigar": "Bent Cigar Function",
	"sharp_ridge": "Sharp Ridge Function",
	"different_powers": "Different Powers Function",
	# Part 4: Multi-modal functions with adequate global structure
	"rastrigin_rotated": "Rastrigin Function",
	"weierstrass": "Weierstrass Function",
	"schaffers_f7": "Schaffers F7 Function",
	"schaffers_f7_ill_conditioned": "Schaffers F7 Function, moderately ill-conditioned",
	"griewank_rosenbrock": "Composite Griewank-Rosenbrock Function F8F2",
	# Part 5: Multi-modal functions with weak global structure
	"schwefel": "Schwefel Function",
	"gallagher_101_me": "Gallagher's Gaussian 101-me Peaks Function",
	"gallagher_21_hi": "Gallagher's Gaussian 21-hi Peaks Function",
	"katsuura": "Katsuura Function",
	"lunacek": "Lunacek bi-Rastrigin Function",
	# Extra
	"ackley": "Ackley Function",
	"dixon_price": "Dixon-Price Function",
	"griewank": "Griewank Function",
	"salomon": "Salomon Function",
	"levy": "Levy Function",
}

fn_names_short_dict = {
	# Part 1: Separable functions
	"sphere": "Sphere",
	"ellipsoidal": "Ellipsoidal",
	"rastrigin": "Rastrigin",
	"bueche_rastrigin": "Büche-Rastrigin",
	"linear_slope": "Linear Slope",
	# Part 2: Functions with low or moderate conditioning
	"attractive_sector": "Attractive Sector",
	"step_ellipsoidal": "Step Ellipsoidal",
	"rosenbrock": "Rosenbrock",
	"rosenbrock_rotated": "Rosenbrock Rotated",
	# Part 3: Functions with high conditioning and unimodal
	"ellipsoidal_rotated": "Ellipsoidal Rotated",
	"discus": "Discus",
	"bent_cigar": "Bent Cigar",
	"sharp_ridge": "Sharp Ridge",
	"different_powers": "Different Powers",
	# Part 4: Multi-modal functions with adequate global structure
	"rastrigin_rotated": "Rastrigin Rotated",
	"weierstrass": "Weierstrass",
	"schaffers_f7": "Schaffers F7",
	"schaffers_f7_ill_conditioned": "Schaffers F7 Ill-cond.",
	"griewank_rosenbrock": "Griewank-Rosenbrock",
	# Part 5: Multi-modal functions with weak global structure
	"schwefel": "Schwefel",
	"gallagher_101_me": "Gallagher 101-me",
	"gallagher_21_hi": "Gallagher 21-hi",
	"katsuura": "Katsuura",
	"lunacek": "Lunacek",
	# Extra
	"exponential": "Exponential",
	"ackley": "Ackley",
	"dixon_price": "Dixon-Price",
	"griewank": "Griewank",
	"salomon": "Salomon",
	"levy": "Levy",
}

robot_task = {
	"hopper_feet_contact_velocity": "Hopper - Velocity Feet Contact",
	"walker2d_feet_contact": "Walker2d - Feet Contact",
	"halfcheetah_feet_contact": "Half Cheetah - Feet Contact",
	"ant_feet_contact": "Ant - Feet Contact",
	"ant_velocity": "Ant - Velocity",
	"arm": "Arm",
}


def customize_axis(ax):
	# Remove spines
	ax.spines["top"].set_visible(False)
	ax.spines["right"].set_visible(False)
	# Add grid behind the plot
	ax.grid(which="major", axis="y", color="0.9")
	# Set the grid to be behind by setting the axis zorder higher
	ax.set_axisbelow(True)
	return ax


def plot_evolution(
	fitness: jax.Array, descriptor: jax.Array, batch_size: int = 0
) -> animation.FuncAnimation:
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
