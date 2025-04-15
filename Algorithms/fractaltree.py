import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import to_rgba, ListedColormap, LinearSegmentedColormap
import math
import shutil
from tqdm import tqdm
import time


INITIAL_LENGTH = 0.8
INITIAL_THICKNESS = 0.15
MAX_LEVEL = 8
BRANCHING_ANGLE = 25.0
NUM_BRANCHES = 3
LENGTH_SCALE = 0.65
THICKNESS_SCALE = 0.75
LEAF_LEVEL_START = 5
SPHERE_DENSITY = 35
MIN_LENGTH_STOP = 0.015
ANGLE_VARIATION = 8.0
DIRECTION_VARIATION = 0.1


TRUNK_COLOR_BASE = '#8B4513'
LEAF_COLOR_BASE = '#228B22'
COLOR_VARIATION = 0.15


FLOOR_SIZE = 1.2
TERRAIN_GRID_POINTS = 50
TERRAIN_NOISE_FACTOR = 0.05
TERRAIN_COLOR_DIRT = '#A0522D'
TERRAIN_COLOR_GRASS_LOW = '#6B8E23'
TERRAIN_COLOR_GRASS_HIGH = '#556B2F'
TERRAIN_Z_OFFSET = -0.05

STAR_COUNT = 200
STAR_SIZE_MIN = 0.5
STAR_SIZE_MAX = 2.0
STAR_ALPHA_MIN = 0.3
STAR_ALPHA_MAX = 0.7


PARTICLE_EFFECT_ENABLED = True
PARTICLES_PER_BURST = 18
PARTICLE_LIFESPAN_FRAMES = 12
PARTICLE_START_SIZE = 35
PARTICLE_END_SIZE = 1
PARTICLE_BASE_COLORS = ['#32CD32', '#7CFC00', '#00FF00', '#ADFF2F']
PARTICLE_BURST_SPEED = 0.06
PARTICLE_BURST_SPEED_VARIATION = 0.03
PARTICLE_ALPHA_START = 0.9
PARTICLE_ALPHA_END = 0.0
PARTICLE_BIRTH_GLOW_FRAMES = 3


FPS = 30
DURATION_GROWTH_SEC = 25
DURATION_ROTATE_SEC = 10
FRAMES_GROWTH = DURATION_GROWTH_SEC * FPS
FRAMES_ROTATE = DURATION_ROTATE_SEC * FPS
TOTAL_FRAMES = FRAMES_GROWTH + FRAMES_ROTATE


FIG_WIDTH_INCHES = 8
FIG_HEIGHT_INCHES = 10
OUTPUT_FILENAME = "fractal_tree_burst_particles_solid.mp4"
VIDEO_DPI = 150
VIEW_ELEV_START = 25
VIEW_AZIM_START = -70
AZIM_DRIFT_SPEED = 180 / (DURATION_GROWTH_SEC + DURATION_ROTATE_SEC)
ELEV_OSCILLATION = 5
ELEV_OSCILLATION_SPEED = 0.5


prev_last_idx = -1
particles = []


def rotate_vector(vector, axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    return vector * cos_a + \
           np.cross(axis, vector) * sin_a + \
           axis * np.dot(axis, vector) * (1 - cos_a)

def get_perpendicular_vector(vector):
    norm_vec = vector / np.linalg.norm(vector)
    if np.abs(norm_vec[2]) < 0.99:
        perp = np.cross(norm_vec, [0, 0, 1])
    else:
        perp = np.cross(norm_vec, [0, 1, 0])
    if np.linalg.norm(perp) < 1e-9:
        perp = np.cross(norm_vec, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-9:
            perp = np.array([1.0, 0.0, 0.0]) if abs(norm_vec[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    return perp / np.linalg.norm(perp)


def vary_color(hex_color, variation):
    rgba = np.array(to_rgba(hex_color))
    noise = np.random.uniform(-variation, variation, 3)
    rgba[:3] = np.clip(rgba[:3] + noise, 0, 1)
    return rgba

def generate_background_stars(n_stars, limit):
    outer_limit = limit * 3.0
    positions = np.random.uniform(-outer_limit, outer_limit, size=(n_stars, 3))
    dist_sq = np.sum(positions**2, axis=1)
    mask_close = dist_sq < (limit * 1.5)**2
    positions[mask_close] *= 2

    sizes = np.random.uniform(STAR_SIZE_MIN, STAR_SIZE_MAX, size=n_stars)
    alphas = np.random.uniform(STAR_ALPHA_MIN, STAR_ALPHA_MAX, size=n_stars)
    base_colors = np.random.choice(['#ffffff', '#f0f8ff', '#e6e6fa', '#b0c4de'], size=n_stars)
    rgba_colors = np.zeros((n_stars, 4))
    for i in range(n_stars):
        rgba_colors[i] = to_rgba(base_colors[i], alpha=alphas[i])
    return positions, sizes, rgba_colors


def generate_random_unit_vectors(count):
    vectors = np.random.randn(count, 3)
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    vectors = vectors / norms[:, np.newaxis]
    return vectors


def create_fractal_tree_recursive_sequential(start_point, direction, length, thickness, level, sphere_data):
    if level > MAX_LEVEL or length < MIN_LENGTH_STOP:
        return

    direction = direction / np.linalg.norm(direction)
    end_point = start_point + direction * length
    is_leaf = level >= LEAF_LEVEL_START
    base_color = LEAF_COLOR_BASE if is_leaf else TRUNK_COLOR_BASE

    num_spheres_on_branch = max(2, int(length * SPHERE_DENSITY))
    t_values = np.linspace(0, 1, num_spheres_on_branch)
    branch_points = start_point + t_values[:, np.newaxis] * (end_point - start_point)
    branch_thicknesses = np.linspace(thickness, thickness * THICKNESS_SCALE, num_spheres_on_branch)

    base_multiplier = 1800
    min_size = 2.0
    size_scale = base_multiplier * branch_thicknesses
    if is_leaf:
        size_scale *= 0.5
    sphere_sizes = np.maximum(min_size, size_scale)

    for i in range(num_spheres_on_branch):
        varied_color = vary_color(base_color, COLOR_VARIATION)
        sphere_data.append({
            'pos': branch_points[i],
            'size': sphere_sizes[i],
            'color': varied_color,
            'level': level
        })

    if level < MAX_LEVEL:
        perp_axis = get_perpendicular_vector(direction)
        base_angle_rad = np.radians(BRANCHING_ANGLE)
        for i in range(NUM_BRANCHES):
            angle_dev = np.radians(np.random.uniform(-ANGLE_VARIATION, ANGLE_VARIATION))
            new_dir_rotated = rotate_vector(direction, perp_axis, base_angle_rad + angle_dev)

            spread_angle = (i * 2 * np.pi / NUM_BRANCHES) + np.random.uniform(-np.pi/NUM_BRANCHES, np.pi/NUM_BRANCHES) * 0.3
            final_new_dir = rotate_vector(new_dir_rotated, direction, spread_angle)
            noise = np.random.uniform(-DIRECTION_VARIATION, DIRECTION_VARIATION, 3)
            final_new_dir += noise
            final_new_dir = final_new_dir / np.linalg.norm(final_new_dir)

            create_fractal_tree_recursive_sequential(
                start_point=end_point,
                direction=final_new_dir,
                length=length * (LENGTH_SCALE + np.random.uniform(-0.05, 0.05)),
                thickness=thickness * THICKNESS_SCALE,
                level=level + 1,
                sphere_data=sphere_data
            )


def generate_terrain(size, grid_points, noise_factor, z_offset):
    x = np.linspace(-size, size, grid_points)
    y = np.linspace(-size, size, grid_points)
    X, Y = np.meshgrid(x, y)
    Z = np.random.uniform(-noise_factor, noise_factor, size=(grid_points, grid_points))
    Z += z_offset

    z_min, z_max = Z.min(), Z.max()
    norm_z = (Z - z_min) / (z_max - z_min) if (z_max - z_min) > 1e-6 else np.zeros_like(Z)

    colors = [(0, TERRAIN_COLOR_DIRT), (0.4, TERRAIN_COLOR_GRASS_LOW), (1.0, TERRAIN_COLOR_GRASS_HIGH)]
    cmap_terrain = LinearSegmentedColormap.from_list("terrain_cmap", colors)
    facecolors = cmap_terrain(norm_z)

    variation = 0.05
    noise = np.random.uniform(-variation, variation, size=(grid_points, grid_points, 3))
    facecolors[:, :, :3] = np.clip(facecolors[:, :, :3] + noise, 0, 1)

    return X, Y, Z, facecolors



print("Generating full tree data (sequentially)...")
trunk_start_point = np.array([0.0, 0.0, 0.0])
trunk_direction = np.array([0.0, 0.0, 1.0])
sphere_data = []
create_fractal_tree_recursive_sequential(trunk_start_point, trunk_direction, INITIAL_LENGTH, INITIAL_THICKNESS, level=0, sphere_data=sphere_data)

if not sphere_data:
    raise ValueError("Tree generation failed, no points created.")

n_spheres_total = len(sphere_data)
max_tree_level = max(item['level'] for item in sphere_data) if n_spheres_total > 0 else 0
print(f"Generated {n_spheres_total} spheres sequentially across {max_tree_level + 1} levels.")


SPHERES_PER_FRAME = math.ceil(n_spheres_total / FRAMES_GROWTH) if FRAMES_GROWTH > 0 else n_spheres_total

tree_positions = np.array([item['pos'] for item in sphere_data])
tree_sizes = np.array([item['size'] for item in sphere_data])
tree_final_colors = np.array([item['color'] for item in sphere_data])
tree_final_colors = np.clip(tree_final_colors, 0, 1)
tree_levels = np.array([item['level'] for item in sphere_data])


fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')


print("Generating terrain...")
X_terrain, Y_terrain, Z_terrain, C_terrain = generate_terrain(
    FLOOR_SIZE, TERRAIN_GRID_POINTS, TERRAIN_NOISE_FACTOR, TERRAIN_Z_OFFSET
)
terrain_surface = ax.plot_surface(
    X_terrain, Y_terrain, Z_terrain, facecolors=C_terrain,
    rstride=1, cstride=1,
    linewidth=0, antialiased=True,
    shade=True, alpha=0.9,
    zorder=1
)
print("Terrain generated.")


max_branch_length_at_level = lambda l: INITIAL_LENGTH * (LENGTH_SCALE ** l)
est_max_height = sum(max_branch_length_at_level(i) for i in range(MAX_LEVEL + 1))
star_limit = max(FLOOR_SIZE * 1.5, est_max_height * 1.2)

star_positions, star_sizes, star_rgba_colors = generate_background_stars(STAR_COUNT, star_limit)
star_scatter = ax.scatter(star_positions[:, 0], star_positions[:, 1], star_positions[:, 2],
                           s=star_sizes, c=star_rgba_colors, depthshade=False,
                           edgecolors='none', marker='*', zorder=0)


initial_tree_colors = tree_final_colors.copy()
initial_tree_colors[:, 3] = 0.0
tree_scatter = ax.scatter(tree_positions[:, 0], tree_positions[:, 1], tree_positions[:, 2],
                           s=tree_sizes,
                           c=initial_tree_colors,
                           depthshade=False,
                           alpha=None,
                           edgecolors='none',
                           zorder=10
                          )

particle_scatter = ax.scatter(np.empty((0, 3)), np.empty((0, 3)), np.empty((0, 3)),
                               s=np.empty((0,)), c=np.empty((0, 4)),
                               depthshade=False,
                               marker='.', edgecolors='none',
                               zorder=15
                              )


ax.set_xlim(-FLOOR_SIZE, FLOOR_SIZE)
ax.set_ylim(-FLOOR_SIZE, FLOOR_SIZE)
max_z_tree = np.max(tree_positions[:, 2]) if n_spheres_total > 0 else INITIAL_LENGTH
min_z_terrain = np.min(Z_terrain) if Z_terrain.size > 0 else TERRAIN_Z_OFFSET
ax.set_zlim(min_z_terrain - 0.1, max(max_z_tree * 1.1, est_max_height * 1.1))

ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()
ax.view_init(elev=VIEW_ELEV_START, azim=VIEW_AZIM_START)

z_range = ax.get_zlim()[1] - ax.get_zlim()[0]
ax.set_box_aspect([2 * FLOOR_SIZE, 2 * FLOOR_SIZE, z_range])


def update(frame):
    global prev_last_idx, particles

    current_visible_spheres = 0
    last_visible_sphere_index = -1
    if frame < FRAMES_GROWTH:
        current_visible_spheres = min(n_spheres_total, int((frame + 1) * SPHERES_PER_FRAME))
        last_visible_sphere_index = current_visible_spheres - 1
    else:
        current_visible_spheres = n_spheres_total
        last_visible_sphere_index = n_spheres_total - 1

    if last_visible_sphere_index != prev_last_idx:
        current_tree_colors = np.zeros((n_spheres_total, 4))
        if current_visible_spheres > 0:
            current_tree_colors[:current_visible_spheres] = tree_final_colors[:current_visible_spheres]

        current_tree_colors = np.clip(current_tree_colors, 0, 1)
        tree_scatter.set_facecolors(current_tree_colors)


    new_particles_generated = False
    if PARTICLE_EFFECT_ENABLED and frame < FRAMES_GROWTH:
        first_new_index = prev_last_idx + 1
        last_new_index = last_visible_sphere_index
        num_new_spheres = max(0, last_new_index - first_new_index + 1)

        if num_new_spheres > 0:
            particle_gen_stride = max(1, SPHERE_DENSITY // 6)
            indices_for_particles = np.arange(first_new_index, last_new_index + 1, particle_gen_stride)

            if len(indices_for_particles) > 0:
                level_changes = []
                for idx in indices_for_particles:
                    current_level = tree_levels[idx]
                    is_last = idx == n_spheres_total - 1
                    is_level_change = False
                    if not is_last and idx + 1 < n_spheres_total:
                        next_level = tree_levels[idx + 1]
                        is_level_change = next_level != current_level

                    if is_level_change or is_last:
                        level_changes.append(idx)

                burst_points = level_changes
                if not burst_points:
                   burst_points = indices_for_particles[::3]

                for idx in burst_points:
                    burst_pos = tree_positions[idx]
                    burst_vectors = generate_random_unit_vectors(PARTICLES_PER_BURST)

                    for i in range(PARTICLES_PER_BURST):
                        random_color = np.random.choice(PARTICLE_BASE_COLORS)
                        color_variation = 0.15
                        particle_color = vary_color(random_color, color_variation)

                        burst_direction = burst_vectors[i]
                        speed = PARTICLE_BURST_SPEED + np.random.uniform(-PARTICLE_BURST_SPEED_VARIATION,
                                                                        PARTICLE_BURST_SPEED_VARIATION)
                        burst_velocity = burst_direction * speed

                        level_factor = min(1.0, tree_levels[idx] / MAX_LEVEL * 1.2)
                        size_factor = 0.8 + level_factor * 0.4

                        particles.append([
                            burst_pos.copy(),
                            burst_velocity,
                            particle_color[:3],
                            PARTICLE_ALPHA_START,
                            PARTICLE_START_SIZE * size_factor,
                            frame
                        ])

                    new_particles_generated = True

    next_active_particles = []
    current_particle_pos_list = []
    current_particle_colors_list = []
    current_particle_sizes_list = []
    particles_updated = False

    if particles:
        particles_updated = True
        for p_data in particles:
            p_pos, p_velocity, p_base_rgb, p_start_alpha, p_start_size, p_creation_frame = p_data
            age = frame - p_creation_frame

            if age < PARTICLE_LIFESPAN_FRAMES:
                p_pos = p_pos + p_velocity
                life_fraction = age / PARTICLE_LIFESPAN_FRAMES

                birth_boost = 0
                if age < PARTICLE_BIRTH_GLOW_FRAMES:
                    birth_boost = 0.3 * (1 - age / PARTICLE_BIRTH_GLOW_FRAMES)

                ease_factor = 1.0 - (1.0 - life_fraction) ** 2
                current_alpha = p_start_alpha * (1.0 - ease_factor) + birth_boost
                current_alpha = np.clip(current_alpha, 0, 1)

                if life_fraction < 0.3:
                    size_factor = 1.0 + life_fraction/0.3 * 0.2
                else:
                    size_factor = 1.2 * (1.0 - ((life_fraction - 0.3) / 0.7))
                current_size = p_start_size * size_factor

                if current_alpha > 0.01 and current_size > 0.5:
                    current_particle_pos_list.append(p_pos)

                    glow_color = p_base_rgb.copy()
                    if birth_boost > 0:
                        glow_color = glow_color * (1 - birth_boost) + np.array([1, 1, 1]) * birth_boost
                        glow_color = np.clip(glow_color, 0, 1)

                    current_particle_colors_list.append(np.append(glow_color, current_alpha))
                    current_particle_sizes_list.append(current_size)

                    next_active_particles.append([
                        p_pos, p_velocity, p_base_rgb, p_start_alpha, p_start_size, p_creation_frame
                    ])

    if new_particles_generated or particles_updated:
        if current_particle_pos_list:
            pos_array = np.array(current_particle_pos_list)
            particle_scatter._offsets3d = (pos_array[:, 0], pos_array[:, 1], pos_array[:, 2])

            particle_colors_array = np.array(current_particle_colors_list)
            particle_colors_array = np.clip(particle_colors_array, 0, 1)
            particle_scatter.set_facecolors(particle_colors_array)

            particle_scatter.set_sizes(np.array(current_particle_sizes_list))
        elif particles_updated and not current_particle_pos_list:
             particle_scatter._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
             particle_scatter.set_facecolors(np.empty((0, 4)))
             particle_scatter.set_sizes(np.empty((0,)))


    particles = next_active_particles
    prev_last_idx = last_visible_sphere_index

    current_azim = (VIEW_AZIM_START + frame * (AZIM_DRIFT_SPEED / FPS)) % 360
    time_rad = frame * ELEV_OSCILLATION_SPEED / FPS
    current_elev = VIEW_ELEV_START + ELEV_OSCILLATION * math.sin(time_rad)
    ax.view_init(elev=current_elev, azim=current_azim)

    return (tree_scatter, particle_scatter)



ffmpeg_path = shutil.which('ffmpeg')
if not ffmpeg_path:
    print("WARNING: ffmpeg not found. Animation saving will likely fail.")
    print("Please install ffmpeg and ensure it's in your system's PATH.")
else:
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path


print("Setting up animation...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                              blit=False,
                              interval=1000/FPS)

class TqdmProgressCallback:
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Saving Video", unit="frame", ncols=100)
    def __call__(self, current_frame, total_frames):
        self.pbar.update(1)
    def close(self):
        self.pbar.close()

progress_bar = TqdmProgressCallback(TOTAL_FRAMES)

print(f"Saving animation to {OUTPUT_FILENAME}...")

writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='Fractal Tree Visualizer'), bitrate=3000,
                                extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium'])

try:
    ani.save(OUTPUT_FILENAME, writer=writer, dpi=VIDEO_DPI, progress_callback=progress_bar)
    print("Video saving complete.")
except Exception as e:
    print(f"\nError during video saving: {e}")
    print("Check ffmpeg installation, memory usage, or plotting parameters.")
finally:
    progress_bar.close()

plt.close(fig)
print("Figure closed.")