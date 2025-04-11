import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from tqdm import tqdm
import math
import shutil

N_SAMPLES = 500
CUBE_LIMIT = 2.0
CUBE_ALPHA = 0.15
CUBE_COLOR = '#888888'
DT = 0.1
MAX_SPEED = 0.25
MIN_SPEED = 0.05
MAX_FORCE = 0.05
PERCEPTION_RADIUS = 1.2
MINIMUM_DIST = 0.2
STRONG_SEPARATION_FORCE = 20.0
SEPARATION_FORCE = 2.0
ALIGNMENT_FORCE_SAME = 2.0
COHESION_FORCE_SAME = 1.5
SEPARATION_FORCE_DIFF = 6.0
MINIMUM_DIFF_COLOR_DIST = 0.45
STRONG_REPULSION_FORCE = 15.0
WANDER_STRENGTH = 0.08
FORWARD_BIAS = 0.01
CONTAINMENT_STRENGTH = 0.15
PERTURBATION_INTERVAL = 80
PERTURBATION_STRENGTH = 0.05
TRAIL_LENGTH = 10
TRAIL_ALPHA_START = 0.6
TRAIL_ALPHA_END = 0.0
TRAIL_LINEWIDTH = 1.0
PARTICLE_SIZE_MIN = 6
PARTICLE_SIZE_MAX = 12
PARTICLE_ALPHA = 0.9
PULSE_FREQUENCY = 0.4
PULSE_AMPLITUDE = 0.25
N_STARS = 250
STAR_SIZE_MIN = 0.1
STAR_SIZE_MAX = 0.6
STAR_ALPHA_MIN = 0.2
STAR_ALPHA_MAX = 0.6
ASPECT_RATIO = 9 / 16
FIG_WIDTH_INCHES = 6
FIG_HEIGHT_INCHES = FIG_WIDTH_INCHES / ASPECT_RATIO
FPS = 30
DURATION_FORMATION_SEC = 30
DURATION_STABLE_SEC = 10
FRAMES_FORMATION = DURATION_FORMATION_SEC * FPS
FRAMES_STABLE = DURATION_STABLE_SEC * FPS
TOTAL_FRAMES = FRAMES_FORMATION + FRAMES_STABLE
OUTPUT_FILENAME = "color_flocks_from_anarchy_cubic.mp4"
VIDEO_DPI = 150
COLOR_GRADIENTS = [
    [(0, '#ff1493'), (1, '#ff69b4')],
    [(0, '#4169e1'), (1, '#00bfff')],
    [(0, '#32cd32'), (1, '#98fb98')],
    [(0, '#ff4500'), (1, '#ff8c00')],
    [(0, '#ffff00'), (1, '#ffd700')],
    [(0, '#9932cc'), (1, '#da70d6')],
]
N_COLOR_GROUPS = len(COLOR_GRADIENTS)

def get_cube_edges(limit):
    edges = []
    points = np.array([
        [-limit, -limit, -limit], [+limit, -limit, -limit], [+limit, +limit, -limit], [-limit, +limit, -limit],
        [-limit, -limit, +limit], [+limit, -limit, +limit], [+limit, +limit, +limit], [-limit, +limit, +limit]
    ])
    edges.append(points[[0, 1, 2, 3, 0], :])
    edges.append(points[[4, 5, 6, 7, 4], :])
    edges.append(points[[0, 4], :]); edges.append(points[[1, 5], :])
    edges.append(points[[2, 6], :]); edges.append(points[[3, 7], :])
    return edges

def generate_background_stars(n_stars, limit):
    outer_limit = limit * 2.0
    positions = np.random.uniform(-outer_limit, outer_limit, size=(n_stars, 3))
    sizes = np.random.uniform(STAR_SIZE_MIN, STAR_SIZE_MAX, size=n_stars)
    alphas = np.random.uniform(STAR_ALPHA_MIN, STAR_ALPHA_MAX, size=n_stars)
    base_colors = np.random.choice(['#ffffff', '#f0f8ff', '#e6e6fa', '#b0c4de'], size=n_stars)
    rgba_colors = np.zeros((n_stars, 4))
    for i in range(n_stars):
        rgba_colors[i] = to_rgba(base_colors[i], alpha=alphas[i])
    return positions, sizes, rgba_colors

def limit_vector(vector, max_val):
    mag_sq = np.sum(vector**2)
    if mag_sq > max_val**2 and max_val > 0:
        mag = np.sqrt(mag_sq)
        vector = vector / mag * max_val
    elif max_val <= 0:
        vector = np.zeros_like(vector)
    return vector

def limit_vectors(vectors, max_val):
    max_val_arr = np.atleast_1d(np.asarray(max_val))
    if max_val_arr.ndim == 1 and max_val_arr.shape[0] == 1:
        max_val_arr = max_val_arr.reshape(1, 1)
    elif max_val_arr.ndim == 1 and max_val_arr.shape[0] == vectors.shape[0]:
        max_val_arr = max_val_arr.reshape(-1, 1)
    elif max_val_arr.shape != (vectors.shape[0], 1) and max_val_arr.shape != (1, 1):
        raise ValueError(f"max_val shape {max_val_arr.shape} not compatible")
    if np.all(max_val_arr <= 1e-12): return np.zeros_like(vectors)
    mag_sq = np.sum(vectors**2, axis=1, keepdims=True)
    mag = np.sqrt(np.maximum(mag_sq, 1e-12))
    limit_mask = (mag > max_val_arr) & (max_val_arr > 1e-12)
    scaling_factor = np.where(limit_mask, max_val_arr / mag, 1.0)
    return vectors * scaling_factor

def enforce_min_speed(velocities, min_speed):
    velocity_mags = np.sqrt(np.sum(velocities**2, axis=1, keepdims=True))
    too_slow_mask = velocity_mags < min_speed
    if np.any(too_slow_mask):
        norm_velocities = velocities / np.maximum(velocity_mags, 1e-6)
        near_zero_mask = velocity_mags < (min_speed * 0.1)
        random_dirs = np.random.uniform(-1, 1, size=velocities.shape)
        random_norms = np.sqrt(np.sum(random_dirs**2, axis=1, keepdims=True))
        random_dirs = random_dirs / np.maximum(random_norms, 1e-6)
        chosen_dirs = np.where(near_zero_mask, random_dirs, norm_velocities)
        adjusted_velocities = chosen_dirs * min_speed
        velocities = np.where(too_slow_mask, adjusted_velocities, velocities)
    return velocities

X_positions = np.random.uniform(-CUBE_LIMIT * 0.9, CUBE_LIMIT * 0.9, size=(N_SAMPLES, 3))
X_velocities = np.random.uniform(-MAX_SPEED * 0.8, MAX_SPEED * 0.8, size=(N_SAMPLES, 3))
group_labels = np.zeros(N_SAMPLES, dtype=int)
boids_per_group = N_SAMPLES // N_COLOR_GROUPS
remainder = N_SAMPLES % N_COLOR_GROUPS
for i in range(N_COLOR_GROUPS):
    start_idx = i * boids_per_group + min(i, remainder)
    end_idx = (i + 1) * boids_per_group + min(i + 1, remainder)
    if end_idx > start_idx:
        group_labels[start_idx:end_idx] = i
np.random.shuffle(group_labels)
X_trail_history = np.zeros((N_SAMPLES, TRAIL_LENGTH, 3))
for i in range(TRAIL_LENGTH): X_trail_history[:, i, :] = X_positions
particle_colors = np.zeros((N_SAMPLES, 4))
cmaps = []
for i, gradient in enumerate(COLOR_GRADIENTS):
    cmaps.append(LinearSegmentedColormap.from_list(f"custom_{i}", gradient))
for i in range(N_COLOR_GROUPS):
    group_indices = np.where(group_labels == i)[0]
    if len(group_indices) == 0: continue
    color_positions = np.linspace(0.3, 0.9, len(group_indices))
    np.random.shuffle(color_positions)
    cmap_idx = i % len(cmaps)
    for j, idx in enumerate(group_indices):
        particle_colors[idx] = to_rgba(cmaps[cmap_idx](color_positions[j]), alpha=PARTICLE_ALPHA)
particle_sizes = np.random.uniform(PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX, size=N_SAMPLES)
star_positions, star_sizes, star_rgba_colors = generate_background_stars(N_STARS, CUBE_LIMIT)
fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES), facecolor='black')
ax = fig.add_axes([0, 0, 1, 1], projection='3d', facecolor='black')
stars = ax.scatter(star_positions[:, 0], star_positions[:, 1], star_positions[:, 2],
                    s=star_sizes, c=star_rgba_colors, depthshade=True, edgecolors='none')
scatter = ax.scatter(X_positions[:, 0], X_positions[:, 1], X_positions[:, 2],
                    c=particle_colors, s=particle_sizes, depthshade=False, edgecolors='none')
cube_edges_data = get_cube_edges(CUBE_LIMIT)
for edge_data in cube_edges_data:
    ax.plot(edge_data[:, 0], edge_data[:, 1], edge_data[:, 2], color=CUBE_COLOR, lw=0.8, alpha=CUBE_ALPHA)
trail_lines = []
for i in range(N_SAMPLES):
    line, = ax.plot([], [], [], linewidth=TRAIL_LINEWIDTH, alpha=0, color=particle_colors[i, :3])
    trail_lines.append(line)
max_extent = CUBE_LIMIT * 1.1
ax.set_xlim(-max_extent, max_extent); ax.set_ylim(-max_extent, max_extent); ax.set_zlim(-max_extent, max_extent)
ax.set_box_aspect((1, 1, 1))
ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('k'); ax.yaxis.pane.set_edgecolor('k'); ax.zaxis.pane.set_edgecolor('k')
ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.set_axis_off()
initial_elev = 25; initial_azim = -50
azim_drift_speed = 0.05
elev_drift_speed = 0.0
ax.view_init(elev=initial_elev, azim=initial_azim)

def calculate_color_boid_forces(idx, positions, velocities, group_labels, perception_radius):
    my_pos = positions[idx]; my_vel = velocities[idx]; my_group = group_labels[idx]
    diff = positions - my_pos; dist_sq = np.sum(diff**2, axis=1); dist = np.sqrt(np.maximum(dist_sq, 1e-9))
    neighbor_mask_all = (dist_sq > 1e-9) & (dist_sq < perception_radius**2); neighbor_indices_all = np.where(neighbor_mask_all)[0]
    neighbor_count_all = len(neighbor_indices_all)
    steer_sep = np.zeros(3); strong_sep_total = np.zeros(3); normal_sep_total = np.zeros(3)
    if neighbor_count_all > 0:
        neighbor_diffs = diff[neighbor_indices_all]; neighbor_dists_1d = dist[neighbor_indices_all]; neighbor_dists = neighbor_dists_1d.reshape(-1, 1)
        close_mask_1d = neighbor_dists_1d < MINIMUM_DIST; normal_mask_1d = neighbor_dists_1d >= MINIMUM_DIST
        if np.any(close_mask_1d):
            close_diffs_subset = neighbor_diffs[close_mask_1d, :]; close_dists_subset = neighbor_dists[close_mask_1d, :]
            safe_close_dists = np.maximum(close_dists_subset, 1e-6)
            repulsion_factors = np.exp(STRONG_SEPARATION_FORCE * (1.0 - safe_close_dists / MINIMUM_DIST))
            repulsion_vectors = -close_diffs_subset / safe_close_dists * repulsion_factors
            strong_sep_total = np.sum(repulsion_vectors, axis=0)
        if np.any(normal_mask_1d):
            normal_diffs_subset = neighbor_diffs[normal_mask_1d, :]; normal_dists_subset = neighbor_dists[normal_mask_1d, :]
            safe_normal_dists_sq = np.maximum(normal_dists_subset**2, 1e-6)
            separation_vectors = -normal_diffs_subset / safe_normal_dists_sq
            normal_sep_total = np.sum(separation_vectors, axis=0)
        desired_sep = strong_sep_total + (normal_sep_total * SEPARATION_FORCE)
        if np.sum(desired_sep**2) > 1e-9:
            allow_boost = np.any(close_mask_1d); current_max_speed = MAX_SPEED * 1.3 if allow_boost else MAX_SPEED
            current_max_force = MAX_FORCE * 1.8 if allow_boost else MAX_FORCE
            desired_sep = limit_vector(desired_sep, current_max_speed); steer_sep = desired_sep - my_vel
            steer_sep = limit_vector(steer_sep, current_max_force)
    steer_sep_diff = np.zeros(3); all_diff_color_mask = (group_labels != my_group); diff_color_indices = np.where(all_diff_color_mask)[0]
    if len(diff_color_indices) > 0:
        diff_color_diffs = diff[diff_color_indices]; diff_color_dists_1d = dist[diff_color_indices]; diff_color_dists = diff_color_dists_1d.reshape(-1, 1)
        close_diff_mask_1d = (diff_color_dists_1d < MINIMUM_DIFF_COLOR_DIST) & (diff_color_dists_1d >= MINIMUM_DIST)
        normal_diff_mask_1d = (diff_color_dists_1d >= MINIMUM_DIFF_COLOR_DIST) & (diff_color_dists_1d < perception_radius)
        strong_repulsion_total = np.zeros(3); normal_repulsion_total = np.zeros(3)
        if np.any(close_diff_mask_1d):
            close_diffs_subset = diff_color_diffs[close_diff_mask_1d, :]; close_dists_subset = diff_color_dists[close_diff_mask_1d, :]
            safe_close_dists = np.maximum(close_dists_subset, 1e-6)
            repulsion_factors = np.exp(STRONG_REPULSION_FORCE * (1.0 - safe_close_dists / MINIMUM_DIFF_COLOR_DIST))
            repulsion_vectors = -close_diffs_subset / safe_close_dists * repulsion_factors
            strong_repulsion_total = np.sum(repulsion_vectors, axis=0)
        if np.any(normal_diff_mask_1d):
            normal_diffs_subset = diff_color_diffs[normal_diff_mask_1d, :]; normal_dists_subset = diff_color_dists[normal_diff_mask_1d, :]
            safe_normal_dists_sq = np.maximum(normal_dists_subset**2, 1e-6)
            normal_repulsion = -normal_diffs_subset / safe_normal_dists_sq
            normal_repulsion_total = np.sum(normal_repulsion, axis=0)
        desired_sep_diff = strong_repulsion_total + (normal_repulsion_total * SEPARATION_FORCE_DIFF)
        if np.sum(desired_sep_diff**2) > 1e-9:
            allow_boost = np.any(close_diff_mask_1d); current_max_speed = MAX_SPEED * 1.5 if allow_boost else MAX_SPEED
            current_max_force = MAX_FORCE * 2.0 if allow_boost else MAX_FORCE
            desired_sep_diff = limit_vector(desired_sep_diff, current_max_speed); steer_sep_diff = desired_sep_diff - my_vel
            steer_sep_diff = limit_vector(steer_sep_diff, current_max_force)
    steer_ali = np.zeros(3); steer_coh = np.zeros(3)
    same_color_radius = perception_radius * 1.2; neighbor_mask_same_color = (dist_sq > 1e-9) & (dist_sq < same_color_radius**2)
    neighbor_indices_extended = np.where(neighbor_mask_same_color)[0]
    if len(neighbor_indices_extended) > 0:
        neighbor_groups_extended = group_labels[neighbor_indices_extended]; same_color_mask = (neighbor_groups_extended == my_group)
        neighbor_indices_same = neighbor_indices_extended[same_color_mask]; neighbor_count_same = len(neighbor_indices_same)
        if neighbor_count_same > 0:
            avg_velocity_same = np.sum(velocities[neighbor_indices_same], axis=0) / neighbor_count_same
            desired_ali = limit_vector(avg_velocity_same, MAX_SPEED); steer_ali = desired_ali - my_vel
            steer_ali = limit_vector(steer_ali, MAX_FORCE)
            center_of_mass_same = np.sum(positions[neighbor_indices_same], axis=0) / neighbor_count_same
            desired_coh_vec = center_of_mass_same - my_pos
            if np.sum(desired_coh_vec**2) > 1e-9:
                desired_coh = limit_vector(desired_coh_vec, MAX_SPEED); steer_coh = desired_coh - my_vel
                steer_coh = limit_vector(steer_coh, MAX_FORCE)
    return steer_sep, steer_sep_diff, steer_ali, steer_coh

def update(frame):
    global X_positions, X_velocities, X_trail_history, particle_sizes
    X_trail_history = np.roll(X_trail_history, 1, axis=1); X_trail_history[:, 0, :] = X_positions
    if frame < FRAMES_FORMATION:
        phase = "formation"; formation_progress = frame / max(1, FRAMES_FORMATION)
        adjusted_progress = 0.5 - 0.5 * np.cos(formation_progress * np.pi)
        initial_factor_strength = 0.01; alignment_factor = initial_factor_strength + adjusted_progress**1.5 * (1.0 - initial_factor_strength)
        cohesion_factor = initial_factor_strength + adjusted_progress**1.5 * (1.0 - initial_factor_strength)
        current_wander = WANDER_STRENGTH * (1.0 - adjusted_progress); current_azim_drift = azim_drift_speed
    else:
        phase = "stable"; stable_progress = (frame - FRAMES_FORMATION) / max(1, FRAMES_STABLE)
        alignment_factor = 1.0; cohesion_factor = 1.0; current_wander = WANDER_STRENGTH * 0.05; current_azim_drift = azim_drift_speed * 0.3
    steer_sep_all = np.zeros_like(X_velocities); steer_sep_diff_all = np.zeros_like(X_velocities)
    steer_ali_all = np.zeros_like(X_velocities); steer_coh_all = np.zeros_like(X_velocities)
    for i in range(N_SAMPLES):
        steer_sep, steer_sep_diff, steer_ali, steer_coh = calculate_color_boid_forces(i, X_positions, X_velocities, group_labels, PERCEPTION_RADIUS)
        steer_sep_all[i] = steer_sep; steer_sep_diff_all[i] = steer_sep_diff; steer_ali_all[i] = steer_ali; steer_coh_all[i] = steer_coh
    boid_force = (steer_sep_all + steer_sep_diff_all + steer_ali_all * ALIGNMENT_FORCE_SAME * alignment_factor + steer_coh_all * COHESION_FORCE_SAME * cohesion_factor)
    wander_force = np.random.uniform(-1, 1, size=X_velocities.shape) * current_wander
    if frame > 0 and frame % PERTURBATION_INTERVAL == 0:
        perturbation = np.random.uniform(-1, 1, size=X_velocities.shape) * PERTURBATION_STRENGTH
        wander_force += perturbation
    containment_force = np.zeros_like(X_velocities); wall_distance_left = X_positions[:, 0] + CUBE_LIMIT
    wall_distance_right = CUBE_LIMIT - X_positions[:, 0]; wall_distance_bottom = X_positions[:, 1] + CUBE_LIMIT
    wall_distance_top = CUBE_LIMIT - X_positions[:, 1]; wall_distance_back = X_positions[:, 2] + CUBE_LIMIT
    wall_distance_front = CUBE_LIMIT - X_positions[:, 2]; wall_factor = 0.3; safe_distance = 0.2
    containment_force[:, 0] += np.where(wall_distance_left < safe_distance, np.exp(wall_factor * (safe_distance - np.maximum(wall_distance_left, 0))), 0) * CONTAINMENT_STRENGTH
    containment_force[:, 0] -= np.where(wall_distance_right < safe_distance, np.exp(wall_factor * (safe_distance - np.maximum(wall_distance_right, 0))), 0) * CONTAINMENT_STRENGTH
    containment_force[:, 1] += np.where(wall_distance_bottom < safe_distance, np.exp(wall_factor * (safe_distance - np.maximum(wall_distance_bottom, 0))), 0) * CONTAINMENT_STRENGTH
    containment_force[:, 1] -= np.where(wall_distance_top < safe_distance, np.exp(wall_factor * (safe_distance - np.maximum(wall_distance_top, 0))), 0) * CONTAINMENT_STRENGTH
    containment_force[:, 2] += np.where(wall_distance_back < safe_distance, np.exp(wall_factor * (safe_distance - np.maximum(wall_distance_back, 0))), 0) * CONTAINMENT_STRENGTH
    containment_force[:, 2] -= np.where(wall_distance_front < safe_distance, np.exp(wall_factor * (safe_distance - np.maximum(wall_distance_front, 0))), 0) * CONTAINMENT_STRENGTH
    velocity_mags = np.sqrt(np.sum(X_velocities**2, axis=1, keepdims=True)); norm_directions = X_velocities / np.maximum(velocity_mags, 1e-6)
    forward_bias = norm_directions * FORWARD_BIAS; total_force = boid_force + wander_force + containment_force + forward_bias
    X_velocities += total_force * DT; X_velocities = limit_vectors(X_velocities, MAX_SPEED)
    X_velocities = enforce_min_speed(X_velocities, MIN_SPEED); X_positions += X_velocities * DT
    X_positions = np.clip(X_positions, -CUBE_LIMIT + 0.01, CUBE_LIMIT - 0.01)
    current_azim = (initial_azim + frame * current_azim_drift) % 360; current_elev = initial_elev + math.sin(frame * 0.01) * 2
    ax.view_init(elev=current_elev, azim=current_azim)
    velocity_mags_1d = np.sqrt(np.sum(X_velocities**2, axis=1)); velocity_size_factor = np.interp(velocity_mags_1d, [MIN_SPEED, MAX_SPEED * 0.8], [PARTICLE_SIZE_MIN, PARTICLE_SIZE_MAX])
    current_time_sec = frame / FPS; pulse_factor = 1.0 + PULSE_AMPLITUDE * np.sin(2 * np.pi * PULSE_FREQUENCY * current_time_sec)
    pulse_phase = 2 * np.pi * (0.2 * (group_labels / max(N_COLOR_GROUPS, 1)) + current_time_sec * 0.3)
    individual_pulse = 1.0 + 0.15 * np.sin(pulse_phase); particle_sizes = velocity_size_factor * pulse_factor * individual_pulse
    particle_sizes = np.clip(particle_sizes, PARTICLE_SIZE_MIN * 0.8, PARTICLE_SIZE_MAX * 1.2)
    scatter._offsets3d = (X_positions[:, 0], X_positions[:, 1], X_positions[:, 2]); scatter.set_sizes(particle_sizes)
    trail_alpha_scale = 1.0 if phase == "formation" else max(0.0, 1.0 - stable_progress * 1.5)
    for i in range(N_SAMPLES):
        trail_x = X_trail_history[i, :, 0]; trail_y = X_trail_history[i, :, 1]; trail_z = X_trail_history[i, :, 2]
        if np.any(np.isfinite(trail_x)):
            trail_lines[i].set_data(trail_x, trail_y); trail_lines[i].set_3d_properties(trail_z)
            vel_mag = velocity_mags_1d[i]; intensity_boost = np.clip(vel_mag / (MAX_SPEED * 0.6 + 1e-6), 0.5, 1.3)
            max_alpha = TRAIL_ALPHA_START * trail_alpha_scale * intensity_boost
            trail_lines[i].set_alpha(np.clip(max_alpha, 0, 1)); trail_lines[i].set_linewidth(np.clip(TRAIL_LINEWIDTH * (0.8 + 0.4 * intensity_boost), 0.5, TRAIL_LINEWIDTH * 1.5))
        else:
            trail_lines[i].set_alpha(0)
    return (scatter,) + tuple(trail_lines)

ffmpeg_path = shutil.which('ffmpeg')
if ffmpeg_path: plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=False, interval=1000/FPS)

class TqdmProgressCallback:
    def __init__(self, total): self.pbar = tqdm(total=total, desc="Saving Video", unit="frame", ncols=100)
    def __call__(self, current_frame, total_frames): self.pbar.update(1)
    def close(self): self.pbar.close()

progress_bar = TqdmProgressCallback(TOTAL_FRAMES)
writer = animation.FFMpegWriter(fps=FPS, metadata=dict(artist='AI Boids Simulation'), bitrate=-1, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
ani.save(OUTPUT_FILENAME, writer=writer, dpi=VIDEO_DPI, progress_callback=progress_bar)
progress_bar.close()
plt.close(fig)