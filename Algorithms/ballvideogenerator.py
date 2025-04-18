import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import time
import subprocess
import os
from collections import deque
import colorsys
import wave
from matplotlib.patches import Arc

VIDEO_LENGTH_SECONDS = 35
FPS = 60
WIDTH = 1080
HEIGHT = 1920
DPI = 120
DURATION = VIDEO_LENGTH_SECONDS
FRAMES = DURATION * FPS

BG_COLOR = '#0A0A15'
WHITE = '#FFFFFF'
RED = '#FF0000'
PARTICLE_COLOR = '#00FFFF'
TRAIL_COLOR = '#ADD8E6'
PARTICLE_COLORS = ['#00FFFF', '#80FFFF', '#40E0FF', '#00DFFF', '#00B0FF']

CIRCLE_RADIUS_PERCENT = 0.4
CIRCLE_BORDER_WIDTH = 3
RED_BORDER_WIDTH = 4
MAX_RED_RADIUS_PERCENT = 0.5
RED_GROWTH_RATE = 0.02
ROTATION_SPEED_DEG_PER_SEC = 90

NUM_BALLS = 1
MAIN_BALL_RADIUS = 30
RED_BALL_RADIUS = 15
INITIAL_SPEED_RANGE = 15
GRAVITY = -0.5
BOUNCE_DAMPING = 1.40
RED_BALL_BOUNCE_DAMPING_BASE = 1.4
RED_BALL_BOUNCE_DAMPING_MAX = 1.5
RED_BALL_BOUNCE_PROGRESSION = 0.12
MIN_SPEED = 0.8
MAX_MAIN_BALL_SPEED = 35.0
MAX_RED_BALL_SPEED = 20.0
RED_BOUNCE_MULTIPLIER = 1.8

PARTICLE_LIFETIME = 40
PARTICLE_SIZE = 6
NUM_PARTICLES_PER_EMISSION = 20
PARTICLE_SPREAD = 7
PARTICLE_GLOW_FACTOR = 1.3

SAMPLE_RATE = 44100

def generate_sine_wave(frequency, duration, amplitude=8000):
    t = np.linspace(0, duration, int(duration * SAMPLE_RATE), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    envelope = np.linspace(1.0, 0.0, len(t))
    y *= envelope
    return y.astype(np.int16)

def create_collision_sound(frequency=440, duration=0.09):
    return generate_sine_wave(frequency, duration)

def create_red_collision_sound(frequency=660, duration=0.12):
    return generate_sine_wave(frequency, duration, amplitude=9000)

def combine_sounds(sounds):
    if not sounds: return np.array([], dtype=np.int16)
    max_len = max(len(s) for s in sounds)
    combined_sound = np.zeros(max_len, dtype=np.float64)
    for sound in sounds:
        combined_sound[:len(sound)] += sound.astype(np.float64)
    max_amplitude = np.max(np.abs(combined_sound))
    if max_amplitude > 32767:
        combined_sound = (combined_sound / max_amplitude) * 32767
    elif max_amplitude == 0:
        return np.array([], dtype=np.int16)
    return combined_sound.astype(np.int16)

def play_sound(sound_data, filename="temp_sound.wav"):
    if sound_data.size == 0:
        return

    if os.path.exists(filename):
        try: os.remove(filename)
        except OSError as e: print(f"Warning: Could not remove existing temp file {filename}: {e}")

    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
            wf.writeframes(sound_data.tobytes())
    except Exception as e:
        print(f"Error writing temp sound file '{filename}': {e}")
        return

    try:
        process = subprocess.Popen(
            ['ffplay', '-nodisp', '-autoexit', '-vn', '-loglevel', 'error', filename],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except FileNotFoundError:
        if not hasattr(play_sound, "ffplay_not_found_warned"):
            print("Warning: ffplay (part of FFmpeg) not found in system PATH. Sound effects disabled.")
            play_sound.ffplay_not_found_warned = True
        if os.path.exists(filename):
            try: os.remove(filename)
            except OSError as e_rem: print(f"Warning: Could not remove temp file '{filename}' after FileNotFoundError: {e_rem}")
        return
    except Exception as e:
        print(f"Error trying to play sound with ffplay: {e}")
        if os.path.exists(filename):
            try: os.remove(filename)
            except OSError as e_rem: print(f"Warning: Could not remove temp file '{filename}' after Exception: {e_rem}")
        return

class Ball:
    def __init__(self, x, y, radius, color=WHITE):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.speed_x = random.uniform(-INITIAL_SPEED_RANGE, INITIAL_SPEED_RANGE)
        self.speed_y = random.uniform(INITIAL_SPEED_RANGE * 0.5, INITIAL_SPEED_RANGE * 1.2)
        self.trail = deque(maxlen=12)
        self.bounce_count = 0
        self.current_bounce_damping = RED_BALL_BOUNCE_DAMPING_BASE if color == RED else BOUNCE_DAMPING

    def update(self):
        if self.color == WHITE:
            self.speed_y += GRAVITY

        current_speed = np.sqrt(self.speed_x**2 + self.speed_y**2)
        max_speed = MAX_MAIN_BALL_SPEED if self.color == WHITE else MAX_RED_BALL_SPEED

        if current_speed > max_speed:
            scale_factor = max_speed / current_speed
            self.speed_x *= scale_factor
            self.speed_y *= scale_factor

        self.x += self.speed_x
        self.y += self.speed_y
        self.trail.append((self.x, self.y))

    def bounce(self, circle_x, circle_y, circle_radius, red_arc_start_angle, red_arc_extent_angle):
        hit_type = None
        dx = self.x - circle_x
        dy = self.y - circle_y
        distance_sq = dx*dx + dy*dy
        effective_radius = circle_radius - self.radius

        if distance_sq > effective_radius*effective_radius:
            distance = np.sqrt(distance_sq)
            if distance == 0: return None

            nx = dx / distance
            ny = dy / distance
            dot_product = self.speed_x * nx + self.speed_y * ny

            if dot_product >= 0:
                self.bounce_count += 1

                collision_angle_deg = np.degrees(np.arctan2(dy, dx)) % 360
                red_arc_end_angle = (red_arc_start_angle + red_arc_extent_angle)
                norm_start_angle = red_arc_start_angle % 360
                norm_end_angle = red_arc_end_angle % 360

                is_red_hit = False
                if norm_start_angle <= norm_end_angle:
                    if norm_start_angle <= collision_angle_deg <= norm_end_angle: is_red_hit = True
                else:
                    if collision_angle_deg >= norm_start_angle or collision_angle_deg <= norm_end_angle: is_red_hit = True

                hit_type = 'red' if is_red_hit else 'white'

                bounce_randomness = 0.05
                rand_nx = nx * (1 + random.uniform(-bounce_randomness, bounce_randomness))
                rand_ny = ny * (1 + random.uniform(-bounce_randomness, bounce_randomness))

                rand_len = np.sqrt(rand_nx**2 + rand_ny**2)
                rand_nx /= rand_len
                rand_ny /= rand_len

                self.speed_x -= 2 * dot_product * rand_nx
                self.speed_y -= 2 * dot_product * rand_ny

                if self.color == RED:
                    if self.bounce_count <= 10:
                        self.current_bounce_damping = min(
                            RED_BALL_BOUNCE_DAMPING_MAX,
                            RED_BALL_BOUNCE_DAMPING_BASE + (RED_BALL_BOUNCE_PROGRESSION * self.bounce_count)
                        )

                    self.speed_x *= self.current_bounce_damping
                    self.speed_y *= self.current_bounce_damping
                    self.speed_y += random.uniform(0.5, 1.5)
                else:
                    self.speed_x *= BOUNCE_DAMPING
                    self.speed_y *= BOUNCE_DAMPING

                if hit_type == 'red' and self.color == WHITE:
                    self.speed_x *= RED_BOUNCE_MULTIPLIER
                    self.speed_y *= RED_BOUNCE_MULTIPLIER
                    self.speed_x += random.uniform(-2.0, 2.0)
                    self.speed_y += random.uniform(-2.0, 2.0)

                overshoot = distance - effective_radius
                push_back_dist = overshoot + 0.2
                self.x -= nx * push_back_dist
                self.y -= ny * push_back_dist

                current_speed_sq = self.speed_x**2 + self.speed_y**2
                min_speed_sq = MIN_SPEED**2
                if 1e-9 < current_speed_sq < min_speed_sq:
                    scale = MIN_SPEED / np.sqrt(current_speed_sq)
                    self.speed_x *= scale; self.speed_y *= scale
                elif current_speed_sq <= 1e-9:
                    self.speed_x = random.uniform(-MIN_SPEED*2, MIN_SPEED*2)
                    self.speed_y = random.uniform(MIN_SPEED, MIN_SPEED * 2.5)

                current_speed = np.sqrt(self.speed_x**2 + self.speed_y**2)
                max_speed = MAX_MAIN_BALL_SPEED if self.color == WHITE else MAX_RED_BALL_SPEED

                if current_speed > max_speed:
                    scale_factor = max_speed / current_speed
                    self.speed_x *= scale_factor
                    self.speed_y *= scale_factor

        return hit_type

class Particle:
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME, dx_mod=1.0, dy_mod=1.0):
        self.x, self.y = x, y
        if isinstance(color, str):
            self.color = random.choice(PARTICLE_COLORS) if color == PARTICLE_COLOR else color
        else:
            self.color = color

        self.size = size * random.uniform(0.6, 1.4)
        self.lifetime = lifetime * random.uniform(0.7, 1.3)
        self.max_lifetime = max(1, self.lifetime)
        self.initial_size = self.size

        self.dx = random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD) * dx_mod * random.uniform(0.8, 1.2)
        self.dy = random.uniform(-PARTICLE_SPREAD, PARTICLE_SPREAD) * dy_mod * random.uniform(0.8, 1.2)
        self.dy += random.uniform(0, PARTICLE_SPREAD * 0.3)

        self.rotation = random.uniform(0, 360) if random.random() < 0.3 else None
        self.rotation_speed = random.uniform(-5, 5) if self.rotation is not None else 0

        self.pulse_factor = random.uniform(0, 0.2) if random.random() < 0.4 else 0
        self.pulse_speed = random.uniform(0.1, 0.3)
        self.pulse_offset = random.uniform(0, 6.28)

    def update(self):
        self.dy += GRAVITY * 0.08
        self.x += self.dx
        self.y += self.dy
        self.lifetime -= 1

        life_fraction = self.lifetime / self.max_lifetime
        self.size = 0 if life_fraction <= 0 else self.initial_size * (life_fraction**0.7)

        if self.pulse_factor > 0:
            pulse = np.sin(self.pulse_offset + (1.0 - life_fraction) * 10 * self.pulse_speed) * self.pulse_factor
            self.size *= (1.0 + pulse)

        self.size = max(0, self.size)

        if self.rotation is not None:
            self.rotation += self.rotation_speed

        self.dx *= 0.96
        self.dy *= 0.96

        return self.lifetime > 0

def generate_frame(balls, circle_x, circle_y, circle_radius, red_radius_angle, current_rotation_angle, particles, ax):
    ax.clear()
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')

    for i in range(3):
        glow_radius = circle_radius * (1 + 0.03 * i)
        glow_alpha = 0.03 * (3 - i)
        ax.add_patch(plt.Circle((circle_x, circle_y), glow_radius, ec=WHITE, fc='none',
                                lw=CIRCLE_BORDER_WIDTH - i*0.5, alpha=glow_alpha, zorder=4))

    ax.add_patch(plt.Circle((circle_x, circle_y), circle_radius, ec=WHITE, fc='none',
                            lw=CIRCLE_BORDER_WIDTH, zorder=5))

    if red_radius_angle > 0:
        for i in range(2):
            glow_lw = RED_BORDER_WIDTH - i*0.8
            glow_alpha = 0.2 * (2 - i)
            ax.add_patch(Arc((circle_x, circle_y), circle_radius * 2 + i*4, circle_radius * 2 + i*4,
                             angle=0, theta1=current_rotation_angle, theta2=current_rotation_angle + red_radius_angle,
                             ec='#FF3333', lw=glow_lw, alpha=glow_alpha, fill=False, zorder=5))

        ax.add_patch(Arc((circle_x, circle_y), circle_radius * 2, circle_radius * 2,
                         angle=0, theta1=current_rotation_angle, theta2=current_rotation_angle + red_radius_angle,
                         ec=RED, lw=RED_BORDER_WIDTH, fill=False, zorder=6))

    collision_sounds_this_frame = []
    spawn_requests = []
    red_hit_occurred_this_frame = False
    active_balls = []

    for ball in balls:
        ball.update()
        hit_result = ball.bounce(circle_x, circle_y, circle_radius, current_rotation_angle, red_radius_angle)

        dx_b, dy_b = ball.x - circle_x, ball.y - circle_y
        dist_b = np.sqrt(dx_b*dx_b + dy_b*dy_b)
        px_mod, py_mod = (dx_b/dist_b, dy_b/dist_b) if dist_b > 0 else (0, 1)

        if hit_result == 'red':
            if ball.color == WHITE:
                red_hit_occurred_this_frame = True

            collision_sounds_this_frame.append(create_red_collision_sound())

            if ball.color == WHITE:
                spawn_requests.append({'x': ball.x, 'y': ball.y, 'radius': RED_BALL_RADIUS})

                for _ in range(NUM_PARTICLES_PER_EMISSION):
                    angle = random.uniform(0, 2 * np.pi)
                    speed_mod = random.uniform(0.8, 1.5)
                    dx_mod = np.cos(angle) * speed_mod
                    dy_mod = np.sin(angle) * speed_mod

                    size_mod = random.uniform(0.7, 1.8)
                    lifetime_mod = random.uniform(0.8, 1.2)

                    particles.append(Particle(
                        ball.x, ball.y, RED,
                        size=PARTICLE_SIZE * 1.4 * size_mod,
                        lifetime=PARTICLE_LIFETIME * lifetime_mod,
                        dx_mod=dx_mod * 1.5,
                        dy_mod=dy_mod * 1.5
                    ))

        elif hit_result == 'white':
            collision_sounds_this_frame.append(create_collision_sound())

            if ball.color == WHITE:
                for _ in range(NUM_PARTICLES_PER_EMISSION // 2):
                    angle = random.uniform(0, 2 * np.pi)
                    speed_mod = random.uniform(0.7, 1.3)
                    dx_mod = np.cos(angle) * speed_mod
                    dy_mod = np.sin(angle) * speed_mod

                    particles.append(Particle(
                        ball.x, ball.y, PARTICLE_COLOR,
                        size=PARTICLE_SIZE * random.uniform(0.7, 1.1),
                        lifetime=PARTICLE_LIFETIME * random.uniform(0.6, 0.9),
                        dx_mod=dx_mod, dy_mod=dy_mod
                    ))

        if len(ball.trail) > 1:
            num_points = len(ball.trail)

            if ball.color == WHITE and len(ball.trail) > 2:
                for i, (tx, ty) in enumerate(reversed(ball.trail)):
                    if i % 2 == 0:
                        alpha = 0.2 * (1 - i / num_points)**1.2
                        size = ball.radius * (0.3 + 0.8 * (1 - i / num_points))
                        ax.add_patch(plt.Circle((tx, ty), size * 1.5, color='#B0E0FF',
                                                alpha=max(0, alpha * 0.4), zorder=0, ec='none'))

            for i, (tx, ty) in enumerate(reversed(ball.trail)):
                alpha = 0.8 * (1 - i / num_points)**1.7
                size = ball.radius * (0.15 + 0.7 * (1 - i / num_points))
                trail_color = '#FF9999' if ball.color == RED else TRAIL_COLOR
                ax.add_patch(plt.Circle((tx, ty), size, color=trail_color,
                                        alpha=max(0, alpha), zorder=1, ec='none'))

        if ball.color == WHITE:
            ax.add_patch(plt.Circle((ball.x, ball.y), ball.radius * 1.2, color='#AAAAFF',
                                    alpha=0.15, zorder=1, ec='none'))
            ax.add_patch(plt.Circle((ball.x, ball.y), ball.radius * 1.1, color='#CCCCFF',
                                    alpha=0.2, zorder=1, ec='none'))

        ax.add_patch(plt.Circle((ball.x, ball.y), ball.radius, color=ball.color, zorder=2))

        highlight_offset = ball.radius * 0.3
        highlight_size = ball.radius * 0.25
        ax.add_patch(plt.Circle((ball.x - highlight_offset, ball.y - highlight_offset),
                                highlight_size, color='#FFFFFF', alpha=0.5, zorder=3, ec='none'))

        active_balls.append(ball)

    balls[:] = active_balls

    active_particles = []
    for particle in particles:
        if particle.update():
            active_particles.append(particle)

            alpha = max(0, min(1, (particle.lifetime / particle.max_lifetime)**0.7)) * 0.95

            if particle.size > 1.0:
                glow_size = particle.size * PARTICLE_GLOW_FACTOR
                glow_alpha = alpha * 0.4
                ax.add_patch(plt.Circle((particle.x, particle.y), glow_size,
                                        color=particle.color, alpha=glow_alpha, zorder=2, ec='none'))

            ax.add_patch(plt.Circle((particle.x, particle.y), particle.size,
                                    color=particle.color, alpha=alpha, zorder=3, ec='none'))

    particles[:] = active_particles

    if collision_sounds_this_frame:
        play_sound(combine_sounds(collision_sounds_this_frame))

    return ax, spawn_requests, red_hit_occurred_this_frame

def main():
    fig, ax = plt.subplots(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR); fig.canvas.draw()
    w, h = fig.canvas.get_width_height(); print(f"Canvas dimensions: {w}x{h}")
    circle_x, circle_y = w / 2, h / 2
    circle_radius = min(w, h) * CIRCLE_RADIUS_PERCENT

    red_radius_angle = 60
    max_red_angle = 360 * MAX_RED_RADIUS_PERCENT
    red_growth_amount = 360 * RED_GROWTH_RATE * 1.5

    current_rotation_angle = 0.0
    rotation_per_frame = ROTATION_SPEED_DEG_PER_SEC / FPS

    balls = [Ball(circle_x, circle_y + circle_radius * 0.5, MAIN_BALL_RADIUS, color=WHITE)]
    particles = []
    all_frames_data = []

    print("Starting frame generation...")
    start_time = time.time()

    for frame_num in range(FRAMES):
        current_rotation_angle = (current_rotation_angle + rotation_per_frame) % 360
        ax, spawn_requests, red_hit_occurred = generate_frame(balls, circle_x, circle_y, circle_radius, red_radius_angle, current_rotation_angle, particles, ax)

        if red_hit_occurred:
            red_radius_angle = min(red_radius_angle + red_growth_amount, max_red_angle)

        for req in spawn_requests:
            spawn_x, spawn_y = req['x'], req['y']
            new_ball_radius = req['radius']
            for _ in range(2):
                offset_angle = random.uniform(0, 2 * np.pi)
                offset_rad = new_ball_radius * random.uniform(1.5, 3.0)
                balls.append(Ball(spawn_x + offset_rad * np.cos(offset_angle),
                                  spawn_y + offset_rad * np.sin(offset_angle),
                                  new_ball_radius, color=RED))

        fig.canvas.draw()
        frame_data_argb = fig.canvas.tostring_argb()
        img_np = np.frombuffer(frame_data_argb, dtype=np.uint8).reshape((h, w, 4))
        frame_data = img_np[:, :, 1:4].tobytes()
        all_frames_data.append(frame_data)

        if (frame_num + 1) % FPS == 0 or frame_num == FRAMES - 1:
            elapsed = time.time() - start_time
            fps_current = (frame_num + 1) / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_num + 1}/{FRAMES} ({fps_current:.1f} FPS) | Balls: {len(balls)} | Red Arc: {red_radius_angle:.1f}° @ {current_rotation_angle:.1f}° | Particles: {len(particles)}")

    total_time = time.time() - start_time
    print(f"Finished generating {FRAMES} frames in {total_time:.2f} seconds.")

    output_file = "bouncing_balls_enhanced_v9_super_energetic.mp4"
    print(f"Creating video '{output_file}' ({w}x{h} @ {FPS}fps)...")
    ffmpeg_process = None
    try:
        command = ['ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'rgb24', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-crf', '21', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-metadata', f'comment=Gravity={GRAVITY}, Rotation={ROTATION_SPEED_DEG_PER_SEC}deg/s, BounceDamping={BOUNCE_DAMPING}, RedBoost={RED_BOUNCE_MULTIPLIER}, MaxSpeed={MAX_MAIN_BALL_SPEED}', output_file]
        ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Writing frames to FFmpeg...")
        for frame_data in all_frames_data:
            try: ffmpeg_process.stdin.write(frame_data)
            except IOError as e: print(f"\nError writing frame data: {e}"); break
        print("Finished writing frames. Closing FFmpeg pipe...")
        if ffmpeg_process.stdin: ffmpeg_process.stdin.close()
        print("Waiting for FFmpeg to complete...")
        stdout_data, stderr_data = ffmpeg_process.communicate(timeout=60)
        if ffmpeg_process.returncode != 0:
            print(f"\nError: FFmpeg exited code {ffmpeg_process.returncode}")
            print("--- FFmpeg Stderr ---"); print(stderr_data.decode(errors='ignore')); print("---------------------")
        else:
            print(f"Video saved successfully to '{output_file}'")
    except FileNotFoundError: print("\nError: 'ffmpeg' command not found. Ensure FFmpeg is installed and in PATH.")
    except subprocess.TimeoutExpired:
        print("\nError: FFmpeg process timed out.")
        if ffmpeg_process: ffmpeg_process.kill(); _, stderr_data = ffmpeg_process.communicate(); print("--- FFmpeg Stderr (timeout) ---"); print(stderr_data.decode(errors='ignore')); print("-----------------------------")
    except Exception as e: print(f"\nVideo creation error: {e}")
    finally:
        if ffmpeg_process and ffmpeg_process.poll() is None:
            print("Terminating potentially hung FFmpeg process...")
            ffmpeg_process.terminate()
            try:
                ffmpeg_process.wait(timeout=5)
                print("FFmpeg process terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("FFmpeg did not terminate gracefully after 5s, killing.")
                ffmpeg_process.kill()
                print("FFmpeg process killed.")

        if os.path.exists("temp_sound.wav"):
            try:
                os.remove("temp_sound.wav")
            except Exception as e:
                print(f"Warning: Could not remove temporary sound file 'temp_sound.wav': {e}")

    plt.close(fig); print("Script finished.")

if __name__ == "__main__":
    main()