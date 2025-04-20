import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
from collections import deque
import colorsys
import math

np.random.seed(42)

FIG_WIDTH = 9
FIG_HEIGHT = 16
DPI = 200

MAZE_WIDTH = 43
MAZE_HEIGHT = 43

BG_COLOR = '#0A0A15'
WALL_COLOR = '#FFFFFF'
CURRENT_COLOR = '#FF1493'
VISITED_COLOR = '#3A1C71'
PATH_COLOR = '#FF3333'
START_COLOR = '#00FF7F'
END_COLOR = '#FF4500'

SIDEWINDER_PARTICLE = '#FF124B'
BIDIRECT_PARTICLE_A = '#00FF9F'
BIDIRECT_PARTICLE_B = '#FFC800'

PARTICLE_LIFETIME = 10
PARTICLE_SIZE = 1.5

GEN_DURATION = 5
SOLVE_DURATION = 10
TOTAL_DURATION = GEN_DURATION + SOLVE_DURATION
TARGET_FPS = 30
GEN_FRAMES = GEN_DURATION * TARGET_FPS
SOLVE_FRAMES = SOLVE_DURATION * TARGET_FPS
TOTAL_FRAMES = TOTAL_DURATION * TARGET_FPS

GLOW_INTENSITY = 1.2
PULSE_SPEED = 0.15

class Particle:
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME, velocity=(0, 0)):
        self.x = x
        self.y = y
        self.original_color = color
        self.color = color
        self.size = size * random.uniform(0.8, 1.2)
        self.max_lifetime = lifetime
        self.lifetime = lifetime
        self.decay = random.uniform(0.8, 0.9)
        self.vx = velocity[0] * random.uniform(0.7, 1.3)
        self.vy = velocity[1] * random.uniform(0.7, 1.3)

    def update(self):
        self.lifetime -= 1
        self.size *= self.decay
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.92
        self.vy *= 0.92

        alpha = self.lifetime / self.max_lifetime
        r, g, b = self.original_color
        self.color = (r, g, b, alpha)
        return self.lifetime > 0 and self.size > 0.1

class SwirlParticle(Particle):
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME, center_x=None, center_y=None):
        super().__init__(x, y, color, size, lifetime)
        self.decay = 0.9
        self.center_x = center_x if center_x is not None else x
        self.center_y = center_y if center_y is not None else y

        self.radius = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        self.angle = math.atan2(y - self.center_y, x - self.center_x)
        self.angular_velocity = random.uniform(0.1, 0.2) * (1 if random.random() > 0.5 else -1)
        self.radial_velocity = random.uniform(-0.01, 0.01)

    def update(self):
        self.lifetime -= 1
        self.size *= self.decay

        self.angle += self.angular_velocity
        self.radius += self.radial_velocity

        self.x = self.center_x + self.radius * math.cos(self.angle)
        self.y = self.center_y + self.radius * math.sin(self.angle)

        alpha = self.lifetime / self.max_lifetime
        r, g, b = self.original_color
        self.color = (r, g, b, alpha)
        return self.lifetime > 0 and self.size > 0.1 and self.radius > 0

class EnergyParticle(Particle):
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME, is_forward=True):
        super().__init__(x, y, color, size, lifetime)
        self.decay = 0.92
        self.is_forward = is_forward

        self.pulse_speed = random.uniform(0.2, 0.4)
        self.pulse_amplitude = random.uniform(0.1, 0.2)
        self.time = random.uniform(0, 6.28)

        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(0.01, 0.03)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def update(self):
        self.lifetime -= 1
        self.time += self.pulse_speed

        pulse_factor = 1.0 + self.pulse_amplitude * math.sin(self.time)
        self.size *= self.decay * pulse_factor

        self.x += self.vx
        self.y += self.vy

        self.vx *= 0.95
        self.vy *= 0.95

        alpha = self.lifetime / self.max_lifetime
        r, g, b = self.original_color
        self.color = (r, g, b, alpha)
        return self.lifetime > 0 and self.size > 0.05

class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'N': True, 'E': True, 'S': True, 'W': True}
        self.visited = False
        self.in_path = False
        self.is_start = False
        self.is_end = False
        self.is_frontier = False
        self.generation_order = -1
        self.is_in_run = False

        self.fwd_visited = False
        self.bwd_visited = False
        self.fwd_parent = None
        self.bwd_parent = None

def create_maze_sidewinder(width, height, p_east=0.5):
    grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
    generation_states = []
    cells_added = 0

    for x in range(width - 1):
        grid[x][0].visited = True
        grid[x][0].generation_order = cells_added
        cells_added += 1

        grid[x][0].walls['E'] = False
        grid[x+1][0].walls['W'] = False

        active_cells = [(x, 0), (x+1, 0)]
        for cx, cy in active_cells:
            grid[cx][cy].is_in_run = True

        generation_states.append(capture_generation_state(
            grid, grid[x][0], [], [], active_cells))

        for cx, cy in active_cells:
            grid[cx][cy].is_in_run = False

    grid[width-1][0].visited = True
    grid[width-1][0].generation_order = cells_added
    cells_added += 1

    generation_states.append(capture_generation_state(
        grid, grid[width-1][0], [], [], [(width-1, 0)]))

    for y in range(1, height):
        run = []

        for x in range(width):
            grid[x][y].visited = True
            grid[x][y].generation_order = cells_added
            cells_added += 1

            run.append((x, y))

            for cx, cy in run:
                grid[cx][cy].is_in_run = True

            end_run = x == width - 1 or (random.random() > p_east)

            if not end_run:
                grid[x][y].walls['E'] = False
                grid[x+1][y].walls['W'] = False

                active_cells = run.copy()
                if x < width - 1:
                    active_cells.append((x+1, y))
                generation_states.append(capture_generation_state(
                    grid, grid[x][y], [], [], active_cells))
            else:
                cell_x, _ = random.choice(run)
                grid[cell_x][y].walls['N'] = False
                grid[cell_x][y-1].walls['S'] = False

                active_cells = run.copy()
                active_cells.append((cell_x, y-1))
                generation_states.append(capture_generation_state(
                    grid, grid[x][y], [], [], active_cells))

                for cx, cy in run:
                    grid[cx][cy].is_in_run = False

                run = []

    entrance_x = random.randint(1, width-2)
    exit_x = random.randint(1, width-2)
    grid[entrance_x][0].walls['N'] = False
    grid[exit_x][height-1].walls['S'] = False
    grid[entrance_x][0].visited = True
    grid[exit_x][height-1].visited = True
    grid[entrance_x][0].is_start = True
    grid[exit_x][height-1].is_end = True
    entrance_pos = (entrance_x, 0)
    exit_pos = (exit_x, height-1)

    for x in range(width):
        for y in range(height):
            if y == 0 and x != entrance_x:
                grid[x][y].walls['N'] = True
            if y == height-1 and x != exit_x:
                grid[x][y].walls['S'] = True
            if x == 0:
                grid[x][y].walls['W'] = True
            if x == width-1:
                grid[x][y].walls['E'] = True

    generation_states.append(capture_generation_state(grid, None, [], []))
    return grid, generation_states, cells_added, entrance_pos, exit_pos

def solve_maze_bidirectional_bfs(grid, start_pos, end_pos):
    width = len(grid)
    height = len(grid[0])
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    for x in range(width):
        for y in range(height):
            grid[x][y].visited = False
            grid[x][y].in_path = False
            grid[x][y].fwd_visited = False
            grid[x][y].bwd_visited = False
            grid[x][y].fwd_parent = None
            grid[x][y].bwd_parent = None

    grid[start_x][start_y].is_start = True
    grid[end_x][end_y].is_end = True

    fwd_queue = deque([(start_x, start_y)])
    bwd_queue = deque([(end_x, end_y)])

    grid[start_x][start_y].fwd_visited = True
    grid[end_x][end_y].bwd_visited = True

    solving_states = []
    fwd_cells = [grid[start_x][start_y]]
    bwd_cells = [grid[end_x][end_y]]

    solving_states.append(capture_solving_state(
        grid, None, [], [], False,
        fwd_active_cells=[(start_x, start_y)],
        bwd_active_cells=[(end_x, end_y)],
        fwd_cells=fwd_cells,
        bwd_cells=bwd_cells))

    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]

    meeting_cell = None

    while fwd_queue and bwd_queue:
        if fwd_queue:
            fwd_size = len(fwd_queue)
            fwd_frontier = []

            for _ in range(fwd_size):
                x, y = fwd_queue.popleft()

                if grid[x][y].bwd_visited:
                    meeting_cell = (x, y)
                    break

                for direction, dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].fwd_visited:
                        if (direction == 'N' and not grid[x][y].walls['N']) or \
                           (direction == 'S' and not grid[x][y].walls['S']) or \
                           (direction == 'E' and not grid[x][y].walls['E']) or \
                           (direction == 'W' and not grid[x][y].walls['W']):

                            grid[nx][ny].fwd_visited = True
                            grid[nx][ny].fwd_parent = (x, y)
                            fwd_queue.append((nx, ny))
                            fwd_frontier.append((nx, ny))

                            fwd_cells.append(grid[nx][ny])
                            if len(fwd_cells) > 20:
                                fwd_cells = fwd_cells[-20:]

            if meeting_cell:
                break

            solving_states.append(capture_solving_state(
                grid, None, [], [], False,
                fwd_active_cells=fwd_frontier,
                bwd_active_cells=[],
                fwd_cells=fwd_cells,
                bwd_cells=bwd_cells))

        if bwd_queue:
            bwd_size = len(bwd_queue)
            bwd_frontier = []

            for _ in range(bwd_size):
                x, y = bwd_queue.popleft()

                if grid[x][y].fwd_visited:
                    meeting_cell = (x, y)
                    break

                for direction, dx, dy in directions:
                    nx, ny = x + dx, y + dy

                    if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].bwd_visited:
                        if (direction == 'N' and not grid[x][y].walls['N']) or \
                           (direction == 'S' and not grid[x][y].walls['S']) or \
                           (direction == 'E' and not grid[x][y].walls['E']) or \
                           (direction == 'W' and not grid[x][y].walls['W']):

                            grid[nx][ny].bwd_visited = True
                            grid[nx][ny].bwd_parent = (x, y)
                            bwd_queue.append((nx, ny))
                            bwd_frontier.append((nx, ny))

                            bwd_cells.append(grid[nx][ny])
                            if len(bwd_cells) > 20:
                                bwd_cells = bwd_cells[-20:]

            if meeting_cell:
                break

            solving_states.append(capture_solving_state(
                grid, None, [], [], False,
                fwd_active_cells=[],
                bwd_active_cells=bwd_frontier,
                fwd_cells=fwd_cells,
                bwd_cells=bwd_cells))

    if meeting_cell:
        fwd_path = []
        current = meeting_cell
        while current != start_pos:
            fwd_path.append(current)
            current = grid[current[0]][current[1]].fwd_parent
        fwd_path.append(start_pos)
        fwd_path.reverse()

        bwd_path = []
        current = meeting_cell
        while current != end_pos:
            if current != meeting_cell:
                bwd_path.append(current)
            current = grid[current[0]][current[1]].bwd_parent
        bwd_path.append(end_pos)

        complete_path = fwd_path + bwd_path

        for pos in complete_path:
            px, py = pos
            grid[px][py].in_path = True

        for i in range(1, len(complete_path) + 1):
            partial_path = []
            for j in range(i):
                pos = complete_path[j]
                partial_path.append(grid[pos[0]][pos[1]])

            if i <= len(fwd_path):
                active_pos = complete_path[i-1]
                solving_states.append(capture_solving_state(
                    grid, grid[active_pos[0]][active_pos[1]],
                    partial_path, [], True,
                    fwd_active_cells=[active_pos],
                    bwd_active_cells=[],
                    is_solution=True))
            else:
                active_pos = complete_path[i-1]
                solving_states.append(capture_solving_state(
                    grid, grid[active_pos[0]][active_pos[1]],
                    partial_path, [], True,
                    fwd_active_cells=[],
                    bwd_active_cells=[active_pos],
                    is_solution=True))

        final_path_cells = [grid[p[0]][p[1]] for p in complete_path]
        for _ in range(10):
            solving_states.append(capture_solving_state(
                grid, None, final_path_cells, [], True,
                fwd_active_cells=[start_pos],
                bwd_active_cells=[end_pos],
                is_solution=True))

    return solving_states

def capture_generation_state(grid, current_cell, path, walls, active_cells=None):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'is_frontier': [[cell.is_frontier for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'is_in_run': [[cell.is_in_run for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'walls_to_check': walls,
        'phase': 'generation',
        'active_cells': active_cells or []
    }
    return state

def capture_solving_state(grid, current_cell, path, frontier_positions, is_solution_phase=False,
                        fwd_active_cells=None, bwd_active_cells=None, fwd_cells=None, bwd_cells=None,
                        is_solution=False):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path or cell in path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'fwd_visited': [[cell.fwd_visited for cell in row] for row in grid],
        'bwd_visited': [[cell.bwd_visited for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'frontier_positions': frontier_positions,
        'phase': 'solving',
        'is_solution_phase': is_solution_phase,
        'fwd_active_cells': fwd_active_cells or [],
        'bwd_active_cells': bwd_active_cells or [],
        'fwd_cells': [(cell.x, cell.y) for cell in fwd_cells] if fwd_cells else [],
        'bwd_cells': [(cell.x, cell.y) for cell in bwd_cells] if bwd_cells else [],
        'is_solution': is_solution
    }
    return state

def create_gradient_color(order, total):
    if order < 0:
        return (0.1, 0.1, 0.2)
    norm_pos = min(1.0, order / total)
    h1, s1, v1 = 0.95, 0.9, 0.9
    h2, s2, v2 = 0.98, 0.9, 0.7
    h = h1 + (h2 - h1) * norm_pos
    s = s1 + (s2 - s1) * norm_pos
    v = v1 + (v2 - v1) * norm_pos
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def create_animation_frames(generation_states, solving_states):
    frames = []
    gen_total_states = len(generation_states)
    for frame_idx in range(GEN_FRAMES):
        progress = frame_idx / GEN_FRAMES
        state_idx = min(int(progress * gen_total_states), gen_total_states - 1)
        frames.append(generation_states[state_idx])

    solve_total_states = len(solving_states)
    for frame_idx in range(SOLVE_FRAMES):
        if frame_idx >= SOLVE_FRAMES - 30:
            frames.append(solving_states[-1])
        else:
            progress = frame_idx / (SOLVE_FRAMES - 30)
            state_idx = min(int(progress * (solve_total_states - 1)), solve_total_states - 2)
            if state_idx >= 0 and state_idx < len(solving_states):
                frames.append(solving_states[state_idx])
            elif solving_states:
                frames.append(solving_states[-1])

    return frames

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def get_random_direction(speed=0.02):
    angle = random.uniform(0, 2 * np.pi)
    return math.cos(angle) * speed, math.sin(angle) * speed

def create_animation(frames, width, height, total_cells, algorithm_name, solving_method, gen_particle_color, solving_particle_colors):
    cell_size = min(FIG_HEIGHT / height, FIG_WIDTH / width) * 0.9
    fig, axes = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    x_offset = (FIG_WIDTH - width * cell_size) / 2
    y_offset = (FIG_HEIGHT - height * cell_size) / 2
    particles = []

    def update(i):
        nonlocal particles
        if i >= len(frames):
            return

        axes.clear()
        axes.set_xlim(0, FIG_WIDTH)
        axes.set_ylim(0, FIG_HEIGHT)
        axes.set_facecolor(BG_COLOR)
        axes.axis('off')

        frame = frames[i]
        walls_data = frame['walls']
        visited_data = frame['visited']
        in_path_data = frame['in_path']
        is_start_data = frame['is_start']
        is_end_data = frame['is_end']
        generation_order_data = frame['generation_order']
        current_pos = frame['current']
        path_positions = frame['path']
        phase = frame['phase']

        is_in_run_data = None
        if phase == 'generation' and 'is_in_run' in frame:
            is_in_run_data = frame['is_in_run']

        fwd_visited_data = None
        bwd_visited_data = None
        fwd_active_cells = []
        bwd_active_cells = []
        fwd_cells = []
        bwd_cells = []
        is_solution = False

        if phase == 'solving':
            if 'fwd_visited' in frame:
                fwd_visited_data = frame['fwd_visited']
            if 'bwd_visited' in frame:
                bwd_visited_data = frame['bwd_visited']
            if 'fwd_active_cells' in frame:
                fwd_active_cells = frame['fwd_active_cells']
            if 'bwd_active_cells' in frame:
                bwd_active_cells = frame['bwd_active_cells']
            if 'fwd_cells' in frame:
                fwd_cells = frame['fwd_cells']
            if 'bwd_cells' in frame:
                bwd_cells = frame['bwd_cells']
            if 'is_solution' in frame:
                is_solution = frame['is_solution']

        if phase == 'generation' and 'walls_to_check' in frame:
            walls_to_check = frame['walls_to_check']
        else:
            walls_to_check = []

        generation_phase = phase == 'generation'

        if generation_phase:
            progress_percent = min(100, int((i / max(1, GEN_FRAMES - 1)) * 100))
            axes.text(FIG_WIDTH/2, y_offset + height * cell_size + 2.0,
                   "Maze Generation",
                   color='white', fontsize=20, ha='center', weight='bold')
            axes.text(FIG_WIDTH/2, y_offset + height * cell_size + 1.3,
                   f"Algorithm used: {algorithm_name}",
                   color='white', fontsize=12, ha='center')
            axes.text(FIG_WIDTH/2, y_offset + height * cell_size + 0.8,
                   f"Progress: {progress_percent}%",
                   color='white', fontsize=12, ha='center')
        else:
            is_solution_phase = 'is_solution_phase' in frame and frame['is_solution_phase']
            if is_solution_phase:
                title_text = "Solution Path"
            else:
                title_text = "Exploring Maze"

            progress_percent = min(100, int(((i - GEN_FRAMES) / max(1, SOLVE_FRAMES - 1)) * 100))
            axes.text(FIG_WIDTH/2, y_offset + height * cell_size + 2.0,
                   title_text,
                   color='white', fontsize=20, ha='center', weight='bold')
            axes.text(FIG_WIDTH/2, y_offset + height * cell_size + 1.3,
                   f"Algorithm used: {solving_method}",
                   color='white', fontsize=12, ha='center')
            axes.text(FIG_WIDTH/2, y_offset + height * cell_size + 0.8,
                   f"Progress: {progress_percent}%",
                   color='white', fontsize=12, ha='center')

        pulse_factor = 1.0 + 0.1 * np.sin(i * PULSE_SPEED)

        active_cells = []
        if 'active_cells' in frame:
            active_cells = frame['active_cells']

        if generation_phase:
            for ax, ay in active_cells:
                cell_center_x = x_offset + ax * cell_size + cell_size/2
                cell_center_y = y_offset + (height - 1 - ay) * cell_size + cell_size/2

                is_in_run = False
                if is_in_run_data:
                    is_in_run = is_in_run_data[ax][ay]

                if is_in_run:
                    run_cells = [(x, y) for x, y in active_cells if is_in_run_data[x][y]]
                    if run_cells:
                        avg_x = sum(x for x, _ in run_cells) / len(run_cells)
                        avg_y = sum(y for _, y in run_cells) / len(run_cells)
                        center_x = x_offset + avg_x * cell_size + cell_size/2
                        center_y = y_offset + (height - 1 - avg_y) * cell_size + cell_size/2

                        if random.random() < 0.1:
                            for _ in range(random.randint(2, 4)):
                                h = random.uniform(0.95, 1.0)
                                s = random.uniform(0.8, 1.0)
                                v = random.uniform(0.8, 1.0)
                                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                                swirl_color = (r, g, b)

                                particles.append(SwirlParticle(
                                    center_x + random.uniform(-0.1, 0.1) * cell_size,
                                    center_y + random.uniform(-0.1, 0.1) * cell_size,
                                    swirl_color,
                                    size=random.uniform(1.2, 1.8) * PARTICLE_SIZE,
                                    lifetime=random.randint(8, 12),
                                    center_x=center_x,
                                    center_y=center_y
                                ))
                else:
                    for _ in range(random.randint(3, 5)):
                        h = random.uniform(0.95, 1.0)
                        s = random.uniform(0.7, 0.9)
                        v = random.uniform(0.7, 0.9)
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        particle_color = (r, g, b)

                        vel_x, vel_y = get_random_direction(0.02)

                        particles.append(Particle(
                            cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                            cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                            particle_color,
                            size=random.uniform(1.2, 1.8) * PARTICLE_SIZE,
                            lifetime=random.randint(5, 8),
                            velocity=(vel_x, vel_y)
                        ))

        if not generation_phase:
            for ax, ay in fwd_active_cells:
                cell_center_x = x_offset + ax * cell_size + cell_size/2
                cell_center_y = y_offset + (height - 1 - ay) * cell_size + cell_size/2

                fwd_color = solving_particle_colors[0]
                particle_color = hex_to_rgb(fwd_color)

                for _ in range(random.randint(4, 7)):
                    h = random.uniform(0.45, 0.5)
                    s = random.uniform(0.8, 1.0)
                    v = random.uniform(0.8, 1.0)
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    energy_color = (r, g, b)

                    particles.append(EnergyParticle(
                        cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                        cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                        energy_color,
                        size=random.uniform(1.2, 1.8) * PARTICLE_SIZE,
                        lifetime=random.randint(6, 9),
                        is_forward=True
                    ))

            for ax, ay in bwd_active_cells:
                cell_center_x = x_offset + ax * cell_size + cell_size/2
                cell_center_y = y_offset + (height - 1 - ay) * cell_size + cell_size/2

                bwd_color = solving_particle_colors[1]
                particle_color = hex_to_rgb(bwd_color)

                for _ in range(random.randint(4, 7)):
                    h = random.uniform(0.12, 0.17)
                    s = random.uniform(0.8, 1.0)
                    v = random.uniform(0.8, 1.0)
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    energy_color = (r, g, b)

                    particles.append(EnergyParticle(
                        cell_center_x + random.uniform(-0.3, 0.3) * cell_size,
                        cell_center_y + random.uniform(-0.3, 0.3) * cell_size,
                        energy_color,
                        size=random.uniform(1.2, 1.8) * PARTICLE_SIZE,
                        lifetime=random.randint(6, 9),
                        is_forward=False
                    ))

            if is_solution:
                for px, py in path_positions:
                    cell_center_x = x_offset + px * cell_size + cell_size/2
                    cell_center_y = y_offset + (height - 1 - py) * cell_size + cell_size/2

                    if random.random() < 0.1:
                        is_forward = True
                        if fwd_visited_data and bwd_visited_data:
                            if not fwd_visited_data[px][py] and bwd_visited_data[px][py]:
                                is_forward = False

                        if is_forward:
                            h = random.uniform(0.45, 0.5)
                            particle_base_color = solving_particle_colors[0]
                        else:
                            h = random.uniform(0.12, 0.17)
                            particle_base_color = solving_particle_colors[1]

                        s = random.uniform(0.8, 1.0)
                        v = random.uniform(0.8, 1.0)
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        path_color = (r, g, b)

                        particles.append(EnergyParticle(
                            cell_center_x + random.uniform(-0.2, 0.2) * cell_size,
                            cell_center_y + random.uniform(-0.2, 0.2) * cell_size,
                            path_color,
                            size=random.uniform(1.5, 2.0) * PARTICLE_SIZE,
                            lifetime=random.randint(8, 12),
                            is_forward=is_forward
                        ))

        for x in range(width):
            for y in range(height):
                cell_x = x_offset + x * cell_size
                cell_y = y_offset + (height - 1 - y) * cell_size
                is_current = current_pos and (x, y) == current_pos
                in_path = in_path_data[x][y]
                visited = visited_data[x][y]
                is_start = is_start_data[x][y]
                is_end = is_end_data[x][y]
                generation_order = generation_order_data[x][y]

                is_in_run = False
                if is_in_run_data:
                    is_in_run = is_in_run_data[x][y]

                fwd_visited = False
                bwd_visited = False
                if fwd_visited_data:
                    fwd_visited = fwd_visited_data[x][y]
                if bwd_visited_data:
                    bwd_visited = bwd_visited_data[x][y]

                is_frontier_wall = False
                if generation_phase:
                    for wall_x, wall_y, _ in walls_to_check:
                        if (x, y) == (wall_x, wall_y):
                            is_frontier_wall = True
                            break

                glow = False
                glow_size_factor = 1.0
                cell_color = 'white'
                alpha = 0.05
                zorder = 1

                if is_start:
                    cell_color = START_COLOR
                    alpha = 1.0
                    zorder = 25
                    glow = True
                    glow_color = START_COLOR
                    glow_size_factor = 1.2 * pulse_factor
                elif is_end:
                    cell_color = END_COLOR
                    alpha = 1.0
                    zorder = 25
                    glow = True
                    glow_color = END_COLOR
                    glow_size_factor = 1.2 * pulse_factor
                elif is_current:
                    if generation_phase:
                        cell_color = gen_particle_color
                    else:
                        if fwd_visited:
                            cell_color = solving_particle_colors[0]
                        elif bwd_visited:
                            cell_color = solving_particle_colors[1]
                        else:
                            cell_color = PATH_COLOR
                    alpha = 0.9
                    zorder = 20
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.15 * pulse_factor
                elif is_frontier_wall and generation_phase:
                    cell_color = '#AAAAFF'
                    alpha = 0.6
                    zorder = 15
                    glow = False
                elif is_in_run and generation_phase:
                    h = 0.95
                    s = 0.8
                    v = 0.8
                    r, g, b = colorsys.hsv_to_rgb(h, s, v)
                    cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                    alpha = 0.5
                    zorder = 12
                    glow = True
                    glow_color = cell_color
                    glow_size_factor = 1.05
                elif in_path:
                    if not generation_phase and is_solution:
                        if fwd_visited and not bwd_visited:
                            h = 0.45
                            s = 0.9
                            v = 0.9
                        elif bwd_visited and not fwd_visited:
                            h = 0.12
                            s = 0.9
                            v = 0.9
                        else:
                            h = 0.0
                            s = 0.9
                            v = 0.9

                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                    else:
                        cell_color = PATH_COLOR

                    if not generation_phase and i >= GEN_FRAMES + SOLVE_FRAMES - 30:
                        alpha = 1.0
                        glow = True
                        glow_color = cell_color
                        glow_size_factor = 1.1 * pulse_factor
                    else:
                        alpha = 0.8
                        glow = True
                        glow_color = cell_color
                        glow_size_factor = 1.05 * pulse_factor
                    zorder = 15
                elif not generation_phase:
                    if fwd_visited and bwd_visited:
                        h = 0.0
                        s = 0.7
                        v = 0.7
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                        alpha = 0.7
                        zorder = 10
                        glow = True
                        glow_color = cell_color
                        glow_size_factor = 1.05
                    elif fwd_visited:
                        h = 0.45
                        s = 0.5
                        v = 0.7
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                        alpha = 0.6
                        zorder = 8
                        glow = False
                    elif bwd_visited:
                        h = 0.12
                        s = 0.5
                        v = 0.7
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        cell_color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
                        alpha = 0.6
                        zorder = 8
                        glow = False
                    else:
                        cell_color = 'white'
                        alpha = 0.05
                        zorder = 1
                elif visited and generation_phase:
                    if generation_order >= 0:
                        cell_color = create_gradient_color(generation_order, total_cells)
                        alpha = 0.7
                    else:
                        cell_color = (0.2, 0.3, 0.5)
                        alpha = 0.7
                    zorder = 5
                    glow = False
                else:
                    cell_color = 'white'
                    alpha = 0.05
                    zorder = 1

                cell_rect = patches.Rectangle(
                    (cell_x, cell_y), cell_size, cell_size,
                    fill=True, color=cell_color, alpha=alpha,
                    linewidth=0, zorder=zorder
                )
                axes.add_patch(cell_rect)

                if glow:
                    glow_size = cell_size * glow_size_factor
                    glow_offset = (cell_size - glow_size) / 2
                    glow_rect = patches.Rectangle(
                        (cell_x + glow_offset, cell_y + glow_offset),
                        glow_size, glow_size,
                        fill=False, edgecolor=glow_color, alpha=0.4,
                        linewidth=1.5,
                        zorder=zorder-1
                    )
                    axes.add_patch(glow_rect)

        new_particles = []
        for particle in particles:
            if particle.update():
                new_particles.append(particle)

                circle = plt.Circle(
                    (particle.x, particle.y),
                    particle.size * cell_size * 0.07,
                    color=particle.color,
                    alpha=particle.lifetime / particle.max_lifetime,
                    zorder=100
                )
                axes.add_patch(circle)

        particles = new_particles

        for x in range(width):
            for y in range(height):
                walls = walls_data[x][y]
                cell_x = x_offset + x * cell_size
                cell_y = y_offset + (height - 1 - y) * cell_size
                wall_color = WALL_COLOR
                line_width = 1.0
                if walls['N']:
                    axes.plot([cell_x, cell_x + cell_size],
                           [cell_y + cell_size, cell_y + cell_size],
                           wall_color, linewidth=line_width, alpha=0.7, zorder=30)
                if walls['E']:
                    axes.plot([cell_x + cell_size, cell_x + cell_size],
                           [cell_y, cell_y + cell_size],
                           wall_color, linewidth=line_width, alpha=0.7, zorder=30)
                if walls['S']:
                    axes.plot([cell_x, cell_x + cell_size],
                           [cell_y, cell_y],
                           wall_color, linewidth=line_width, alpha=0.7, zorder=30)
                if walls['W']:
                    axes.plot([cell_x, cell_x],
                           [cell_y, cell_y + cell_size],
                           wall_color, linewidth=line_width, alpha=0.7, zorder=30)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        blit=False,
        interval=1000 / TARGET_FPS,
        repeat=False
    )

    writer = animation.FFMpegWriter(
        fps=TARGET_FPS,
        metadata=dict(artist='Maze Generation & Solving'),
        bitrate=5000
    )

    return ani, writer, fig

def main():
    combo = {
            "generation_algo": "create_maze_sidewinder",
            "solving_algo": "solve_maze_bidirectional_bfs",
            "gen_name": "Sidewinder Algorithm",
            "solving_name": "Bidirectional BFS",
            "gen_particle_color": SIDEWINDER_PARTICLE,
            "solving_particle_colors": [BIDIRECT_PARTICLE_A, BIDIRECT_PARTICLE_B],
            "output_file": "sidewinder_bidirectional_maze.mp4"
    }

    print("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈")
    print(f"📱 GENERATING & SOLVING MAZE ANIMATION - {MAZE_WIDTH}x{MAZE_HEIGHT} MAZE 📱")
    print(f"Generation: {combo['gen_name']} - Solving: {combo['solving_name']}")
    print("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈")

    print(f"🧩 Generating maze using {combo['gen_name']} ({MAZE_WIDTH}x{MAZE_HEIGHT})...")
    grid, generation_states, total_cells, entrance_pos, exit_pos = create_maze_sidewinder(MAZE_WIDTH, MAZE_HEIGHT)

    start_pos = entrance_pos
    end_pos = exit_pos

    print(f"🔍 Solving maze using {combo['solving_name']}...")
    solving_states = solve_maze_bidirectional_bfs(grid, start_pos, end_pos)

    print(f"🎬 Creating {TOTAL_FRAMES} animation frames...")
    frames = create_animation_frames(generation_states, solving_states)

    print(f"🎨 Building animation...")
    ani, writer, fig = create_animation(
        frames, MAZE_WIDTH, MAZE_HEIGHT, total_cells,
        combo['gen_name'], combo['solving_name'],
        combo['gen_particle_color'], combo['solving_particle_colors']
    )

    output_file = combo['output_file']
    print(f"💾 Saving animation to {output_file}...")
    print("    (This may take several minutes for high quality)")

    ani.save(output_file, writer=writer)
    plt.close(fig)

    print(f"✅ Animation saved successfully to {output_file}")
    print("🚀 Animation ready!")

if __name__ == "__main__":
    main()