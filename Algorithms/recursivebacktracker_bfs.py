import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import random
from collections import deque
import colorsys
import heapq

np.random.seed(42)

FIG_WIDTH = 9
FIG_HEIGHT = 16
DPI = 200

MAZE_WIDTH = 43
MAZE_HEIGHT = 43

BG_COLOR = '#0A0A15'
WALL_COLOR = '#FFFFFF'
CURRENT_COLOR = '#FF1493'
GENERATION_PARTICLE = '#00FF7F'
SOLVING_PARTICLE = '#1E90FF'
VISITED_COLOR = '#3A1C71'
PATH_COLOR = '#FF3333'
START_COLOR = '#00FF7F'
END_COLOR = '#FF4500'

RBT_GENERATION_PARTICLE = '#00FFAA'

BFS_SOLVING_PARTICLE = '#00CCFF'

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
    def __init__(self, x, y, color, size=PARTICLE_SIZE, lifetime=PARTICLE_LIFETIME):
        self.x = x
        self.y = y
        self.original_color = color
        self.color = color
        self.size = size * random.uniform(0.8, 1.2)
        self.max_lifetime = lifetime
        self.lifetime = lifetime
        self.decay = 0.85

    def update(self):
        self.lifetime -= 1
        self.size *= self.decay
        alpha = self.lifetime / self.max_lifetime
        r, g, b = self.original_color
        self.color = (r, g, b, alpha)
        return self.lifetime > 0 and self.size > 0.1

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

def create_maze_recursive_backtracker(width, height):
    grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
    generation_states = []
    cells_added = 0
    start_x, start_y = random.randint(0, width-1), random.randint(0, height-1)
    grid[start_x][start_y].visited = True
    grid[start_x][start_y].generation_order = cells_added
    cells_added += 1
    stack = [(start_x, start_y)]
    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]
    active_cells = [(start_x, start_y)]
    generation_states.append(capture_generation_state(grid, grid[start_x][start_y], [], [], active_cells))
    while stack:
        x, y = stack[-1]
        unvisited_neighbors = []
        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].visited:
                unvisited_neighbors.append((nx, ny, direction))
        if unvisited_neighbors:
            nx, ny, direction = random.choice(unvisited_neighbors)
            grid[x][y].walls[direction] = False
            opposite = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}
            grid[nx][ny].walls[opposite[direction]] = False
            grid[nx][ny].visited = True
            grid[nx][ny].generation_order = cells_added
            cells_added += 1
            stack.append((nx, ny))
            active_cells = [(x, y), (nx, ny)]
            generation_states.append(capture_generation_state(grid, grid[nx][ny], [], [], active_cells))
        else:
            stack.pop()
            if stack:
                active_cells = [stack[-1]]
            if stack: # Check if stack is not empty after popping
                generation_states.append(capture_generation_state(grid, grid[stack[-1][0]][stack[-1][1]], [], [], active_cells))

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

def capture_generation_state(grid, current_cell, path, walls, active_cells=None):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'is_frontier': [[cell.is_frontier for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'walls_to_check': walls,
        'phase': 'generation',
        'active_cells': active_cells or []
    }
    return state

def create_gradient_color(order, total):
    if order < 0:
        return (0.1, 0.1, 0.2)
    norm_pos = min(1.0, order / total)
    h1, s1, v1 = 0.75, 0.9, 0.5
    h2, s2, v2 = 0.55, 0.8, 0.9
    h = h1 + (h2 - h1) * norm_pos
    s = s1 + (s2 - s1) * norm_pos
    v = v1 + (v2 - v1) * norm_pos
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def solve_maze_bfs(grid, start_pos, end_pos):
    width = len(grid)
    height = len(grid[0])
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    grid[start_x][start_y].is_start = True
    grid[end_x][end_y].is_end = True
    for x in range(width):
        for y in range(height):
            grid[x][y].visited = False
            grid[x][y].in_path = False
    queue = deque([(start_x, start_y)])
    visited = set([(start_x, start_y)])
    grid[start_x][start_y].visited = True
    came_from = {}
    solving_states = []
    exploration_path = [grid[start_x][start_y]]
    active_cells = [(start_x, start_y)]
    solving_states.append(capture_solving_state(grid, grid[start_x][start_y],
                          exploration_path, [], False, active_cells))
    solution_found = False
    final_path = []
    while queue and not solution_found:
        current_pos = queue.popleft()
        x, y = current_pos
        exploration_path.append(grid[x][y])
        if len(exploration_path) > 20:
            exploration_path = exploration_path[-20:]
        if current_pos == end_pos:
            solution_found = True
            temp_pos = current_pos
            while temp_pos in came_from:
                final_path.append(temp_pos)
                temp_pos = came_from[temp_pos]
            final_path.append(start_pos)
            final_path.reverse()
            active_cells = [(end_x, end_y)]
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                  exploration_path, [], False, active_cells))
            continue
        frontier_positions = []
        active_cells = [(x, y)]
        directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]
        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if (direction == 'N' and not grid[x][y].walls['N']) or \
                   (direction == 'S' and not grid[x][y].walls['S']) or \
                   (direction == 'E' and not grid[x][y].walls['E']) or \
                   (direction == 'W' and not grid[x][y].walls['W']):
                    visited.add((nx, ny))
                    grid[nx][ny].visited = True
                    came_from[(nx, ny)] = (x, y)
                    queue.append((nx, ny))
                    frontier_positions.append((nx, ny))
                    active_cells.append((nx, ny))
        if not solution_found:
            solving_states.append(capture_solving_state(grid, grid[x][y],
                                  exploration_path, frontier_positions, False, active_cells))
    if solution_found:
        for i in range(1, len(final_path) + 1):
            partial_path = []
            for j in range(i):
                pos = final_path[j]
                partial_path.append(grid[pos[0]][pos[1]])
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                  partial_path, [], True))
        for pos in final_path:
            px, py = pos
            grid[px][py].in_path = True
        final_path_cells = [grid[p[0]][p[1]] for p in final_path]
        for _ in range(10):
            solving_states.append(capture_solving_state(grid, grid[end_x][end_y],
                                  final_path_cells, [], True))
    return solving_states


def capture_solving_state(grid, current_cell, path, frontier_positions, is_solution_phase=False, active_cells=None):
    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path or cell in path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': [(cell.x, cell.y) for cell in path],
        'frontier_positions': frontier_positions,
        'phase': 'solving',
        'is_solution_phase': is_solution_phase,
        'active_cells': active_cells or []
    }
    return state

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
            # Ensure state_idx is valid
            if state_idx >= 0 and state_idx < len(solving_states):
                frames.append(solving_states[state_idx])
            elif solving_states: # If index is out of bounds but list is not empty, append last element
                 frames.append(solving_states[-1])

    return frames

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

def create_animation(frames, width, height, total_cells, algorithm_name, solving_method, gen_particle_color, solving_particle_color):
    cell_size = min(FIG_HEIGHT / height, FIG_WIDTH / width) * 0.9
    fig, axes = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    x_offset = (FIG_WIDTH - width * cell_size) / 2
    y_offset = (FIG_HEIGHT - height * cell_size) / 2
    particles = []

    def update(i):
        nonlocal particles
        if i >= len(frames): # Safety check
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

        if phase == 'generation' and 'walls_to_check' in frame:
            walls_to_check = frame['walls_to_check']
        else:
            walls_to_check = []

        frontier_positions = []
        if phase == 'solving' and 'frontier_positions' in frame:
            frontier_positions = frame['frontier_positions']

        generation_phase = phase == 'generation'

        if generation_phase:
            progress_percent = min(100, int((i / max(1, GEN_FRAMES - 1)) * 100)) # Avoid division by zero
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

            progress_percent = min(100, int(((i - GEN_FRAMES) / max(1, SOLVE_FRAMES - 1)) * 100)) # Avoid division by zero
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
                particle_color = hex_to_rgb(gen_particle_color)
                for _ in range(random.randint(5, 8)):
                    start_x = cell_center_x + random.uniform(-0.3, 0.3) * cell_size
                    start_y = cell_center_y + random.uniform(-0.3, 0.3) * cell_size
                    particles.append(Particle(
                        start_x, start_y,
                        particle_color,
                        size=random.uniform(1.5, 2.0) * PARTICLE_SIZE,
                        lifetime=random.randint(5, 8)
                    ))

        if not generation_phase:
            for ax, ay in active_cells:
                cell_center_x = x_offset + ax * cell_size + cell_size/2
                cell_center_y = y_offset + (height - 1 - ay) * cell_size + cell_size/2
                particle_color = hex_to_rgb(solving_particle_color)
                for _ in range(random.randint(6, 10)):
                    start_x = cell_center_x + random.uniform(-0.3, 0.3) * cell_size
                    start_y = cell_center_y + random.uniform(-0.3, 0.3) * cell_size
                    particles.append(Particle(
                        start_x, start_y,
                        particle_color,
                        size=random.uniform(1.5, 2.0) * PARTICLE_SIZE,
                        lifetime=random.randint(5, 8)
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

                is_frontier_wall = False
                if generation_phase:
                    for wall_x, wall_y, _ in walls_to_check:
                        if (x, y) == (wall_x, wall_y):
                            is_frontier_wall = True
                            break

                in_frontier = False
                if not generation_phase:
                    in_frontier = (x, y) in frontier_positions

                glow = False
                glow_size_factor = 1.0
                cell_color = 'white' # Default
                alpha = 0.05 # Default
                zorder = 1 # Default


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
                        cell_color = solving_particle_color
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
                elif in_frontier:
                    cell_color = solving_particle_color
                    alpha = 0.7
                    zorder = 18
                    glow = True
                    glow_color = solving_particle_color
                    glow_size_factor = 1.05
                elif in_path:
                    cell_color = PATH_COLOR
                    if not generation_phase and i >= GEN_FRAMES + SOLVE_FRAMES - 30:
                         alpha = 1.0
                         glow = True
                         glow_color = PATH_COLOR
                         glow_size_factor = 1.1 * pulse_factor
                    else:
                        alpha = 0.8
                        glow = True
                        glow_color = PATH_COLOR
                        glow_size_factor = 1.05 * pulse_factor
                    zorder = 15
                elif visited and not generation_phase:
                    cell_color = VISITED_COLOR
                    alpha = 0.7
                    zorder = 8
                    glow = False
                elif visited and generation_phase:
                    if generation_order >= 0: # Use >= 0 to include the start cell
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
        repeat=False # Set repeat to False
    )

    writer = animation.FFMpegWriter(
        fps=TARGET_FPS,
        metadata=dict(artist='Maze Generation & Solving'),
        bitrate=5000
    )

    return ani, writer, fig

def main():
    combo = {
            "generation_algo": "create_maze_recursive_backtracker",
            "solving_algo": "solve_maze_bfs",
            "gen_name": "Recursive Backtracker",
            "solving_name": "Breadth-First Search (BFS)",
            "gen_particle_color": RBT_GENERATION_PARTICLE,
            "solving_particle_color": BFS_SOLVING_PARTICLE,
            "output_file": "recursive_bfs_maze.mp4"
    }

    print("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈")
    print(f"📱 GENERATING & SOLVING MAZE ANIMATION - {MAZE_WIDTH}x{MAZE_HEIGHT} MAZE 📱")
    print(f"Generation: {combo['gen_name']} - Solving: {combo['solving_name']}")
    print("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈")

    print(f"🧩 Generating maze using {combo['gen_name']} ({MAZE_WIDTH}x{MAZE_HEIGHT})...")
    grid, generation_states, total_cells, entrance_pos, exit_pos = create_maze_recursive_backtracker(MAZE_WIDTH, MAZE_HEIGHT)

    start_pos = entrance_pos
    end_pos = exit_pos

    print(f"🔍 Solving maze using {combo['solving_name']}...")
    solving_states = solve_maze_bfs(grid, start_pos, end_pos)

    print(f"🎬 Creating {TOTAL_FRAMES} animation frames...")
    frames = create_animation_frames(generation_states, solving_states)

    print(f"🎨 Building animation...")
    ani, writer, fig = create_animation(
        frames, MAZE_WIDTH, MAZE_HEIGHT, total_cells,
        combo['gen_name'], combo['solving_name'],
        combo['gen_particle_color'], combo['solving_particle_color']
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