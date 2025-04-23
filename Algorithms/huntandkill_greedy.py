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

HUNTANDKILL_GENERATION_PARTICLE = '#00E6AA'
GREEDY_SOLVING_PARTICLE = '#FF9500'

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
        if isinstance(self.original_color, tuple) and len(self.original_color) == 3:
             r, g, b = self.original_color
             self.color = (r, g, b, alpha)
        else:
             self.color = (1, 1, 1, alpha) # Default white

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

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def create_maze_huntandkill(width, height):
    grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
    generation_states = []
    cells_added = 0

    exploration_path = []
    max_path_length = 20

    entrance_x = random.randint(0, width - 1) if width > 1 else 0
    exit_x = random.randint(0, width - 1) if width > 1 else 0

    start_x, start_y = entrance_x, 0
    grid[start_x][start_y].visited = True
    grid[start_x][start_y].generation_order = cells_added
    grid[start_x][start_y].is_start = True
    grid[start_x][start_y].walls['N'] = False
    cells_added += 1
    current_x, current_y = start_x, start_y

    exploration_path.append(grid[current_x][current_y])

    active_cells = [(current_x, current_y)]
    generation_states.append(capture_generation_state(grid, grid[current_x][current_y], exploration_path, [], active_cells))

    exit_y = height - 1
    grid[exit_x][exit_y].is_end = True

    directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]
    opposite = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

    total_cells_count = width * height
    visited_count = 1

    while visited_count < total_cells_count:
        walked = False
        while True:
            unvisited_neighbors = []
            for direction, dx, dy in directions:
                nx, ny = current_x + dx, current_y + dy
                if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny].visited:
                    unvisited_neighbors.append((nx, ny, direction))

            if not unvisited_neighbors:
                 break

            nx, ny, direction = random.choice(unvisited_neighbors)
            grid[current_x][current_y].walls[direction] = False
            grid[nx][ny].walls[opposite[direction]] = False
            grid[nx][ny].visited = True
            grid[nx][ny].generation_order = cells_added
            cells_added += 1
            visited_count += 1

            current_x, current_y = nx, ny

            exploration_path.append(grid[current_x][current_y])
            if len(exploration_path) > max_path_length:
                exploration_path = exploration_path[-max_path_length:]

            active_cells = [(current_x, current_y)]
            generation_states.append(capture_generation_state(grid, grid[current_x][current_y], exploration_path, [], active_cells))
            walked = True

        if visited_count < total_cells_count:
            hunt_successful = False
            hunted_from_cell = None

            for y in range(height):
                for x in range(width):
                    if not grid[x][y].visited:
                        visited_neighbors = []
                        for direction, dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height and grid[nx][ny].visited:
                                visited_neighbors.append((nx, ny, direction))

                        if visited_neighbors:
                            nx, ny, direction = random.choice(visited_neighbors)

                            grid[nx][ny].walls[opposite[direction]] = False
                            grid[x][y].walls[direction] = False
                            grid[x][y].visited = True
                            grid[x][y].generation_order = cells_added
                            cells_added += 1
                            visited_count += 1

                            current_x, current_y = x, y
                            hunted_from_cell = (nx, ny)

                            exploration_path.append(grid[nx][ny])
                            exploration_path.append(grid[x][y])
                            if len(exploration_path) > max_path_length:
                                exploration_path = exploration_path[-max_path_length:]

                            active_cells = [(nx, ny), (x, y)]
                            generation_states.append(capture_generation_state(grid, grid[current_x][current_y], exploration_path, [], active_cells))
                            hunt_successful = True
                            break
                if hunt_successful:
                    break

            if not hunt_successful and visited_count < total_cells_count:
                 print("Error: Hunt phase failed to find a connection while unvisited cells remain.")
                 generation_states.append(capture_generation_state(grid, None, exploration_path, [], active_cells))
                 break

    if 0 <= exit_x < width and 0 <= exit_y < height:
         grid[exit_x][exit_y].walls['S'] = False

    for x in range(width):
        if 0 <= x < width and 0 < height:
           if x != entrance_x:
                 grid[x][0].walls['N'] = True
        if 0 <= x < width and height > 1:
            if x != exit_x:
                 grid[x][height-1].walls['S'] = True

    for y in range(height):
        if 0 < width:
            grid[0][y].walls['W'] = True
        if width > 1:
            grid[width-1][y].walls['E'] = True

    if 0 <= entrance_x < width:
         grid[entrance_x][0].walls['N'] = False
    if 0 <= exit_x < width:
        if 0 <= exit_y < height:
            grid[exit_x][exit_y].walls['S'] = False

    generation_states.append(capture_generation_state(grid, None, [], []))
    entrance_pos = (entrance_x, 0)
    exit_pos = (exit_x, height - 1)
    return grid, generation_states, cells_added, entrance_pos, exit_pos


def solve_maze_greedy(grid, start_pos, end_pos):
    width = len(grid)
    height = len(grid[0])
    start_x, start_y = start_pos
    end_x, end_y = end_pos

    if 0 <= start_x < width and 0 <= start_y < height:
        grid[start_x][start_y].is_start = True
    else:
        print(f"Warning: Start position {start_pos} is out of bounds.")


    if 0 <= end_x < width and 0 <= end_y < height:
        grid[end_x][end_y].is_end = True
    else:
         print(f"Warning: End position {end_pos} is out of bounds.")


    for x in range(width):
        for y in range(height):
            grid[x][y].visited = False
            grid[x][y].in_path = False

    count = 0
    solving_states = []
    exploration_path = []
    active_cells = []
    solution_found = False
    final_path = []
    came_from = {}
    visited = set()


    if 0 <= start_x < width and 0 <= start_y < height:
        priority_queue = [(manhattan_distance((start_x, start_y), end_pos), count, (start_x, start_y))]
        visited.add((start_x, start_y))
        grid[start_x][start_y].visited = True
        exploration_path = [grid[start_x][start_y]]
        active_cells = [(start_x, start_y)]
        solving_states.append(capture_solving_state(grid, grid[start_x][start_y],
                              exploration_path, [], False, active_cells))
    else:
        print("Error: Cannot start solving maze due to invalid start position.")
        solving_states.append(capture_solving_state(grid, None, [], [], False, []))
        return solving_states


    while priority_queue and not solution_found:
        try:
            _, _, current_pos = heapq.heappop(priority_queue)
            x, y = current_pos
        except IndexError:
            print("Error: Priority queue became empty unexpectedly.")
            break


        exploration_path.append(grid[x][y])
        if len(exploration_path) > 20:
            exploration_path = exploration_path[-20:]

        if current_pos == end_pos:
            solution_found = True
            temp_pos = current_pos
            while temp_pos in came_from:
                final_path.append(temp_pos)
                next_pos = came_from[temp_pos]
                if not isinstance(next_pos, tuple) or len(next_pos) != 2:
                     print(f"Error: Invalid 'came_from' value encountered: {next_pos}")
                     solution_found = False
                     break
                temp_pos = next_pos


            if solution_found:
                 final_path.append(start_pos)
                 final_path.reverse()


            active_cells = [(end_x, end_y)] if 0 <= end_x < width and 0 <= end_y < height else []
            end_cell = grid[end_x][end_y] if 0 <= end_x < width and 0 <= end_y < height else None
            solving_states.append(capture_solving_state(grid, end_cell,
                                  exploration_path, [], False, active_cells))
            continue

        frontier_positions = []
        active_cells = [(x, y)]
        directions = [('N', 0, -1), ('E', 1, 0), ('S', 0, 1), ('W', -1, 0)]

        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                can_move = False
                current_cell_walls = grid[x][y].walls
                neighbor_cell_walls = grid[nx][ny].walls


                if direction == 'N' and not current_cell_walls.get('N', True) and not neighbor_cell_walls.get('S', True):
                     can_move = True
                elif direction == 'S' and not current_cell_walls.get('S', True) and not neighbor_cell_walls.get('N', True):
                      can_move = True
                elif direction == 'E' and not current_cell_walls.get('E', True) and not neighbor_cell_walls.get('W', True):
                     can_move = True
                elif direction == 'W' and not current_cell_walls.get('W', True) and not neighbor_cell_walls.get('E', True):
                     can_move = True


                if can_move:
                    visited.add((nx, ny))
                    grid[nx][ny].visited = True
                    came_from[(nx, ny)] = (x, y)
                    count += 1
                    priority = manhattan_distance((nx, ny), end_pos)
                    heapq.heappush(priority_queue, (priority, count, (nx, ny)))
                    frontier_positions.append((nx, ny))
                    active_cells.append((nx, ny))

        if not solution_found:
            solving_states.append(capture_solving_state(grid, grid[x][y],
                       exploration_path, frontier_positions, False, active_cells))

    if solution_found and final_path:
        end_cell = grid[end_x][end_y] if 0 <= end_x < width and 0 <= end_y < height else None
        for i in range(1, len(final_path) + 1):
             partial_path_coords = final_path[:i]
             partial_path_cells = []
             valid_path = True
             for p_idx, pos in enumerate(partial_path_coords):
                px, py = pos
                if 0 <= px < width and 0 <= py < height:
                    partial_path_cells.append(grid[px][py])
                else:
                    print(f"Error: Invalid coordinate {pos} at index {p_idx} in final path.")
                    valid_path = False
                    break


             if not valid_path:
                 break


             if end_cell:
                  solving_states.append(capture_solving_state(grid, end_cell,
                                      partial_path_cells, [], True, active_cells))
             else:
                 print("Error: Cannot capture final path state due to invalid end cell.")
                 break


        if valid_path:
            for pos in final_path:
                px, py = pos
                if 0 <= px < width and 0 <= py < height:
                    grid[px][py].in_path = True


            final_path_cells_valid = []
            for pos in final_path:
                 px, py = pos
                 if 0 <= px < width and 0 <= py < height:
                    final_path_cells_valid.append(grid[px][py])
                 else:
                    print(f"Warning: Skipping invalid pos {pos} in final path cells list.")


            if end_cell and final_path_cells_valid:
                 for _ in range(10):
                     solving_states.append(capture_solving_state(grid, end_cell,
                                 final_path_cells_valid, [], True))
            elif not end_cell:
                 print("Error: Cannot add final pause frames due to invalid end cell.")
            elif not final_path_cells_valid:
                 print("Warning: Final path list is empty or contains only invalid cells.")


    elif not solution_found:
         print("Solution not found.")
         last_cell = grid[x][y] if 'x' in locals() and 'y' in locals() and 0 <= x < width and 0 <= y < height else None
         solving_states.append(capture_solving_state(grid, last_cell, exploration_path, [], False, active_cells))

    return solving_states


def capture_generation_state(grid, current_cell, path, walls, active_cells=None):
    if not grid or not grid[0]:
        print("Error: Attempting to capture state from empty grid.")
        return { 'walls': [], 'visited': [], 'in_path': [], 'is_start': [], 'is_end': [],
                 'is_frontier': [], 'generation_order': [], 'current': None, 'path': [],
                 'walls_to_check': [], 'phase': 'generation', 'active_cells': [] }

    path_coords = [(cell.x, cell.y) for cell in path if isinstance(cell, Cell)]

    state = {
        'walls': [[cell.walls.copy() for cell in row] for row in grid],
        'visited': [[cell.visited for cell in row] for row in grid],
        'in_path': [[cell.in_path for cell in row] for row in grid],
        'is_start': [[cell.is_start for cell in row] for row in grid],
        'is_end': [[cell.is_end for cell in row] for row in grid],
        'is_frontier': [[cell.is_frontier for cell in row] for row in grid],
        'generation_order': [[cell.generation_order for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if current_cell else None,
        'path': path_coords,
        'walls_to_check': walls,
        'phase': 'generation',
        'active_cells': active_cells or []
    }
    return state

def capture_solving_state(grid, current_cell, path, frontier_positions, is_solution_phase=False, active_cells=None):
    if not grid or not grid[0]:
        print("Error: Attempting to capture solving state from empty grid.")
        return {'walls': [], 'visited': [], 'in_path': [],
                'is_start': [], 'is_end': [],
                'generation_order': [], 'current': None, 'path': [], 'frontier_positions': [],
                'phase': 'solving', 'is_solution_phase': is_solution_phase, 'active_cells': []}


    path_coords = set((cell.x, cell.y) for cell in path if isinstance(cell, Cell))
    in_path_grid = []
    for r_idx, row in enumerate(grid):
         row_path = []
         for c_idx, cell in enumerate(row):
            is_on_path = isinstance(cell, Cell) and ((cell.x, cell.y) in path_coords)
            row_path.append(is_on_path or (isinstance(cell, Cell) and cell.in_path))
         in_path_grid.append(row_path)


    state = {
        'walls': [[cell.walls.copy() if isinstance(cell, Cell) else {} for cell in row] for row in grid],
        'visited': [[cell.visited if isinstance(cell, Cell) else False for cell in row] for row in grid],
        'in_path': in_path_grid,
        'is_start': [[cell.is_start if isinstance(cell, Cell) else False for cell in row] for row in grid],
        'is_end': [[cell.is_end if isinstance(cell, Cell) else False for cell in row] for row in grid],
        'generation_order': [[cell.generation_order if isinstance(cell, Cell) else -1 for cell in row] for row in grid],
        'current': (current_cell.x, current_cell.y) if isinstance(current_cell, Cell) else None,
        'path': [(cell.x, cell.y) for cell in path if isinstance(cell, Cell)],
        'frontier_positions': frontier_positions,
        'phase': 'solving',
        'is_solution_phase': is_solution_phase,
        'active_cells': active_cells or []
    }
    return state


def create_gradient_color(order, total):
    if order < 0 or total <= 0:
        return (0.1, 0.1, 0.2)
    norm_pos = max(0.0, min(1.0, order / total))
    h1, s1, v1 = 0.75, 0.9, 0.5
    h2, s2, v2 = 0.55, 0.8, 0.9
    h = h1 + (h2 - h1) * norm_pos
    s = s1 + (s2 - s1) * norm_pos
    v = v1 + (v2 - v1) * norm_pos
    try:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (r, g, b)
    except Exception as e:
         print(f"Error creating gradient color: {e}")
         return (0.5, 0.5, 0.5)


def create_animation_frames(generation_states, solving_states):
    frames = []
    gen_total_states = len(generation_states)
    solve_total_states = len(solving_states)

    if gen_total_states > 0:
        for frame_idx in range(GEN_FRAMES):
            progress = frame_idx / max(1, GEN_FRAMES -1)
            state_idx = min(int(progress * gen_total_states), gen_total_states - 1)
            frames.append(generation_states[state_idx])
    else:
         print("Warning: No generation states to create frames from.")


    if solve_total_states > 0:
        num_final_pause_frames = 10
        num_exploration_states = max(0, solve_total_states - num_final_pause_frames)
        frames_for_exploration = max(1, SOLVE_FRAMES - 30)


        for frame_idx in range(frames_for_exploration):
            if num_exploration_states > 0:
                progress = frame_idx / max(1, frames_for_exploration -1)
                state_idx = min(int(progress * num_exploration_states), num_exploration_states - 1)
                frames.append(solving_states[state_idx])
            else:
                 frames.append(solving_states[0])


        num_solution_display_frames = 30
        if solve_total_states > 0:
             last_state_idx = solve_total_states - 1
             for _ in range(num_solution_display_frames):
                 frames.append(solving_states[last_state_idx])
        else:
             print("Warning: No solving states to create final frames from.")


    expected_total_frames = GEN_FRAMES + SOLVE_FRAMES
    if len(frames) < expected_total_frames:
         print(f"Warning: Generated {len(frames)} frames, expected {expected_total_frames}. Padding...")
         if frames:
            padding_needed = expected_total_frames - len(frames)
            last_frame = frames[-1]
            frames.extend([last_frame] * padding_needed)
         else:
             print("Error: Cannot pad frames as the list is empty.")


    elif len(frames) > expected_total_frames:
         print(f"Warning: Generated {len(frames)} frames, expected {expected_total_frames}. Truncating...")
         frames = frames[:expected_total_frames]


    return frames


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
         print(f"Warning: Invalid hex color format '{hex_color}'. Using default white.")
         return (1.0, 1.0, 1.0)
    try:
        return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))
    except ValueError:
         print(f"Error converting hex '{hex_color}' to RGB. Using default white.")
         return (1.0, 1.0, 1.0)

def create_animation(frames, width, height, total_cells, algorithm_name, solving_method, gen_particle_color, solving_particle_color):
    if not frames:
        print("Error: No frames provided for animation.")
        return None, None, None


    if width <= 0 or height <= 0:
         print(f"Error: Invalid maze dimensions ({width}x{height}).")
         return None, None, None


    cell_size = min(FIG_HEIGHT / height, FIG_WIDTH / width) * 0.9 if height > 0 and width > 0 else 10
    fig, axes = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    x_offset = (FIG_WIDTH - width * cell_size) / 2
    y_offset = (FIG_HEIGHT - height * cell_size) / 2
    particles = []


    def update(i):
        nonlocal particles
        if i >= len(frames):
            print(f"Warning: Frame index {i} out of bounds (total frames: {len(frames)}). Skipping update.")
            return

        axes.clear()
        axes.set_xlim(0, FIG_WIDTH)
        axes.set_ylim(0, FIG_HEIGHT)
        axes.set_facecolor(BG_COLOR)
        axes.axis('off')


        frame = frames[i]
        walls_data = frame.get('walls', [[{} for _ in range(height)] for _ in range(width)])
        visited_data = frame.get('visited', [[False for _ in range(height)] for _ in range(width)])
        in_path_data = frame.get('in_path', [[False for _ in range(height)] for _ in range(width)])
        is_start_data = frame.get('is_start', [[False for _ in range(height)] for _ in range(width)])
        is_end_data = frame.get('is_end', [[False for _ in range(height)] for _ in range(width)])
        generation_order_data = frame.get('generation_order', [[-1 for _ in range(height)] for _ in range(width)])
        current_pos = frame.get('current')
        path_positions = frame.get('path', [])
        phase = frame.get('phase', 'generation')
        active_cells = frame.get('active_cells', [])

        path_pos_list = path_positions if isinstance(path_positions, list) else []


        progress_percent = 0
        title_text = "Maze Generation"
        if phase == 'generation':
             progress_percent = min(100, int((i / max(1, GEN_FRAMES - 1)) * 100)) if GEN_FRAMES > 1 else 0
             title_text = "Maze Generation"
             subtitle1 = f"Algorithm: {algorithm_name}"
             subtitle2 = f"Progress: {progress_percent}%"
        else:
             progress_percent = min(100, int(((i - GEN_FRAMES) / max(1, SOLVE_FRAMES - 1)) * 100)) if SOLVE_FRAMES > 1 else 0
             is_solution_phase = frame.get('is_solution_phase', False)
             title_text = "Solution Path" if is_solution_phase else "Exploring Maze"
             subtitle1 = f"Algorithm: {solving_method}"
             subtitle2 = f"Progress: {progress_percent}%"


        text_y_base = y_offset + height * cell_size + 0.5
        axes.text(FIG_WIDTH/2, text_y_base + 1.5, title_text, color='white', fontsize=20, ha='center', weight='bold')
        axes.text(FIG_WIDTH/2, text_y_base + 0.8, subtitle1, color='white', fontsize=12, ha='center')
        axes.text(FIG_WIDTH/2, text_y_base + 0.3, subtitle2, color='white', fontsize=12, ha='center')

        particle_color_hex = gen_particle_color if phase == 'generation' else solving_particle_color
        particle_color_rgb = hex_to_rgb(particle_color_hex)
        num_particles = random.randint(5, 8) if phase == 'generation' else random.randint(6, 10)


        for ax, ay in active_cells:
             if 0 <= ax < width and 0 <= ay < height:
                cell_center_x = x_offset + ax * cell_size + cell_size/2
                cell_center_y = y_offset + (height - 1 - ay) * cell_size + cell_size/2
                for _ in range(num_particles):
                    start_x = cell_center_x + random.uniform(-0.3, 0.3) * cell_size
                    start_y = cell_center_y + random.uniform(-0.3, 0.3) * cell_size
                    particles.append(Particle(
                        start_x, start_y,
                        particle_color_rgb,
                        size=random.uniform(1.5, 2.0) * PARTICLE_SIZE,
                         lifetime=random.randint(5, 8)
                    ))
             else:
                  print(f"Warning: Active cell ({ax}, {ay}) is out of bounds.")


        pulse_factor = 1.0 + 0.1 * np.sin(i * PULSE_SPEED)

        cells_in_path = set()
        for pos in path_pos_list:
            if isinstance(pos, tuple) and len(pos) == 2:
                cells_in_path.add(pos)

        for x in range(width):
            for y in range(height):
                cell_x = x_offset + x * cell_size
                cell_y = y_offset + (height - 1 - y) * cell_size


                is_current = current_pos == (x, y) if current_pos else False
                in_path = in_path_data[x][y] if x < len(in_path_data) and y < len(in_path_data[x]) else False
                visited = visited_data[x][y] if x < len(visited_data) and y < len(visited_data[x]) else False
                is_start = is_start_data[x][y] if x < len(is_start_data) and y < len(is_start_data[x]) else False
                is_end = is_end_data[x][y] if x < len(is_end_data) and y < len(is_end_data[x]) else False
                generation_order = generation_order_data[x][y] if x < len(generation_order_data) and y < len(generation_order_data[x]) else -1

                in_exploration_path = (x, y) in cells_in_path

                in_frontier = (x, y) in frame.get('frontier_positions', []) if phase == 'solving' else False

                glow = False
                glow_size_factor = 1.0
                cell_color_hex = '#FFFFFF'
                alpha = 0.05
                zorder = 1

                if is_start:
                    cell_color_hex = START_COLOR
                    alpha = 1.0; zorder = 25;
                    glow = True; glow_color_hex = START_COLOR; glow_size_factor = 1.2 * pulse_factor
                elif is_end:
                    cell_color_hex = END_COLOR
                    alpha = 1.0;
                    zorder = 25; glow = True; glow_color_hex = END_COLOR; glow_size_factor = 1.2 * pulse_factor
                elif is_current:
                    cell_color_hex = gen_particle_color if phase == 'generation' else solving_particle_color
                    alpha = 0.9;
                    zorder = 20; glow = True; glow_color_hex = cell_color_hex; glow_size_factor = 1.15 * pulse_factor
                elif in_frontier:
                    cell_color_hex = solving_particle_color
                    alpha = 0.7;
                    zorder = 18; glow = True; glow_color_hex = solving_particle_color; glow_size_factor = 1.05
                elif in_path or (phase == 'generation' and in_exploration_path):
                    cell_color_hex = PATH_COLOR
                    is_final_display = (phase == 'solving' and
                                        'is_solution_phase' in frame and frame['is_solution_phase'] and
                                        i >= (GEN_FRAMES + SOLVE_FRAMES - 30))

                    if phase == 'generation' and in_exploration_path:
                        alpha = 0.7;
                        glow = True; glow_color_hex = gen_particle_color; glow_size_factor = 1.05
                    elif is_final_display:
                        alpha = 1.0;
                        glow = True; glow_color_hex = PATH_COLOR; glow_size_factor = 1.1 * pulse_factor
                    else:
                        alpha = 0.8;
                        glow = True; glow_color_hex = PATH_COLOR; glow_size_factor = 1.05 * pulse_factor
                    zorder = 15
                elif visited and not phase == 'generation':
                     cell_color_hex = VISITED_COLOR
                     alpha = 0.7; zorder = 8; glow = False
                elif visited and phase == 'generation':
                     effective_total = width * height if total_cells <=0 else total_cells
                     cell_color_rgb = create_gradient_color(generation_order, effective_total)
                     cell_color_hex = None
                     alpha = 0.7;
                     zorder = 5; glow = False
                else:
                    cell_color_hex = '#FFFFFF'
                    alpha = 0.05;
                    zorder = 1; glow = False


                final_cell_color = hex_to_rgb(cell_color_hex) if cell_color_hex else cell_color_rgb
                cell_rect = patches.Rectangle(
                    (cell_x, cell_y), cell_size, cell_size,
                     fill=True, color=final_cell_color, alpha=alpha,
                    linewidth=0, zorder=zorder
                )
                axes.add_patch(cell_rect)


                if glow:
                    glow_size = cell_size * glow_size_factor
                    glow_offset = (cell_size - glow_size) / 2
                    final_glow_color = hex_to_rgb(glow_color_hex)
                    glow_rect = patches.Rectangle(
                         (cell_x + glow_offset, cell_y + glow_offset),
                        glow_size, glow_size,
                        fill=False, edgecolor=final_glow_color, alpha=0.4,
                         linewidth=1.5,
                        zorder=zorder-1
                    )
                    axes.add_patch(glow_rect)


        new_particles = []
        for particle in particles:
            if particle.update():
                new_particles.append(particle)
                p_color = particle.color if isinstance(particle.color, tuple) and len(particle.color) == 4 else (1,1,1,0.5)
                p_alpha = p_color[3]
                circle = plt.Circle(
                    (particle.x, particle.y),
                    particle.size * cell_size * 0.07,
                     color=p_color[:3],
                    alpha=p_alpha,
                    zorder=100
                )
                axes.add_patch(circle)
        particles = new_particles


        wall_color_rgb = hex_to_rgb(WALL_COLOR)
        line_width = 1.0
        for x in range(width):
            for y in range(height):
                walls = walls_data[x][y] if x < len(walls_data) and y < len(walls_data[x]) else {}
                cell_x = x_offset + x * cell_size
                cell_y = y_offset + (height - 1 - y) * cell_size


                if walls.get('N', True):
                    axes.plot([cell_x, cell_x + cell_size], [cell_y + cell_size, cell_y + cell_size],
                              color=wall_color_rgb, linewidth=line_width, alpha=0.7, zorder=30)
                if walls.get('E', True):
                     axes.plot([cell_x + cell_size, cell_x + cell_size], [cell_y, cell_y + cell_size],
                              color=wall_color_rgb, linewidth=line_width, alpha=0.7, zorder=30)
                if walls.get('S', True):
                    axes.plot([cell_x, cell_x + cell_size], [cell_y, cell_y],
                              color=wall_color_rgb, linewidth=line_width, alpha=0.7, zorder=30)
                if walls.get('W', True):
                    axes.plot([cell_x, cell_x], [cell_y, cell_y + cell_size],
                           color=wall_color_rgb, linewidth=line_width, alpha=0.7, zorder=30)


    try:
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            blit=False,
             interval=max(1, 1000 / TARGET_FPS),
            repeat=False
        )
    except Exception as e:
         print(f"Error creating FuncAnimation: {e}")
         plt.close(fig)
         return None, None, None


    try:
        writer = animation.FFMpegWriter(
            fps=TARGET_FPS,
            metadata=dict(artist='Maze Generation & Solving'),
            bitrate=5000
        )
    except Exception as e:
         print(f"Error creating FFmpegWriter (ensure FFmpeg is installed and in PATH): {e}")
         plt.close(fig)
         return None, None, None


    return ani, writer, fig


def main():
    combo = {
            "generation_algo": "create_maze_huntandkill",
            "solving_algo": "solve_maze_greedy",
            "gen_name": "Hunt and Kill (Standard)",
            "solving_name": "Greedy Best-First Search",
            "gen_particle_color": HUNTANDKILL_GENERATION_PARTICLE,
            "solving_particle_color": GREEDY_SOLVING_PARTICLE,
            "output_file": "huntandkill_standard_greedy_maze.mp4"
    }

    print("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈")
    print(f"📱 GENERATING & SOLVING MAZE ANIMATION - {MAZE_WIDTH}x{MAZE_HEIGHT} MAZE 📱")
    print(f"Generation: {combo['gen_name']} - Solving: {combo['solving_name']}")
    print("≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈≈")

    print(f"🧩 Generating maze using {combo['gen_name']} ({MAZE_WIDTH}x{MAZE_HEIGHT})...")
    grid, generation_states, total_cells_generated, entrance_pos, exit_pos = create_maze_huntandkill(MAZE_WIDTH, MAZE_HEIGHT)

    if not grid:
        print("❌ Maze generation failed. Exiting.")
        return


    if not (0 <= entrance_pos[0] < MAZE_WIDTH and 0 <= entrance_pos[1] < MAZE_HEIGHT):
         print(f"❌ Invalid entrance position generated: {entrance_pos}. Cannot solve.")
         return
    if not (0 <= exit_pos[0] < MAZE_WIDTH and 0 <= exit_pos[1] < MAZE_HEIGHT):
         print(f"❌ Invalid exit position generated: {exit_pos}. Cannot solve.")
         return


    start_pos = entrance_pos
    end_pos = exit_pos


    print(f"🔍 Solving maze from {start_pos} to {end_pos} using {combo['solving_name']}...")
    solving_states = solve_maze_greedy(grid, start_pos, end_pos)


    if not solving_states:
         print("❌ Maze solving failed or produced no states. Cannot create animation.")
         return


    print(f"🎬 Creating animation frames (Target: {TOTAL_FRAMES})...")
    frames = create_animation_frames(generation_states, solving_states)


    if not frames:
         print("❌ Failed to create animation frames. Exiting.")
         return


    print(f"🎨 Building animation ({len(frames)} frames)...")
    ani, writer, fig = create_animation(
        frames, MAZE_WIDTH, MAZE_HEIGHT, total_cells_generated,
         combo['gen_name'], combo['solving_name'],
        combo['gen_particle_color'], combo['solving_particle_color']
    )


    if not ani or not writer or not fig:
        print("❌ Animation creation failed. Exiting.")
        if fig:
            plt.close(fig)
        return


    output_file = combo['output_file']
    print(f"💾 Saving animation to {output_file}...")
    print("    (This may take several minutes for high quality)")


    try:
        ani.save(output_file, writer=writer)
        print(f"✅ Animation saved successfully to {output_file}")
    except Exception as e:
        print(f"❌ Error saving animation: {e}")
        print("   Ensure FFmpeg is installed and accessible in your system's PATH.")
    finally:
        plt.close(fig)


    print("🚀 Process finished!")

if __name__ == "__main__":
    main()