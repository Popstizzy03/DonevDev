import os
import random
import numpy as np
import cv2
import math
import time
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Set, Dict


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, 
                 color: Tuple[int, int, int], life: float, size: float = 2.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life
        self.size = size
        self.gravity = 0.1
        
    def update(self, dt: float = 1.0):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += self.gravity * dt
        self.life -= dt / 15.0
        
    def is_alive(self) -> bool:
        return self.life > 0
        
    def get_alpha(self) -> float:
        return max(0, self.life / self.max_life)


class ParticleSystem:
    def __init__(self):
        self.particles: List[Particle] = []
        
    def add_particle(self, particle: Particle):
        self.particles.append(particle)
        
    def add_border_particles(self, x: int, y: int, width: int, height: int, 
                           color: Tuple[int, int, int], count: int = 20):
        border_points = []
        
        for i in range(count // 4):
            px = x + (width * i / (count // 4))
            border_points.append((px, y))
            px = x + (width * i / (count // 4))
            border_points.append((px, y + height))
            
        for i in range(count // 4):
            py = y + (height * i / (count // 4))
            border_points.append((x, py))
            py = y + (height * i / (count // 4))
            border_points.append((x + width, py))
            
        for px, py in border_points:
            vx = random.uniform(-2, 2)
            vy = random.uniform(-3, -1)
            life = random.uniform(0.8, 1.2)
            size = random.uniform(1.5, 3.0)
            
            particle = Particle(px, py, vx, vy, color, life, size)
            self.add_particle(particle)
    
    def add_celebration_particles(self, x: int, y: int, width: int, height: int, 
                                count: int = 50):
        for _ in range(count):
            side = random.randint(0, 3)
            if side == 0:
                px = random.uniform(x, x + width)
                py = y
            elif side == 1:
                px = x + width
                py = random.uniform(y, y + height)
            elif side == 2:
                px = random.uniform(x, x + width)
                py = y + height
            else:
                px = x
                py = random.uniform(y, y + height)
            
            vx = random.uniform(-3, 3)
            vy = random.uniform(-4, -1)
            
            grey = random.randint(120, 180)
            color = (grey, grey, grey)
            
            life = random.uniform(1.0, 2.0)
            size = random.uniform(2.0, 4.0)
            
            particle = Particle(px, py, vx, vy, color, life, size)
            self.add_particle(particle)
    
    def update(self):
        self.particles = [p for p in self.particles if p.is_alive()]
        for particle in self.particles:
            particle.update()
            
    def draw(self, draw: ImageDraw.Draw):
        for particle in self.particles:
            alpha = particle.get_alpha()
            if alpha > 0:
                r, g, b = particle.color
                color = (int(r * alpha), int(g * alpha), int(b * alpha))
                
                size = particle.size * alpha
                x1 = int(particle.x - size)
                y1 = int(particle.y - size)
                x2 = int(particle.x + size)
                y2 = int(particle.y + size)
                
                try:
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                except:
                    pass
    
    def clear(self):
        self.particles.clear()


class SwapAnimation:
    def __init__(self, piece1_idx: int, piece2_idx: int, 
                 pos1: Tuple[int, int], pos2: Tuple[int, int], 
                 piece1_data: Tuple[int, Image.Image], piece2_data: Tuple[int, Image.Image],
                 total_frames: int = 8):
        self.piece1_idx = piece1_idx
        self.piece2_idx = piece2_idx
        self.start_pos1 = pos1
        self.start_pos2 = pos2
        self.end_pos1 = pos2
        self.end_pos2 = pos1
        self.piece1_data = piece1_data
        self.piece2_data = piece2_data
        self.total_frames = total_frames
        self.current_frame = 0
        self.completed = False
    
    def get_current_positions(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if self.current_frame >= self.total_frames:
            self.completed = True
            return self.end_pos1, self.end_pos2
        
        t = self.current_frame / self.total_frames
        t = t * t * (3.0 - 2.0 * t)
        
        pos1_x = int(self.start_pos1[0] + (self.end_pos1[0] - self.start_pos1[0]) * t)
        pos1_y = int(self.start_pos1[1] + (self.end_pos1[1] - self.start_pos1[1]) * t)
        
        pos2_x = int(self.start_pos2[0] + (self.end_pos2[0] - self.start_pos2[0]) * t)
        pos2_y = int(self.start_pos2[1] + (self.end_pos2[1] - self.start_pos2[1]) * t)
        
        return (pos1_x, pos1_y), (pos2_x, pos2_y)
    
    def advance_frame(self):
        self.current_frame += 1
    
    def is_completed(self) -> bool:
        return self.completed


class SortingVisualizer:
    def __init__(self, image_path: str, output_file: str = "sorting_visualization.mp4"):
        self.width = 1080
        self.height = 1920
        self.fps = 60
        self.output_file = output_file
        
        self.grid_size = 10
        self.image_size = 900
        self.piece_size = self.image_size // self.grid_size
        
        self.background_color = (20, 20, 30)
        self.border_color = (255, 255, 255)
        self.border_width = 5
        
        self.swap_frames = 8
        self.instant_popup = True
        
        self.pieces: List[Tuple[int, Image.Image]] = []
        self.image_path = image_path
        
        self.current_swap_animation: Optional[SwapAnimation] = None
        
        self.particle_system = ParticleSystem()
        
        self.video_writer = None
        self.frame_count = 0
        
        self.content_x = (self.width - self.image_size) // 2
        self.content_y = (self.height - self.image_size) // 2
        
    def load_and_split_image(self) -> bool:
        try:
            img = Image.open(self.image_path)
            img = img.convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            
            self.pieces = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    left = col * self.piece_size
                    top = row * self.piece_size
                    right = left + self.piece_size
                    bottom = top + self.piece_size
                    
                    piece = img.crop((left, top, right, bottom))
                    
                    original_index = row * self.grid_size + col
                    
                    self.pieces.append((original_index, piece))
            
            print(f"Image split into {len(self.pieces)} pieces")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def shuffle_pieces(self):
        random.shuffle(self.pieces)
        print("Pieces shuffled")
    
    def get_piece_position(self, piece_index: int) -> Tuple[int, int]:
        row = piece_index // self.grid_size
        col = piece_index % self.grid_size
        
        x = self.content_x + col * self.piece_size
        y = self.content_y + row * self.piece_size
        
        return x, y
    
    def swap_pieces(self, i: int, j: int):
        if i != j and 0 <= i < len(self.pieces) and 0 <= j < len(self.pieces):
            pos1 = self.get_piece_position(i)
            pos2 = self.get_piece_position(j)
            
            if self.instant_popup:
                self._add_swap_particles(i)
                self._add_swap_particles(j)
            
            self.current_swap_animation = SwapAnimation(
                i, j, pos1, pos2, 
                self.pieces[i], self.pieces[j],
                self.swap_frames
            )
            
            while not self.current_swap_animation.is_completed():
                self.create_frame()
                self.current_swap_animation.advance_frame()
            
            self.pieces[i], self.pieces[j] = self.pieces[j], self.pieces[i]
            
            self.current_swap_animation = None
            
            self.create_frame()
    
    def _add_swap_particles(self, piece_index: int):
        x, y = self.get_piece_position(piece_index)

        grey = random.randint(120, 180)
        color = (grey, grey, grey)
        
        self.particle_system.add_border_particles(
            x, y, self.piece_size, self.piece_size, color, count=8
        )
    
    def create_frame(self, highlight_indices: Optional[List[int]] = None):
        self.particle_system.update()
        
        canvas = Image.new('RGB', (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        border_x = self.content_x - self.border_width
        border_y = self.content_y - self.border_width
        border_size = self.image_size + 2 * self.border_width
        
        draw.rectangle([border_x, border_y, 
                       border_x + border_size, 
                       border_y + border_size],
                      fill=self.border_color)
        
        draw.rectangle([self.content_x, self.content_y,
                       self.content_x + self.image_size,
                       self.content_y + self.image_size],
                      fill=self.background_color)
        
        if self.current_swap_animation is not None:
            for position, (original_index, piece) in enumerate(self.pieces):
                if position not in [self.current_swap_animation.piece1_idx, self.current_swap_animation.piece2_idx]:
                    x, y = self.get_piece_position(position)
                    canvas.paste(piece, (x, y))
                    
                    if highlight_indices and position in highlight_indices:
                        draw.rectangle([x, y, x + self.piece_size - 1, y + self.piece_size - 1],
                                     outline=(255, 100, 100), width=3)
            
            pos1, pos2 = self.current_swap_animation.get_current_positions()
            
            canvas.paste(self.current_swap_animation.piece1_data[1], pos1)
            if highlight_indices and self.current_swap_animation.piece1_idx in highlight_indices:
                draw.rectangle([pos1[0], pos1[1], 
                               pos1[0] + self.piece_size - 1, 
                               pos1[1] + self.piece_size - 1],
                             outline=(255, 100, 100), width=3)
            
            canvas.paste(self.current_swap_animation.piece2_data[1], pos2)
            if highlight_indices and self.current_swap_animation.piece2_idx in highlight_indices:
                draw.rectangle([pos2[0], pos2[1], 
                               pos2[0] + self.piece_size - 1, 
                               pos2[1] + self.piece_size - 1],
                             outline=(255, 100, 100), width=3)
        else:
            for position, (original_index, piece) in enumerate(self.pieces):
                x, y = self.get_piece_position(position)
                canvas.paste(piece, (x, y))
                
                if highlight_indices and position in highlight_indices:
                    draw.rectangle([x, y, x + self.piece_size - 1, y + self.piece_size - 1],
                                 outline=(255, 100, 100), width=3)
        
        self.particle_system.draw(draw)
        
        self.frame_count += 1
        
        frame_array = np.array(canvas)
        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame_bgr)
    
    def initialize_video(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_file, fourcc, self.fps, (self.width, self.height)
        )
        print(f"Video writer initialized: {self.output_file}")
    
    def finalize_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            cv2.destroyAllWindows()
            print(f"Video saved as: {self.output_file}")
    
    def add_pause_frames(self, num_frames: int = 60):
        for _ in range(num_frames):
            self.create_frame()
    
    def add_celebration_frames(self, num_frames: int = 120):
        border_x = self.content_x - self.border_width
        border_y = self.content_y - self.border_width
        border_width = self.image_size + 2 * self.border_width
        border_height = self.image_size + 2 * self.border_width
        
        for frame in range(num_frames):
            if frame % 8 == 0:
                self.particle_system.add_celebration_particles(
                    border_x, border_y, border_width, border_height, count=10
                )
            
            self.create_frame()
    
    def add_clean_final_frames(self, num_frames: int = 120):
        self.particle_system.clear()
        
        for _ in range(num_frames):
            self.create_frame()
    
    def is_sorted(self) -> bool:
        for i, (original_index, _) in enumerate(self.pieces):
            if original_index != i:
                return False
        return True
    
    def visualize_sorting(self, algorithm):
        print(f"Starting {algorithm.__class__.__name__} visualization...")
        
        self.initialize_video()
        
        if not self.load_and_split_image():
            return False
        
        self.shuffle_pieces()
        
        print("Showing initial shuffled state...")
        self.add_pause_frames(30)
        
        print("Running sorting algorithm...")
        algorithm.sort(self)
        
        print("Showing celebration with grey particles...")
        self.add_celebration_frames(120)
        
        print("Showing clean final state...")
        self.add_clean_final_frames(60)
        
        if self.is_sorted():
            print("✓ Image successfully sorted!")
        else:
            print("✗ Warning: Image may not be properly sorted")
        
        self.finalize_video()
        
        print(f"Visualization complete! Total frames: {self.frame_count}")
        return True


class SortingAlgorithm(ABC):
    @abstractmethod
    def sort(self, visualizer: SortingVisualizer):
        pass


class BubbleSort(SortingAlgorithm):
    def sort(self, visualizer: SortingVisualizer):
        n = len(visualizer.pieces)
        
        for i in range(n):
            swapped = False
            
            for j in range(0, n - i - 1):
                if visualizer.pieces[j][0] > visualizer.pieces[j + 1][0]:
                    visualizer.swap_pieces(j, j + 1)
                    swapped = True
            
            if not swapped:
                break


class SelectionSort(SortingAlgorithm):
    def sort(self, visualizer: SortingVisualizer):
        n = len(visualizer.pieces)
        
        for i in range(n):
            min_idx = i
            
            for j in range(i + 1, n):
                if visualizer.pieces[j][0] < visualizer.pieces[min_idx][0]:
                    min_idx = j
            
            if min_idx != i:
                visualizer.swap_pieces(i, min_idx)


class InsertionSort(SortingAlgorithm):
    def sort(self, visualizer: SortingVisualizer):
        n = len(visualizer.pieces)
        
        for i in range(1, n):
            j = i
            
            while j > 0 and visualizer.pieces[j][0] < visualizer.pieces[j - 1][0]:
                visualizer.swap_pieces(j, j - 1)
                j -= 1


class QuickSort(SortingAlgorithm):
    def sort(self, visualizer: SortingVisualizer):
        self._quick_sort(visualizer, 0, len(visualizer.pieces) - 1)
    
    def _quick_sort(self, visualizer: SortingVisualizer, low: int, high: int):
        if low < high:
            pi = self._partition(visualizer, low, high)
            self._quick_sort(visualizer, low, pi - 1)
            self._quick_sort(visualizer, pi + 1, high)
    
    def _partition(self, visualizer: SortingVisualizer, low: int, high: int) -> int:
        pivot = visualizer.pieces[high][0]
        i = low - 1
        
        for j in range(low, high):
            if visualizer.pieces[j][0] <= pivot:
                i += 1
                if i != j:
                    visualizer.swap_pieces(i, j)
        
        if i + 1 != high:
            visualizer.swap_pieces(i + 1, high)
        
        return i + 1


class MergeSort(SortingAlgorithm):
    def sort(self, visualizer: SortingVisualizer):
        self._merge_sort(visualizer, 0, len(visualizer.pieces) - 1)
    
    def _merge_sort(self, visualizer: SortingVisualizer, left: int, right: int):
        if left >= right:
            return
        
        mid = (left + right) // 2
        self._merge_sort(visualizer, left, mid)
        self._merge_sort(visualizer, mid + 1, right)
        self._merge(visualizer, left, mid, right)
    
    def _merge(self, visualizer: SortingVisualizer, left: int, mid: int, right: int):
        temp = []
        for i in range(left, right + 1):
            temp.append(visualizer.pieces[i])
        
        i = 0
        j = mid - left + 1
        k = left
        
        while i < (mid - left + 1) and j < len(temp):
            if temp[i][0] <= temp[j][0]:
                if visualizer.pieces[k] != temp[i]:
                    for idx in range(k, right + 1):
                        if visualizer.pieces[idx] == temp[i]:
                            visualizer.swap_pieces(k, idx)
                            break
                i += 1
            else:
                if visualizer.pieces[k] != temp[j]:
                    for idx in range(k, right + 1):
                        if visualizer.pieces[idx] == temp[j]:
                            visualizer.swap_pieces(k, idx)
                            break
                j += 1
            k += 1
        
        while i < (mid - left + 1):
            if visualizer.pieces[k] != temp[i]:
                for idx in range(k, right + 1):
                    if visualizer.pieces[idx] == temp[i]:
                        visualizer.swap_pieces(k, idx)
                        break
            i += 1
            k += 1
        
        while j < len(temp):
            if visualizer.pieces[k] != temp[j]:
                for idx in range(k, right + 1):
                    if visualizer.pieces[idx] == temp[j]:
                        visualizer.swap_pieces(k, idx)
                        break
            j += 1
            k += 1


def main():
    image_path = "gosling.jpg"
    
    if not os.path.exists(image_path):
        print(f"Please place an image file at: {image_path}")
        return
    
    algorithms = {
        '1': ('Bubble Sort', BubbleSort()),
        '2': ('Selection Sort', SelectionSort()),
        '3': ('Insertion Sort', InsertionSort()),
        '4': ('Quick Sort', QuickSort()),
        '5': ('Merge Sort', MergeSort())
    }
    
    print("Available sorting algorithms:")
    for key, (name, _) in algorithms.items():
        print(f"{key}. {name}")
    
    choice = input("Choose algorithm (1-5): ").strip()
    
    if choice in algorithms:
        name, algorithm = algorithms[choice]
        output_file = f"{name.lower().replace(' ', '_')}_visualization.mp4"
        
        visualizer = SortingVisualizer(image_path, output_file)
        visualizer.visualize_sorting(algorithm)
    else:
        print("Invalid choice. Using Bubble Sort as default.")
        visualizer = SortingVisualizer(image_path)
        visualizer.visualize_sorting(BubbleSort())


if __name__ == "__main__":
    main()