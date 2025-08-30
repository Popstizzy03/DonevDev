import subprocess
import time
import os
import random
from PIL import Image, ImageDraw
import math

class ImageSquare:
    def __init__(self, image_data, sort_value, original_pos):
        self.image_data = image_data
        self.sort_value = sort_value
        self.original_pos = original_pos
    
    def __lt__(self, other):
        return self.sort_value < other.sort_value
    
    def __le__(self, other):
        return self.sort_value <= other.sort_value
    
    def __gt__(self, other):
        return self.sort_value > other.sort_value
    
    def __ge__(self, other):
        return self.sort_value >= other.sort_value
    
    def __eq__(self, other):
        return self.sort_value == other.sort_value

class TikTokMergeSortVisualizer:
    def __init__(self, image_path):
        self.width = 1080
        self.height = 1920
        self.background_color = "#0A0A15"
        self.image_content_size = 850
        self.border_width = 8
        self.total_square_size = self.image_content_size + (2 * self.border_width)
        self.output_file = "merge_sort_tiktok.mp4"
        self.image_path = image_path
        self.grid_size = 10
        self.piece_size = self.image_content_size // self.grid_size
        self.squares = []
        self.frames_dir = "merge_sort_frames"
        self.frame_count = 0
        self.fps = 8
        
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
    
    def load_and_prepare_image(self):
        try:
            img = Image.open(self.image_path)
            img = img.convert('RGB')
            img = img.resize((self.image_content_size, self.image_content_size))
            
            self.squares = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    left = col * self.piece_size
                    top = row * self.piece_size
                    right = left + self.piece_size
                    bottom = top + self.piece_size
                    
                    piece = img.crop((left, top, right, bottom))
                    
                    sort_value = row * self.grid_size + col
                    original_pos = (row, col)
                    
                    square = ImageSquare(piece, sort_value, original_pos)
                    self.squares.append(square)
            
            random.shuffle(self.squares)
            print(f"Image split into {len(self.squares)} pieces and shuffled")
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def create_frame(self, squares_list=None, highlight_indices=None, comparison_indices=None, 
                    merge_range=None, swap_indices=None):
        if squares_list is None:
            squares_list = self.squares
            
        canvas = Image.new('RGB', (self.width, self.height), self.background_color)
        draw = ImageDraw.Draw(canvas)
        
        total_square_x = (self.width - self.total_square_size) // 2
        total_square_y = (self.height - self.total_square_size) // 2
        
        content_x = total_square_x + self.border_width
        content_y = total_square_y + self.border_width
        
        draw.rectangle([total_square_x, total_square_y, 
                       total_square_x + self.total_square_size, 
                       total_square_y + self.total_square_size], 
                      fill='white')
        
        draw.rectangle([content_x, content_y,
                       content_x + self.image_content_size,
                       content_y + self.image_content_size],
                      fill=self.background_color)
        
        for i, square in enumerate(squares_list):
            row = i // self.grid_size
            col = i % self.grid_size
            
            x = content_x + col * self.piece_size
            y = content_y + row * self.piece_size
            
            canvas.paste(square.image_data, (x, y))
            
            if swap_indices and i in swap_indices:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='#00FF00', width=5)
            elif highlight_indices and i in highlight_indices:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='red', width=4)
            elif comparison_indices and i in comparison_indices:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='yellow', width=3)
            elif merge_range and merge_range[0] <= i <= merge_range[1]:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='blue', width=2)
        
        frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.png")
        canvas.save(frame_path)
        self.frame_count += 1
        
        return frame_path
    
    def merge_sort_visual(self):
        self.create_frame()
        
        for _ in range(15):
            self.create_frame()
        
        return self._merge_sort_recursive(0, len(self.squares) - 1)
    
    def _merge_sort_recursive(self, left, right):
        if left >= right:
            return
        
        mid = (left + right) // 2
        
        self.create_frame(merge_range=(left, right))
        
        self._merge_sort_recursive(left, mid)
        self._merge_sort_recursive(mid + 1, right)
        
        self._merge_visual_with_swaps(left, mid, right)
    
    def _merge_visual_with_swaps(self, left, mid, right):
        self.create_frame(merge_range=(left, right))
        
        temp_squares = []
        for i in range(left, right + 1):
            temp_squares.append(self.squares[i])
        
        position_map = {}
        for i in range(left, right + 1):
            position_map[id(self.squares[i])] = i
        
        i = 0
        j = mid - left + 1
        k = left
        
        left_size = mid - left + 1
        
        while i < left_size and j < len(temp_squares):
            left_elem = temp_squares[i]
            right_elem = temp_squares[j]
            
            left_pos = position_map[id(left_elem)]
            right_pos = position_map[id(right_elem)]
            
            self.create_frame(comparison_indices=[left_pos, right_pos], merge_range=(left, right))
            
            if left_elem.sort_value <= right_elem.sort_value:
                target_elem = left_elem
                source_pos = left_pos
                i += 1
            else:
                target_elem = right_elem
                source_pos = right_pos
                j += 1
            
            if source_pos != k:
                self.create_frame(swap_indices=[source_pos, k], merge_range=(left, right))
                
                elem_at_k = self.squares[k]
                self.squares[k] = target_elem
                self.squares[source_pos] = elem_at_k
                
                position_map[id(target_elem)] = k
                position_map[id(elem_at_k)] = source_pos
                
                self.create_frame(highlight_indices=[k], merge_range=(left, right))
            else:
                self.create_frame(highlight_indices=[k], merge_range=(left, right))
            
            k += 1
        
        while i < left_size:
            left_elem = temp_squares[i]
            source_pos = position_map[id(left_elem)]
            
            if source_pos != k:
                self.create_frame(swap_indices=[source_pos, k], merge_range=(left, right))
                
                elem_at_k = self.squares[k]
                self.squares[k] = left_elem
                self.squares[source_pos] = elem_at_k
                
                position_map[id(left_elem)] = k
                position_map[id(elem_at_k)] = source_pos
                
                self.create_frame(highlight_indices=[k], merge_range=(left, right))
            else:
                self.create_frame(highlight_indices=[k], merge_range=(left, right))
            
            i += 1
            k += 1
        
        while j < len(temp_squares):
            right_elem = temp_squares[j]
            source_pos = position_map[id(right_elem)]
            
            if source_pos != k:
                self.create_frame(swap_indices=[source_pos, k], merge_range=(left, right))
                
                elem_at_k = self.squares[k]
                self.squares[k] = right_elem
                self.squares[source_pos] = elem_at_k
                
                position_map[id(right_elem)] = k
                position_map[id(elem_at_k)] = source_pos
                
                self.create_frame(highlight_indices=[k], merge_range=(left, right))
            else:
                self.create_frame(highlight_indices=[k], merge_range=(left, right))
            
            j += 1
            k += 1
        
        merged_indices = list(range(left, right + 1))
        self.create_frame(highlight_indices=merged_indices)
        
        for _ in range(3):
            self.create_frame()
    
    def create_video_from_frames(self):
        if self.frame_count == 0:
            print("No frames generated!")
            return False
        
        for _ in range(30):
            self.create_frame()
        
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(self.fps),
            "-i", os.path.join(self.frames_dir, "frame_%06d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1080:1920",
            self.output_file
        ]
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode == 0:
                print(f"Video created successfully: {self.output_file}")
                return True
            else:
                print(f"FFmpeg error: {process.stderr}")
                return False
        except Exception as e:
            print(f"Error creating video: {e}")
            return False
    
    def cleanup_frames(self):
        try:
            for filename in os.listdir(self.frames_dir):
                if filename.startswith("frame_") and filename.endswith(".png"):
                    os.remove(os.path.join(self.frames_dir, filename))
            os.rmdir(self.frames_dir)
            print("Temporary frames cleaned up")
        except Exception as e:
            print(f"Error cleaning up frames: {e}")
    
    def generate(self):
        print("Starting merge sort visualization...")
        
        if not self.load_and_prepare_image():
            return False
        
        print("Creating initial scrambled frame...")
        self.create_frame()
        
        print("Starting merge sort visualization with visible swaps...")
        self.merge_sort_visual()
        
        print(f"Generated {self.frame_count} frames")
        print("Creating video from frames...")
        
        success = self.create_video_from_frames()
        
        self.cleanup_frames()
        
        if success:
            print(f"Merge sort visualization complete! Video saved as: {self.output_file}")
            print(f"The video shows {len(self.squares)} image pieces being sorted")
            print("All pieces remain visible throughout - no duplicates!")
        
        return success

def main():
    image_path = "image.png"
    
    if not os.path.exists(image_path):
        print(f"Please place an image file at: {image_path}")
        print("Or modify the image_path variable to point to your image")
        return
    
    generator = TikTokMergeSortVisualizer(image_path)
    generator.generate()

if __name__ == "__main__":
    main()