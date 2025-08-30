import subprocess
import os
import random
import shutil
from PIL import Image, ImageDraw


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


class TikTokQuickSortVisualizer:
    def __init__(self, image_path):
        self.width = 1080
        self.height = 1920
        self.background_color = "#0A0A15"
        self.image_content_size = 850
        self.border_width = 8
        self.total_square_size = self.image_content_size + (2 * self.border_width)
        self.output_file = "quick_sort_tiktok.mp4"
        self.image_path = image_path
        self.grid_size = 10
        self.piece_size = self.image_content_size // self.grid_size
        self.squares = []
        self.frames_dir = "quick_sort_frames"
        self.frame_count = 0
        self.fps = 8
        self.sorted_elements = set()
        
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
    
    def create_frame(self, squares_list=None, pivot_index=None, comparison_indices=None, 
                    partition_range=None, swap_indices=None, partition_line=None,
                    sorted_indices=None):
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
        
        if partition_line is not None:
            line_row = partition_line // self.grid_size
            line_col = partition_line % self.grid_size
            line_x = content_x + line_col * self.piece_size
            line_y = content_y + line_row * self.piece_size
            
            if line_col < self.grid_size - 1:
                draw.line([(line_x + self.piece_size, content_y),
                          (line_x + self.piece_size, content_y + self.image_content_size)],
                         fill='#00FFFF', width=3)
        
        for i, square in enumerate(squares_list):
            row = i // self.grid_size
            col = i % self.grid_size
            
            x = content_x + col * self.piece_size
            y = content_y + row * self.piece_size
            
            canvas.paste(square.image_data, (x, y))
        
        for i, square in enumerate(squares_list):
            row = i // self.grid_size
            col = i % self.grid_size
            
            x = content_x + col * self.piece_size
            y = content_y + row * self.piece_size
            
            if pivot_index is not None and i == pivot_index:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='#FF00FF', width=6)
            elif swap_indices and i in swap_indices:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='#00FF00', width=5)
            elif comparison_indices and i in comparison_indices:
                draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                             outline='#FFFF00', width=4)
            elif partition_range and partition_range[0] <= i <= partition_range[1]:
                if not (sorted_indices and i in sorted_indices):
                    draw.rectangle([x, y, x + self.piece_size, y + self.piece_size],
                                 outline='#0080FF', width=2)
        
        frame_path = os.path.join(self.frames_dir, f"frame_{self.frame_count:06d}.png")
        canvas.save(frame_path)
        self.frame_count += 1
        
        return frame_path
    
    def quick_sort_visual(self):
        self.create_frame()
        
        for _ in range(15):
            self.create_frame()
        
        self.sorted_elements = set()
        
        return self._quick_sort_recursive(0, len(self.squares) - 1)
    
    def _quick_sort_recursive(self, low, high):
        if low >= high:
            if low == high:
                self.sorted_elements.add(low)
                self.create_frame(sorted_indices=self.sorted_elements)
            return
        
        self.create_frame(partition_range=(low, high), sorted_indices=self.sorted_elements)
        
        pivot_pos = self._partition_visual(low, high)
        
        self.sorted_elements.add(pivot_pos)
        
        self.create_frame(partition_line=pivot_pos, sorted_indices=self.sorted_elements)
        
        for _ in range(5):
            self.create_frame(partition_line=pivot_pos, sorted_indices=self.sorted_elements)
        
        self._quick_sort_recursive(low, pivot_pos - 1)
        self._quick_sort_recursive(pivot_pos + 1, high)
    
    def _partition_visual(self, low, high):
        pivot_idx = high
        pivot_value = self.squares[pivot_idx].sort_value
        
        for _ in range(8):
            self.create_frame(pivot_index=pivot_idx, partition_range=(low, high),
                            sorted_indices=self.sorted_elements)
        
        i = low - 1
        
        for j in range(low, high):
            self.create_frame(pivot_index=pivot_idx, comparison_indices=[j],
                            partition_range=(low, high), sorted_indices=self.sorted_elements)
            
            if self.squares[j].sort_value <= pivot_value:
                i += 1
                
                if i != j:
                    self.create_frame(pivot_index=pivot_idx, swap_indices=[i, j],
                                    partition_range=(low, high), sorted_indices=self.sorted_elements)
                    
                    self.squares[i], self.squares[j] = self.squares[j], self.squares[i]
                    
                    self.create_frame(pivot_index=pivot_idx, partition_range=(low, high),
                                    sorted_indices=self.sorted_elements)
                else:
                    self.create_frame(pivot_index=pivot_idx, partition_range=(low, high),
                                    sorted_indices=self.sorted_elements)
        
        final_pivot_pos = i + 1
        
        if final_pivot_pos != pivot_idx:
            self.create_frame(swap_indices=[final_pivot_pos, pivot_idx],
                            partition_range=(low, high), sorted_indices=self.sorted_elements)
            
            self.squares[final_pivot_pos], self.squares[pivot_idx] = \
                self.squares[pivot_idx], self.squares[final_pivot_pos]
        
        self.create_frame(pivot_index=final_pivot_pos, partition_range=(low, high),
                        sorted_indices=self.sorted_elements)
        
        return final_pivot_pos
    
    def create_video_from_frames(self):
        if self.frame_count == 0:
            print("No frames generated!")
            return False
        
        for _ in range(30):
            self.create_frame(sorted_indices=set(range(len(self.squares))))
        
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
            if os.path.exists(self.frames_dir):
                for filename in os.listdir(self.frames_dir):
                    file_path = os.path.join(self.frames_dir, filename)
                    try:
                        os.remove(file_path)
                    except:
                        pass
                
                os.rmdir(self.frames_dir)
                print("Temporary frames folder removed successfully")
        except Exception as e:
            print(f"Warning: Could not fully clean up frames folder: {e}")
            import shutil
            try:
                shutil.rmtree(self.frames_dir)
                print("Frames folder force-removed")
            except:
                pass
    
    def generate(self):
        print("Starting Quick Sort visualization...")
        
        if not self.load_and_prepare_image():
            return False
        
        try:
            print("Creating initial scrambled frame...")
            self.create_frame()
            
            print("Starting Quick Sort visualization...")
            print("Purple = Pivot | Yellow = Comparing | Green = Swapping")
            print("Blue borders = Current partition | Cyan lines = Partition boundaries")
            print("Sorted pieces blend seamlessly into the final image")
            
            self.quick_sort_visual()
            
            print(f"Generated {self.frame_count} frames")
            print("Creating video from frames...")
            
            success = self.create_video_from_frames()
            
            if success:
                print(f"Quick Sort visualization complete! Video saved as: {self.output_file}")
                print(f"The video shows {len(self.squares)} image pieces being sorted using Quick Sort")
                print("Watch for the partitioning process and how the pivot finds its final position!")
            
            return success
            
        finally:
            self.cleanup_frames()


def main():
    image_path = "image.png"
    
    if not os.path.exists(image_path):
        print(f"Please place an image file at: {image_path}")
        print("Or modify the image_path variable to point to your image")
        return
    
    generator = TikTokQuickSortVisualizer(image_path)
    generator.generate()


if __name__ == "__main__":
    main()