import os
import imageio
from utils.visualization import plot_bin 
from environment.bin import Bin

def heuristic_blb_packing(bin_size, boxes, try_rotations=False, generate_gif=False, gif_name="packing_heuristic.gif"):
    bin = Bin(*bin_size)
    placed_boxes = []

    if generate_gif:
        gif_dir = "heuristic_gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0

    boxes = sorted(boxes, key=lambda box: box.get_volume(), reverse=True)

    for box in boxes:
        placed = False
        rotations = range(6) if try_rotations else [0]

        for rot in rotations:
            for x in range(bin.width):
                for y in range(bin.height):
                    position = (x, y)
                    if bin.place_box(box, position, rot):
                        placed_boxes.append(box)
                        placed = True

                        if generate_gif:
                            frame_path = os.path.join(gif_dir, f"frame_{frame_count:04d}.png")
                            # Plot current bin state and save
                            plot_bin(bin.boxes, bin_size, save_path=frame_path,
                                     title=f"Placed box {len(placed_boxes)} at {position} rot={rot}")
                            frame_count += 1

                        break
                if placed:
                    break
            if placed:
                break

    if generate_gif:
        create_gif(gif_dir, gif_name)
        # Cleanup
        import shutil
        shutil.rmtree(gif_dir)

    return placed_boxes, bin

def create_gif(frame_folder, gif_name="packing_heuristic.gif", fps=2):
    frames = []
    files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for file_name in files:
        image_path = os.path.join(frame_folder, file_name)
        frames.append(imageio.imread(image_path))
    imageio.mimsave(gif_name, frames, fps=fps)
