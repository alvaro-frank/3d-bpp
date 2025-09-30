import os
import imageio
from utils.visualization import plot_bin 
from utils.visualization import create_gif
from environment.bin import Bin

def heuristic_blb_packing(bin_size, boxes, try_rotations=False, generate_gif=False, gif_name="packing_heuristic.gif"):
    """
    Heuristic Bottom-Left-Back (BLB) packing strategy for 3D Bin Packing Problem.

    Strategy:
    - Sort boxes by descending volume (largest-first).
    - Try to place each box starting from the "bottom-left" position of the bin.
    - Optionally test all 6 possible rotations of each box.
    - Iterate over candidate (x, y) positions until a feasible placement is found.

    Args:
    - bin_size (tuple): (width, height, depth) of the bin
    - boxes (list[Box]): list of Box objects to be packed
    - try_rotations (bool): if True, try all 6 orientations of each box
    - generate_gif (bool): if True, record packing steps as GIF
    - gif_name (str): filename for output GIF

    Returns:
    - placed_boxes (list[Box]): boxes successfully placed inside the bin
    - bin (Bin): the final bin object with placed boxes
    """
    bin = Bin(*bin_size)
    placed_boxes = []

    if generate_gif:
        gif_dir = "heuristic_gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0

    # Sort boxes largest-first (greedy heuristic)
    boxes = sorted(boxes, key=lambda box: box.get_volume(), reverse=True)

    # Try placing each box
    for box in boxes:
        placed = False
        
        # test 6 orientations or just the default
        rotations = range(6) if try_rotations else [0]

        # Test all rotations of the current box
        for rot in rotations:
            # Sweep through X-Y grid of the bin
            for x in range(bin.width):
                for y in range(bin.height):
                    position = (x, y)
                    # Try placing the box at (x,y) with rotation 'rot'
                    if bin.place_box(box, position, rot):
                        placed_boxes.append(box)
                        placed = True

                        if generate_gif:
                            frame_path = os.path.join(gif_dir, f"frame_{frame_count:04d}.png")
                            # Plot current bin state and save
                            plot_bin(bin.boxes, bin_size, save_path=frame_path,
                                     title=f"Placed box {len(placed_boxes)} at {position} rot={rot}")
                            frame_count += 1
                        
                        # Stop after successful placement
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
