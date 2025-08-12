
import os
import imageio
from typing import List, Tuple
from utils.visualization import plot_bin 
from environment.bin import Bin
from environment.box import Box

def heuristic_blb_packing(
    bin_size: Tuple[int, int, int],
    boxes: List[Box],
    try_rotations: bool = True,
    generate_gif: bool = False,
    gif_name: str = "packing_heuristic.gif",
):
    """Bottom-Left-Back heuristic with Largest-First ordering.

    - Sort boxes by volume (desc).
    - For each box, scan (y, x) in increasing order (bottom-left), choose rotation that fits
      and yields the lowest supported z via Bin.find_lowest_z.
    - Place at (x, y, z) with the minimal z; ties broken by lower y then lower x.
    """
    bin = Bin(*bin_size)

    if generate_gif:
        gif_dir = "heuristic_gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0

    # Largest-first sorting, stable by id to be reproducible
    boxes_sorted = sorted(boxes, key=lambda b: (-(b.get_volume()), b.id if getattr(b, 'id', None) is not None else 0))

    rotations = [0,1,2,3] if try_rotations else [0]

    for box in boxes_sorted:
        best = None  # (z, y, x, rot)
        for y in range(bin.depth):     # bottom (0) -> up
            for x in range(bin.width): # left (0) -> right
                for rot in rotations:
                    bw, bh, bd = box.rotate(rot)
                    # early bounds check (z to be computed)
                    if x + bw > bin.width or y + bh > bin.height:
                        continue
                    z = bin.find_lowest_z((x, y, 0), (bw, bh, bd))
                    if z + bd > bin.depth:
                        continue
                    if not bin.collides((x, y, z), (bw, bh, bd)):
                        cand = (z, y, x, rot)
                        if best is None or cand < best:
                            best = cand
                # small shortcut: if we found a placement at z=0 for this (x,y), it's optimal for BLB
                if best and best[0] == 0:
                    break
            if best and best[0] == 0:
                break

        if best is None:
            # cannot place this box; skip
            continue

        z, y_best, x_best, rot_best = best
        bw, bh, bd = box.rotate(rot_best)
        box.rotation_type = rot_best
        box.place_at(x_best, y_best, z)
        bin.boxes.append(box)

        if generate_gif:
            frame_path = os.path.join(gif_dir, f"frame_{frame_count:04d}.png")
            plot_bin(bin.boxes, bin_size, save_path=frame_path,
                     title=f"Box {getattr(box,'id','?')} at {(x_best,y_best,z)} rot={rot_best}")
            frame_count += 1

    if generate_gif:
        create_gif(gif_dir, gif_name)
        # Cleanup
        import shutil
        shutil.rmtree(gif_dir)

    return bin.boxes, bin

def create_gif(frame_folder, gif_name="packing_heuristic.gif", fps=2):
    frames = []
    files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for file_name in files:
        image_path = os.path.join(frame_folder, file_name)
        frames.append(imageio.imread(image_path))
    imageio.mimsave(gif_name, frames, fps=fps)
