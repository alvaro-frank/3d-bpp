# ==============================================================================
# FILE: utils/box_generator.py
# DESCRIPTION: Utility for generating box datasets for 3D Bin Packing.
#              Supports structured recursive splitting (guaranteed perfect fit)
#              and purely random sampling.
# ==============================================================================

import random
import numpy as np
from copy import deepcopy

# ------------------------------------------------------------------------------
# BOX GENERATION LOGIC
# ------------------------------------------------------------------------------

def generate_boxes(bin_size, num_items=64, seed=None, structured=True):
    """
    Generate a list of boxes either by structured recursive splitting
    or purely random sampling.

    Args:
        bin_size (list[int]): Dimensions of the bin (width, depth, height).
        num_items (int): Number of boxes to generate.
        seed (int, optional): Random seed for reproducibility.
        structured (bool): If True, use recursive splitting (ensures total volume 
                           matches bin volume). If False, generate random sizes.

    Returns:
        list[list[int]]: List of box sizes as [w, d, h].
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if structured:
        # Recursive splitting approach
        # Each split divides one box into two smaller boxes along a randomly
        dim = len(bin_size)
        item_sizes = [bin_size]

        while len(item_sizes) < num_items:
            # Select a box to split, weighted by volume
            box_vols = [np.prod(box_size) for box_size in item_sizes]
            index = random.choices(list(range(len(item_sizes))), weights=box_vols, k=1)[0]
            box0_size = item_sizes.pop(index)

            # Choose a split axis and split point
            axis = random.choices(list(range(dim)), weights=box0_size, k=1)[0]
            len_edge = box0_size[axis]
            
            # Ensure the selected box can be split
            while len_edge == 1:
                axis = random.choices(list(range(dim)), weights=box0_size, k=1)[0]
                len_edge = box0_size[axis]

            # Determine split point with bias towards center
            if len_edge == 2:
                split_point = 1
            else:
                dist_edge_center = [abs(x - len_edge / 2) for x in range(1, len_edge)]
                weights = np.reciprocal(np.asarray(dist_edge_center) + 1)
                split_point = random.choices(list(range(1, len_edge)), weights=weights, k=1)[0]

            # Create two new boxes from the split
            box1 = list(deepcopy(box0_size))
            box2 = list(deepcopy(box0_size))
            box1[axis] = split_point
            box2[axis] = len_edge - split_point

            # Validate volumes
            assert np.prod(box1) + np.prod(box2) == np.prod(box0_size)
            
            item_sizes.extend([box1, box2])

        return item_sizes

    else:
        # Purely random sampling approach
        return [
            [random.randint(1, 5),
             random.randint(1, 5),
             random.randint(1, 5)]
            for _ in range(num_items)
        ]