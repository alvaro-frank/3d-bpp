# ==============================================================================
# FILE: utils/testsets.py
# DESCRIPTION: Utilities for creating, saving, and loading deterministic test sets.
#              Ensures that agents are evaluated on consistent, perfect-fit benchmarks.
# ==============================================================================

import json
from pathlib import Path
import numpy as np
from utils.box_generator import generate_boxes

# ------------------------------------------------------------------------------
# TEST SET GENERATION
# ------------------------------------------------------------------------------

def make_test_sets(seed: int, n_episodes: int, n_boxes: int, bin_size=(10, 10, 10)):
    """
    Create deterministic test sets using Recursive Splitting (Structured).
    
    This method ensures that the sum of the volumes of the generated boxes 
    perfectly matches the bin volume, providing a "perfect fit" baseline.

    Args:
        seed (int): Master seed to derive individual episode seeds.
        n_episodes (int): Number of distinct test scenarios to create.
        n_boxes (int): Number of boxes per episode.
        bin_size (tuple): Tuple (w, d, h) of the target bin.

    Returns:
        list[list[dict]]: A list of episodes, each containing a list of box dimensions.
    """
    # Note: box_ranges is ignored here because recursive splitting defines sizes
    # based on the container dimensions to guarantee a 100% fill solution.
    
    sets = []
    
    # Generate deterministic seeds for each episode from the master seed
    master_rng = np.random.default_rng(seed)
    episode_seeds = master_rng.integers(0, 1000000, size=n_episodes)

    for i in range(n_episodes):
        # Use generate_boxes with structured=True
        # This guarantees that the generated boxes SUM to the volume of bin_size
        raw_boxes = generate_boxes(
            bin_size=bin_size, 
            num_items=n_boxes, 
            seed=int(episode_seeds[i]), 
            structured=True
        )
        
        # Convert from list [w, d, h] to dictionary {"w":..., "d":..., "h":...}
        ep = [{"w": b[0], "d": b[1], "h": b[2]} for b in raw_boxes]
        sets.append(ep)
        
    return sets

# ------------------------------------------------------------------------------
# I/O OPERATIONS
# ------------------------------------------------------------------------------

def save_test_sets(path: str, sets):
    """
    Save generated test sets to disk in JSON format.

    Args:
        path (str): Filesystem path (e.g., 'data/test_sets_30.json').
        sets (list): The test sets to serialize.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sets))


def load_test_sets(path: str):
    """
    Load test sets from a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        list: Deserialized test sets for evaluation.
    """
    return json.loads(Path(path).read_text())