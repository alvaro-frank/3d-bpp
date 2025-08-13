import json
import numpy as np
from pathlib import Path

def make_test_sets(seed: int, n_episodes: int, n_boxes: int, box_ranges: dict):
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n_episodes):
        ep = []
        for _ in range(n_boxes):
            w = int(rng.integers(box_ranges["w_min"], box_ranges["w_max"] + 1))
            h = int(rng.integers(box_ranges["h_min"], box_ranges["h_max"] + 1))
            d = int(rng.integers(box_ranges["d_min"], box_ranges["d_max"] + 1))
            ep.append({"w": w, "h": h, "d": d})
        sets.append(ep)
    return sets

def save_test_sets(path: str, sets):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sets))

def load_test_sets(path: str):
    return json.loads(Path(path).read_text())