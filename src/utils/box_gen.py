from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import random

@dataclass
class BoxSpec:
    width: int
    height: int
    depth: int
    id: Optional[int] = None

def sample_box(min_size=1, max_size=5) -> BoxSpec:
    w = random.randint(min_size, max_size)
    h = random.randint(min_size, max_size)
    d = random.randint(min_size, max_size)
    return BoxSpec(w, h, d)

def generate_dataset(n_episodes: int, boxes_per_episode: int, *, seed: int, min_size=1, max_size=5):
    random.seed(seed); np.random.seed(seed)
    ds: List[List[BoxSpec]] = []
    for ep in range(n_episodes):
        boxes = [sample_box(min_size=min_size, max_size=max_size) for _ in range(boxes_per_episode)]
        for i, b in enumerate(boxes):
            b.id = i
        ds.append(boxes)
    return ds
