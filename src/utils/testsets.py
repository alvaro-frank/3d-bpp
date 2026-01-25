import json
from pathlib import Path
import numpy as np
from utils.box_generator import generate_boxes

def make_test_sets(seed: int, n_episodes: int, n_boxes: int, bin_size=(10, 10, 10)):
    """
    Create deterministic test sets using Recursive Splitting (Structured).
    
    Args:
    - bin_size: Tuple (w, d, h) of the bin we want to fill perfectly.
    """
    # Nota: box_ranges é ignorado aqui porque o recursive splitting define os tamanhos
    # com base no tamanho do contentor.
    
    sets = []
    
    # Geramos seeds determinísticas para cada episódio a partir da seed mestra
    master_rng = np.random.default_rng(seed)
    episode_seeds = master_rng.integers(0, 1000000, size=n_episodes)

    for i in range(n_episodes):
        # Usamos o generate_boxes com structured=True
        # Isto garante que as caixas geradas SOMAM o volume do bin_size
        raw_boxes = generate_boxes(
            bin_size=bin_size, 
            num_items=n_boxes, 
            seed=int(episode_seeds[i]), 
            structured=True
        )
        
        # Converter de lista [w,d,h] para dicionário {"w":..., "d":..., "h":...}
        ep = [{"w": b[0], "d": b[1], "h": b[2]} for b in raw_boxes]
        sets.append(ep)
        
    return sets


def save_test_sets(path: str, sets):
    """
    Save generated test sets to disk in JSON format.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sets))


def load_test_sets(path: str):
    """
    Load test sets from JSON file.
    """
    return json.loads(Path(path).read_text())