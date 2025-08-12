
import random
import numpy as np
import torch

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def make_seed_sequence(base_seed: int, n: int):
    # simple deterministic sequence
    rng = np.random.RandomState(base_seed)
    return [int(x) for x in rng.randint(0, 1_000_000_000, size=n)]
