import os, random, numpy as np
import torch

def seed_all(seed: int, deterministic_torch: bool = True):
    """
    Set seeds for ALL random number generators used across Python, NumPy, and PyTorch
    to ensure reproducible results.

    Parameters:
    - seed (int): the seed value to apply
    - deterministic_torch (bool): if True, enforce deterministic behavior in PyTorch
      (may reduce speed but guarantees reproducibility)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True