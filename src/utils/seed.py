# ==============================================================================
# FILE: utils/seed.py
# DESCRIPTION: Global seeding utilities for the project.
#              Ensures consistent results across Python, NumPy, and PyTorch.
# ==============================================================================

import os
import random
import numpy as np
import torch

# ------------------------------------------------------------------------------
# SEEDING UTILITIES
# ------------------------------------------------------------------------------

def seed_all(seed: int, deterministic_torch: bool = True):
    """
    Set seeds for ALL random number generators used across Python, NumPy, and PyTorch
    to ensure reproducible results.

    Args:
        seed (int): The seed value to apply across all libraries.
        deterministic_torch (bool): If True, enforce deterministic behavior in 
                                    PyTorch (disables cuDNN benchmarks).

    Notes:
        - Setting deterministic_torch to True may impact performance.
        - Essential for debugging and comparing different agent architectures.
    """
    # Standard Python and Environment seeding
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU and GPU seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for cuDNN
    if deterministic_torch:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True