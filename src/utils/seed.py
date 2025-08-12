import os, random
import numpy as np

try:
    import torch
except Exception:
    torch = None

def set_global_seed(seed: int, cuda_deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            try:
                import torch.backends.cudnn as cudnn
                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass
