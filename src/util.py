import torch
import os
import numpy as np
import random


def reproducability(seed):
    """Set seeds for reproducibility across different libraries.
    Args:
        seed (int): The seed value to set.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
