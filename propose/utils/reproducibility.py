import numpy as np
import torch
import random


def set_random_seed(seed):
    """
    Sets all random seeds
    """

    # Base seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Cuda seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Make GPU operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
