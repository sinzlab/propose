import torch
import numpy as np


def mpjpe(pred, gt, dim=None):
    pjpe = ((pred - gt) ** 2).sum(-1) ** 0.5

    # if pjpe is torch.Tensor use dim if numpy.array use axis
    if isinstance(pjpe, torch.Tensor):
        return pjpe.mean(dim=dim)

    return np.mean(pjpe, axis=dim)
