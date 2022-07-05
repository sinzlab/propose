import torch
import numpy as np


def mpjpe(pred, gt, dim=None):
    pjpe = ((pred - gt) ** 2).sum(-1) ** 0.5

    # if pjpe is torch.Tensor use dim if numpy.array use axis
    if isinstance(pjpe, torch.Tensor):
        if dim is None:
            return pjpe.mean()
        return pjpe.mean(dim=dim)

    if dim is None:
        return np.mean(pjpe)
    return np.mean(pjpe, axis=dim)
