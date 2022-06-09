from nflows.distributions.base import Distribution
from nflows.utils import torchutils

import torch

import numpy as np


class StandardNormal(Distribution):
    """
    A multivariate Normal with zero mean and unit covariance.
    Adapted from nflows.distributions.StandardNormal such that it works with graph data
    """

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)

        self.register_buffer(
            "_log_z",
            torch.tensor(0.5 * np.prod(shape) * np.log(2 * np.pi), dtype=torch.float64),
            persistent=False,
        )

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.dim() == 3:
            inputs = inputs.mean(1)

        neg_energy = -0.5 * torchutils.sum_except_batch(inputs**2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape, device=self._log_z.device)
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(
                context_size * num_samples, *self._shape, device=context.device
            )
            return torchutils.split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return self._log_z.new_zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return context.new_zeros(context.shape[0], *self._shape)
