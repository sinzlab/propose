"""Implementation of Conditional Glow."""

import torch

from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AdditiveCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.normalization import ActNorm

from propose.models.nn import MLP


class ConditionalGlow(Flow):
    """ A version of Conditional Glow for 1-dim inputs.
    Reference:
    > TODO
    """

    def __init__(
            self,
            features=3,
            num_layers=5,
            context_features=2,
            hidden_sizes=None
    ):
        if hidden_sizes is None:
            hidden_sizes = [100, 100, 100]

        coupling_constructor = AdditiveCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        def create_mlp(in_features, out_features):
            return MLP(
                (in_features + context_features,),
                (out_features,),
                hidden_sizes=hidden_sizes,
            )

        layers = []
        for _ in range(num_layers):
            layers.append(ActNorm(features=features))
            layers.append(LULinear(features=features))
            transform = coupling_constructor(
                mask=mask, transform_net_create_fn=create_mlp
            )
            mask *= -1
            layers.append(transform)

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=StandardNormal([features])
        )
