import torch

from propose.models.flows.GraphFlow import GraphFlow
from propose.models.nn.CondGNN import CondGNN

from propose.models.transforms.transform import (
    GraphAffineCouplingTransform,
    GraphCompositeTransform,
    GraphActNorm,
)
from propose.models.distributions import StandardNormal


class CondGraphFlow(GraphFlow):
    def __init__(
            self, features=3, num_layers=5, context_features=2, hidden_features=100
    ):
        def create_net(in_features, out_features):
            return CondGNN(
                in_features=in_features,
                context_features=context_features,
                out_features=out_features,
                hidden_features=hidden_features,
            )

        coupling_constructor = GraphAffineCouplingTransform

        mask = torch.ones(features)
        mask[::2] = -1

        layers = []
        for _ in range(num_layers):
            layers.append(GraphActNorm(features=features))
            layers.append(
                coupling_constructor(mask=mask, transform_net_create_fn=create_net)
            )
            mask *= -1

        super().__init__(
            transform=GraphCompositeTransform(layers),
            distribution=StandardNormal([features]),
        )
