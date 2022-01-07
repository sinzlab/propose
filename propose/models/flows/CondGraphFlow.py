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
        self,
        features=3,
        num_layers=5,
        context_features=2,
        hidden_features=100,
        embedding_net=None,
    ):
        """
        Conditional Graph Flow model. The model is composed of a CondGNN and a GraphFlow.
        :param features: Number of features in the input.
        :param num_layers: Number of flow layers.
        :param context_features: Number of features in the context after embedding.
        :param hidden_features: Number of features in the hidden layers.
        :param embedding_net: (optional) Network to embed the context. default: nn.Identity
        """

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
            embedding_net=embedding_net,
        )
