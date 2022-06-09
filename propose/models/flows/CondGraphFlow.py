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
        gcn_type="slow",
        # mask_idx=[0, 2, 5, 8, 10, 12, 15]
    ):
        """
        Conditional Graph Flow model. The model is composed of a CondGNN and a GraphFlow.
        :param features: Number of features in the input.
        :param num_layers: Number of flow layers.
        :param context_features: Number of features in the context after embedding.
        :param hidden_features: Number of features in the hidden layers.
        :param embedding_net: (optional) Network to embed the context. default: nn.Identity
        :param gcn_type: (optional) Type of GCN to use. default: slow
        """

        def create_net(in_features, out_features):
            return CondGNN(
                in_features=in_features,
                context_features=context_features,
                out_features=out_features,
                hidden_features=hidden_features,
                gcn_type=gcn_type,
            )

        coupling_constructor = GraphAffineCouplingTransform

        layers = []
        for i in range(num_layers):
            mask = -torch.ones(features)
            mask[i % features] = 1  # iterate over feature pairs in a leave-one-out fashion

            layers.append(GraphActNorm(features=features))
            layers.append(
                coupling_constructor(mask=mask, transform_net_create_fn=create_net)
            )

        super().__init__(
            transform=GraphCompositeTransform(layers),
            distribution=StandardNormal([features]),
            embedding_net=embedding_net,
        )

    def forward(self, inputs):
        return self.log_prob(inputs)
