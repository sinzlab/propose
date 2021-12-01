import torch
import torch.nn as nn

import torch_sparse as ts


class CondGCN(nn.Module):
    """
    Conditional GCN layer.
    """

    def __init__(self, in_features: int = 3, context_features: int = 2, out_features: int = 3,
                 hidden_features: int = 10):
        super().__init__()

        self.features = {
            'in': in_features,
            'context': context_features,
            'hidden': hidden_features,
            'out': out_features
        }

        self.layers = nn.ModuleDict({
            # Self loop edges
            'x->x': nn.Linear(in_features, hidden_features),
            'c->c': nn.Linear(context_features, hidden_features),
            # Between node edges
            'x<->x': nn.Linear(in_features, hidden_features),
            'c->x': nn.Linear(context_features, hidden_features)
        })

        self.pool = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU()

        self.aggr = 'add'

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, dict]:
        x, c = x_dict['x'], x_dict['c']

        self_x = self.act(self.layers['x->x'](x))

        message = self.aggregate(self.message(x_dict, edge_index_dict), self_x)

        x_dict['x'] = self.pool(message)

        if c is not None:
            x_dict['c'] = self.act(self.layers['c->c'](c))

        return x_dict, edge_index_dict

    def message(self, x_dict: dict, edge_index_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the message for each edge.
        :param x_dict: x_dict['x'] is the node features.
        :param edge_index_dict: edge_index_dict['x<->x'] is the edge indices.
        :return: message and destination index
        """
        for key in edge_index_dict.keys():
            src_name, _, dst_name = key
            layer_name = ''.join(key)
            src, dst = edge_index_dict[key]

            shape = list(x_dict[dst_name].shape)
            shape[-1] = self.features['hidden']

            # Message computation from source to destination according to the edge type with activation
            message = self.act(self.layers[layer_name](x_dict[src_name][src]))

            # Walrus operator to create a tensor with the same shape as the destination tensor
            # and fill it with the message at the destination indices
            (m := torch.zeros(shape))[dst] = message

            yield m[dst], dst

    def aggregate(self, message: tuple[torch.Tensor, torch.Tensor], self_x: torch.Tensor) -> torch.Tensor:
        """
        Aggregates the messages according to the aggregation method.
        :param message: message tensor
        :param self_x: self loop features
        :return: aggregated message
        """
        message = list(message)

        index = torch.cat([
            *[i for _, i in message],  # concatenate all message indices
            torch.arange(self_x.shape[0], dtype=torch.long)  # add self loop indices
        ])
        index = torch.stack([
            index,
            torch.zeros_like(index)  # Only one column index, thus all zeros
        ])

        # Concatenate all messages with self loop features
        value = torch.cat([
            *[v for v, _ in message],
            self_x
        ])

        # Aggregates the messages according to the aggregation method where the same index is used
        _, aggr_message = ts.coalesce(index, value, m=index[0].max() + 1, n=index[1].max() + 1, op=self.aggr)

        return aggr_message
