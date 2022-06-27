import torch
import torch.nn as nn

import torch_sparse as ts

from typing import Literal


class CondGCN(nn.Module):
    """
    Conditional GCN layer.
    """

    def __init__(
        self,
        in_features: int = 3,
        context_features: int = 2,
        out_features: int = 3,
        hidden_features: int = 10,
        root_features: int = 3,
        aggr: Literal["add", "mean", "max"] = "add",
        relations: list[str] = None,
    ) -> None:
        super().__init__()

        default_relations: list[str] = [
            "x",
            "c",
            "r",  # self loop
            "x->x",
            "x<-x",  # symmetric
            "c->x",
            "r->x",
        ]  # context

        self.relations = relations if relations else default_relations

        self.features = {
            "x": in_features,
            "c": context_features,
            "r": root_features,
            "hidden": hidden_features,
            "out": out_features,
        }

        self.layers = self._build_layers()

        self.pool = nn.Linear(hidden_features, out_features)
        self.act = nn.ReLU()

        self.aggr = aggr

    def forward(self, x_dict: dict, edge_index_dict: dict) -> tuple[dict, dict]:
        x = x_dict["x"]

        self_x = self.act(self.layers["x"](x))  # self loop values

        message = self.aggregate(self.message(x_dict, edge_index_dict), self_x)

        if "c" in x_dict and x_dict["c"] is not None:
            x_dict["c"] = self.act(self.layers["c"](x_dict["c"]))

        if "r" in x_dict and x_dict["r"] is not None:
            x_dict["r"] = self.act(self.layers["r"](x_dict["r"]))

        x_dict["x"] = self.pool(message)

        return x_dict, edge_index_dict

    def message(
        self, x_dict: dict, edge_index_dict: dict, target: Literal["x", "c", "r"] = "x"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the message for each edge.
        :param x_dict: x_dict['x'] is the node features.
        :param edge_index_dict: edge_index_dict['x<->x'] is the edge indices.
        :return: message and destination index
        """
        for key in edge_index_dict.keys():
            src_name, direction, dst_name = key  # e.g. "c", "->", "x"
            layer_name = "".join(key)  # e.g. "c->x"
            src, dst = edge_index_dict[key]

            # If the edge is an inverse edge, swap the source and destination
            if direction == "<-":
                src_name, dst_name = dst_name, src_name
                src, dst = dst, src
                layer_name = "x->x"  # .join(key[::-1])

            if dst_name != target:
                continue

            message = self.act(self.layers[layer_name](x_dict[src_name][src]))

            yield message, dst

    def aggregate(
        self, message: tuple[torch.Tensor, torch.Tensor], self_x: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregates the messages according to the aggregation method.
        :param message: message tensor
        :param self_x: self loop features
        :return: aggregated message
        """
        message = list(message)

        values, indexes = [], []
        if len(message):
            values, indexes = list(zip(*message))

        index = torch.cat(
            [
                *indexes,  # concatenate all message indices
                torch.arange(
                    self_x.shape[0],
                    dtype=torch.long,
                    device=self.device,  # if torch.cuda.is_available() else 'cpu'
                ),  # add self loop indices
            ]
        )
        index = torch.stack(
            [index, torch.zeros_like(index)]  # Only one column index, thus all zeros
        )

        # Concatenate all messages with self loop features

        # when sampling the messages from context need to be repeated to match the number of samples

        if len(values) == 4 and values[0].dim() == 3:
            values = list(values)
            samples = values[0].shape[1]
            if values[2].shape[1] == 1:
                values[2] = values[2].repeat(1, samples, 1)
            if values[3].shape[1] == 1:
                values[3] = values[3].repeat(1, samples, 1)

        if len(values) == 3 and values[0].dim() == 3:
            values = list(values)
            samples = self_x.shape[1]
            if values[0].shape[1] == 1:
                values[0] = values[0].repeat(1, samples, 1)
            if values[2].shape[1] == 1:
                values[2] = values[2].repeat(1, samples, 1)

        if len(values) == 1 and values[0].dim() == 3:
            values = list(values)
            # samples = values[0].shape[1]
            # self_x = self_x.repeat(1, samples, 1)
            samples = self_x.shape[1]
            values[0] = values[0].repeat(1, samples, 1)

        value = torch.cat([*values, self_x])  # concatenate all the message values

        # Aggregates the messages according to the aggregation method where the same index is used
        _, aggr_message = ts.coalesce(
            index, value, m=index[0].max() + 1, n=index[1].max() + 1, op=self.aggr
        )

        return aggr_message

    @property
    def device(self):
        return next(self.parameters()).device

    def _build_layers(self):
        layers_dict = {}
        for relation in self.relations:
            n_features: int = self.features[relation[0]]
            layers_dict[relation] = nn.Linear(n_features, self.features["hidden"])

        return nn.ModuleDict(layers_dict)
