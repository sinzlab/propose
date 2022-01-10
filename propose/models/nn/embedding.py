import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional, Union


class Embedding(nn.Module):
    """
    Base class for the embedding layer.
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Identity()

    def forward(
        self, x: Union[tuple[Tensor, Tensor], Tensor]
    ) -> Union[tuple[Tensor, Tensor], Tensor]:
        if isinstance(x, tuple):
            return self.embedding(x[0]), x[1]
        else:
            return self.embedding(x)


class LinearEmbedding(Embedding):
    """
    Linear embedding layer.
    """

    def __init__(self, input_size: int, output_size: int):
        """
        :param input_size
        :param output_size
        """
        super().__init__()
        self.embedding = nn.Linear(input_size, output_size)


class SplitEmbedding(nn.Module):
    """
    Split embedding layer.
    """

    def __init__(self, split_size: int):
        """
        :param split_size: size of the first split
        """
        super().__init__()
        self.split_size = split_size

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return x[:, : self.split_size], x[:, self.split_size :]


class JoinEmbedding(nn.Module):
    """
    Join embedding layer.
    """

    def forward(self, x: tuple[Tensor]) -> Tensor:
        return torch.cat(x, dim=1)


class SplitLinearEmbedding(Embedding):
    """
    Split linear embedding layer.
    """

    def __init__(self, split_size, input_size, output_size):
        super().__init__()

        self.embedding = nn.Sequential(
            SplitEmbedding(split_size),
            LinearEmbedding(input_size, output_size),
            JoinEmbedding(),
        )
