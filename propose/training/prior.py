import torch
import torch.nn as nn

from tqdm import tqdm

from .utils import get_x_graph

# Typing Imports
from typing import Optional, Union
from torch_geometric.loader import DataLoader as TorchGeometricDataLoader
from torch.utils.data import DataLoader as TorchDataLoader
from nflows.flows.base import Flow
from torch.optim.optimizer import Optimizer


def prior_trainer(dataloader: Union[TorchGeometricDataLoader, TorchDataLoader],
                  flow: Flow, optimizer: Optional[Optimizer] = None, epochs: int = 100) -> None:
    """
    Train only the prior part of the model in a supervised fashion.
    :param dataloader: dataloader for the supervised training
    :param flow: flow to be trained
    :param optimizer: optimizer to be used
    :param epochs: number of epochs
    :return: None
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch: {epoch + 1}/{epochs} | Loss: {0} | Batch')
        for data in pbar:
            optimizer.zero_grad()

            x_graph = get_x_graph(data)
            x_graph.to(flow.device)

            loss = -flow.log_prob(x_graph).mean()
            loss.backward()

            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            optimizer.step()

            pbar.set_description(f'Epoch: {epoch + 1}/{epochs} | Loss {loss.item():.4f} | Batch')
