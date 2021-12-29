import torch
import torch.nn as nn

from tqdm import tqdm

from .utils import get_x_graph


def supervised_trainer(dataloader, flow, optimizer=None, epochs=100):
    """
    Train a flow in a supervised way
    :param dataloader: dataloader for the supervised training
    :param flow: flow to be trained
    :param optimizer: optimizer to be used
    :param epochs: number of epochs
    :return: None
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(flow.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch: {epoch + 1}/{epochs} | NLLoss: 0 | RecLoss: 0 | Batch')
        for data in pbar:
            data.to(flow.device)

            optimizer.zero_grad()

            x_graph = get_x_graph(data)
            x_graph.to(flow.device)

            reg_prior_loss = -flow.log_prob(x_graph).mean()
            reg_posterior_loss = -flow.log_prob(data).mean()

            loss = reg_prior_loss + reg_posterior_loss
            loss.backward()

            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            optimizer.step()

            pbar.set_description(
                f'Epoch: {epoch + 1}/{epochs} | Loss {loss.item():.4f} | Batch')
