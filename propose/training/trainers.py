import torch
import torch.distributions as D

from tqdm import tqdm

from torch_geometric.data import HeteroData


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
            optimizer.zero_grad()

            data_dict = data.to_dict()

            M_only_graph = {
                'x': {**data_dict['x']}
            }

            try:
                M_only_graph[('x', '<->', 'x')] = {**data_dict[('x', '<->', 'x')]}
            except KeyError:
                pass

            if 'c' in M_only_graph:
                del M_only_graph['c']
                del M_only_graph['c', '->', 'x']

            M_only_graph = HeteroData(M_only_graph)

            reg_prior_loss = -flow.log_prob(M_only_graph).mean()
            reg_posterior_loss = -flow.log_prob(data).mean()

            loss = reg_prior_loss + reg_posterior_loss
            loss.backward()

            optimizer.step()

            pbar.set_description(
                f'Epoch: {epoch + 1}/{epochs} | RegPriorLoss {reg_prior_loss.item():.4f} | RegPosteriorLoss {reg_posterior_loss.item():.4f} | Batch')
