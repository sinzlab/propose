import torch
import torch.distributions as D
import torch.nn.functional as F

from tqdm import tqdm

from torch_geometric.data import HeteroData


def get_M_only_graph(data_dict):
    """
    Get the M only graph from a data dictionary
    :param data: data dictionary
    :return: M only graph
    """
    M_only_graph = {
        'x': {**data_dict['x']}
    }

    try:
        M_only_graph[('x', '->', 'x')] = {**data_dict[('x', '->', 'x')]}
    except KeyError:
        pass

    try:
        M_only_graph[('x', '<-', 'x')] = {**data_dict[('x', '<-', 'x')]}
    except KeyError:
        pass

    return HeteroData(M_only_graph)


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

            data_dict = data.to_dict()

            M_only_graph = {
                'x': {**data_dict['x']}
            }

            try:
                M_only_graph[('x', '->', 'x')] = {**data_dict[('x', '->', 'x')]}
            except KeyError:
                pass

            try:
                M_only_graph[('x', '<-', 'x')] = {**data_dict[('x', '<-', 'x')]}
            except KeyError:
                pass

            M_only_graph = HeteroData(M_only_graph)

            reg_prior_loss = -flow.log_prob(M_only_graph).mean()
            reg_posterior_loss = -flow.log_prob(data).mean()

            loss = reg_prior_loss + reg_posterior_loss
            loss.backward()

            nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            optimizer.step()

            pbar.set_description(
                f'Epoch: {epoch + 1}/{epochs} | RegPriorLoss {reg_prior_loss.item():.4f} | RegPosteriorLoss {reg_posterior_loss.item():.4f} | Batch')


def semi_supervised_trainer(dataloader, flow, optimizer=None, epochs=100):
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
            M, q_M_m = flow.sample_and_log_prob(1, context=data)

            M_dict = M.to_dict()
            data_dict = data.to_dict()

            src, dst = M_dict[('c', '->', 'x')]['edge_index']

            m_pred = M_dict['x']['x'][dst, ..., :2]
            m_true = data_dict['c']['x'].unsqueeze(-2)[src]

            posterior_loss = q_M_m.mean()
            prior_loss = -flow.log_prob(M).mean()
            rec_loss = F.mse_loss(m_pred, m_true)

            M_only_graph = get_M_only_graph(data_dict)

            reg_prior_loss = -flow.log_prob(M_only_graph).mean()
            reg_posterior_loss = -flow.log_prob(data).mean()

            loss = 100 * rec_loss + posterior_loss + prior_loss + reg_prior_loss + reg_posterior_loss
            loss.backward()

            optimizer.step()

            pbar.set_description(
                f'Epoch: {epoch + 1}/{epochs} | RegPriorLoss {reg_prior_loss.item():.4f} | RegPosteriorLoss {reg_posterior_loss.item():.4f} | RecLoss {rec_loss.item():.4f} | PriorLoss {prior_loss.item():.4f} | PosteriorLoss {posterior_loss.item():.4f} | Batch')
