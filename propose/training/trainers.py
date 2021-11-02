import torch
import torch.distributions as D

from tqdm import tqdm


def negative_log_likelihood(z, logabsdet, prior=None):
    """
    Negative log likelihood loss evaluated on a prior distribution for normalizing flows
    :param prior: prior distribution for the loss
    :param z: embedded value
    :param logabsdet: log abs determinant of the flow
    :return:
    """
    if prior is None:
        prior = D.MultivariateNormal(torch.zeros(3), torch.eye(3))

    return (-prior.log_prob(z) - logabsdet).mean()


def prior_trainer(flow, dataloader, epochs=1, lr=0.001, weight_decay=1e-5):
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f'Epoch: {epoch + 1}/{epochs} | NLLoss: 0 | Batch')
        for _, M in pbar:
            z, logabsdet = flow._transform(M)

            loss = negative_log_likelihood(z, logabsdet)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(f'Epoch: {epoch + 1}/{epochs} | NLLoss: {loss.item():.4g} | Batch')


def flow_trainer(flow, prior_flow, labeled_dataloader, unlabeled_dataloader, epochs=1, lr=0.001, weight_decay=1e-5):
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        labeled_dataloader_iter = iter(labeled_dataloader)

        pbar = tqdm(unlabeled_dataloader, desc=f'Epoch: {epoch + 1}/{epochs} | NLLoss: 0 | RecLoss: 0 | Batch')
        for m_prime in pbar:
            try:
                m_bar, M_bar = next(labeled_dataloader_iter)
            except StopIteration:
                labeled_dataloader_iter = iter(labeled_dataloader)
                m_bar, M_bar = next(labeled_dataloader_iter)

            m = torch.cat([m_prime, m_bar])

            M, q_M_m = flow.sample_and_log_prob(1000, context=m)

            M_flat = M.reshape((M.shape[0] * M.shape[1], M.shape[2]))

            rec_loss = ((M[..., :2] - torch.unsqueeze(m, 1)) ** 2).mean()  # TODO: change this to use camera
            posterior_loss = q_M_m.mean()
            prior_loss = -prior_flow.log_prob(M_flat).mean()
            regularization_loss = -flow.log_prob(M_bar, context=m_bar).mean()

            loss = 1000 * rec_loss + posterior_loss + prior_loss + regularization_loss

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            pbar.set_description(
                f'Epoch: {epoch + 1}/{epochs} | NLLoss: {prior_loss.item():.4g} | RecLoss: {rec_loss.item():.4g} | Batch')
