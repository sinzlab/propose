import torch

from unittest.mock import MagicMock, patch

from propose.training.trainers import prior_trainer, flow_trainer

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

import torch.distributions as D


class LabeledDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.prior = D.MultivariateNormal(torch.Tensor(torch.zeros(3)), covariance_matrix=torch.eye(3))

        self.samples = self.prior.sample((10,))

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, item):
        return self.samples[item, :2], self.samples[item]


class UnlabeledDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.prior = D.MultivariateNormal(torch.Tensor(torch.zeros(3)), covariance_matrix=torch.eye(3))

        self.samples = self.prior.sample((10,))

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, item):
        return self.samples[item, :2]


@patch('propose.training.trainers.torch.optim.Adam')
def test_prior_smoke(adam_mock):
    model = MagicMock()
    model._transform = MagicMock(return_value=(
        torch.zeros((1, 3), requires_grad=True),
        torch.zeros((1, 3), requires_grad=True)
    ))

    dataloder = DataLoader(LabeledDataset(), batch_size=10)

    prior_trainer(model, dataloder)


@patch('propose.training.trainers.torch.optim.Adam')
def test_prior_transform(adam_mock):
    model = MagicMock()
    model._transform = MagicMock(return_value=(
        torch.zeros((1, 3), requires_grad=True),
        torch.zeros((1, 3), requires_grad=True)
    ))

    dataloder = DataLoader(LabeledDataset(), batch_size=10)

    prior_trainer(model, dataloder)

    assert model._transform.called


@patch('propose.training.trainers.torch.optim.Adam')
def test_flow_trainer_smoke(adam_mock):
    flow = MagicMock()
    flow.sample_and_log_prob = MagicMock(return_value=(
        torch.zeros((1, 1, 3), requires_grad=True),
        torch.zeros((1, 1, 3), requires_grad=True)
    ))
    flow.log_prob = MagicMock(return_value=torch.zeros((1, 3), requires_grad=True))

    prior_flow = MagicMock()
    prior_flow._transform = MagicMock(return_value=(
        torch.zeros((1, 3), requires_grad=True),
        torch.zeros((1, 3), requires_grad=True)
    ))
    prior_flow.log_prob = MagicMock(return_value=torch.zeros((1, 3), requires_grad=True))

    labeled_dataloder = DataLoader(LabeledDataset(), batch_size=10)
    unlabeled_dataloder = DataLoader(UnlabeledDataset(), batch_size=10)

    flow_trainer(flow, prior_flow, labeled_dataloder, unlabeled_dataloder)


@patch('propose.training.trainers.torch.optim.Adam')
def test_flow_labeled_dataloader_shorter(adam_mock):
    flow = MagicMock()
    flow.sample_and_log_prob = MagicMock(return_value=(
        torch.zeros((1, 1, 3), requires_grad=True),
        torch.zeros((1, 1, 3), requires_grad=True)
    ))
    flow.log_prob = MagicMock(return_value=torch.zeros((1, 3), requires_grad=True))

    prior_flow = MagicMock()
    prior_flow._transform = MagicMock(return_value=(
        torch.zeros((1, 3), requires_grad=True),
        torch.zeros((1, 3), requires_grad=True)
    ))
    prior_flow.log_prob = MagicMock(return_value=torch.zeros((1, 3), requires_grad=True))

    labeled_dataloder = DataLoader(LabeledDataset(), batch_size=10)
    unlabeled_dataloder = DataLoader(UnlabeledDataset(), batch_size=1)

    flow_trainer(flow, prior_flow, labeled_dataloder, unlabeled_dataloder)
