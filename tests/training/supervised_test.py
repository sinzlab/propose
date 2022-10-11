from unittest.mock import MagicMock, call

import torch
from torch_geometric.loader import DataLoader

from propose.datasets.toy import SinglePointDataset, ThreePointDataset
from propose.models.flows import CondGraphFlow
from propose.training.supervised import supervised_trainer


def test_smoke_single():
    flow = CondGraphFlow()

    dataset = SinglePointDataset(samples=2)
    dataloader = DataLoader(dataset, batch_size=1)

    supervised_trainer(dataloader, flow, epochs=1, use_mode=False)


def test_smoke_three():
    flow = CondGraphFlow()

    dataset = ThreePointDataset(samples=2)
    dataloader = DataLoader(dataset, batch_size=1)

    supervised_trainer(dataloader, flow, epochs=1, use_mode=False)


def test_updates_weights():
    flow = CondGraphFlow()

    num_samples = 2
    batch_size = 2
    dataset = SinglePointDataset(samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    assert all([param.grad is None for param in flow.parameters()])

    supervised_trainer(dataloader, flow, epochs=1, use_mode=False)

    assert any([param.grad is not None for param in flow.parameters()])

    optimizer = MagicMock(spec=torch.optim.Optimizer)
    optimizer.param_groups = [{"lr": 0.1}]

    supervised_trainer(dataloader, flow, epochs=1, optimizer=optimizer, use_mode=False)

    assert optimizer.mock_calls == [call.zero_grad(), call.step()] * (
        num_samples // batch_size
    )
