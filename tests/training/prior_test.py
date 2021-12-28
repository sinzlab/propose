from propose.models.flows import CondGraphFlow
from propose.training.prior import prior_trainer
from propose.datasets.toy import SinglePointDataset, ThreePointDataset

from torch_geometric.loader import DataLoader
from unittest.mock import MagicMock, call


def test_smoke_single():
    flow = CondGraphFlow()

    dataset = SinglePointDataset(samples=2)
    dataloader = DataLoader(dataset, batch_size=1)

    prior_trainer(dataloader, flow, epochs=1)


def test_smoke_three():
    flow = CondGraphFlow()

    dataset = ThreePointDataset(samples=2)
    dataloader = DataLoader(dataset, batch_size=1)

    prior_trainer(dataloader, flow, epochs=1)


def test_updates_weights():
    flow = CondGraphFlow()

    num_samples = 2
    dataset = SinglePointDataset(samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=1)

    assert all([param.grad is None for param in flow.parameters()])

    prior_trainer(dataloader, flow, epochs=1)

    assert any([param.grad is not None for param in flow.parameters()])

    optimizer = MagicMock()
    prior_trainer(dataloader, flow, epochs=1, optimizer=optimizer)

    assert optimizer.mock_calls == [call.zero_grad(), call.step()] * num_samples
