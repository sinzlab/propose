from unittest.mock import MagicMock, call

from propose.datasets.toy import SinglePointDataset, ThreePointDataset
from propose.models.flows import CondGraphFlow
from propose.training.semi_supervised import semi_supervised_trainer
from torch_geometric.loader import DataLoader


def test_smoke_single():
    flow = CondGraphFlow()

    dataset = SinglePointDataset(samples=2)
    dataloader = DataLoader(dataset, batch_size=1)

    semi_supervised_trainer(dataloader, flow, epochs=1)


def test_smoke_three():
    flow = CondGraphFlow()

    dataset = ThreePointDataset(samples=2)
    dataloader = DataLoader(dataset, batch_size=1)

    semi_supervised_trainer(dataloader, flow, epochs=1)


def test_updates_weights():
    flow = CondGraphFlow()

    num_samples = 2
    dataset = SinglePointDataset(samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=1)

    assert all([param.grad is None for param in flow.parameters()])

    semi_supervised_trainer(dataloader, flow, epochs=1)

    assert any([param.grad is not None for param in flow.parameters()])

    optimizer = MagicMock()
    semi_supervised_trainer(dataloader, flow, epochs=1, optimizer=optimizer)

    assert optimizer.mock_calls == [call.zero_grad(), call.step()] * num_samples
