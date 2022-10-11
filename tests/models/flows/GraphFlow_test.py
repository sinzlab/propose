from unittest import TestCase
from unittest.mock import MagicMock

import torch
import torch.testing as tt
from propose.datasets.toy.Point import SinglePointDataset, SinglePointPriorDataset
from propose.models.distributions import StandardNormal
from propose.models.flows.GraphFlow import GraphFlow
from propose.models.nn.embedding import embeddings
from propose.models.transforms.transform import GraphCompositeTransform
from torch_geometric.loader import DataLoader


class TestGraphFlow(TestCase):
    def test_smoke(self):
        distribution = StandardNormal((1,))
        transform = GraphCompositeTransform([])
        GraphFlow(transform, distribution)

    def test_default_identity_embedding_net(self):
        """
        Test that the default identity embedding net is used if no embedding net is provided.
        """
        distribution = StandardNormal((1,))
        transform = GraphCompositeTransform([])
        gf = GraphFlow(transform, distribution)

        assert isinstance(gf._embedding_net, torch.nn.Identity)

    def test_set_embedding_net(self):
        """
        Test that the embedding net can be set.
        """
        distribution = StandardNormal((1,))
        transform = GraphCompositeTransform([])
        embedding_net = torch.nn.Linear(1, 1)
        gf = GraphFlow(transform, distribution, embedding_net=embedding_net)

        assert gf._embedding_net == embedding_net

    def test_log_prob(self):
        """
        Test that the log probability is computed correctly.
        """
        distribution = StandardNormal((1,))
        transform = GraphCompositeTransform([])
        gf = GraphFlow(transform, distribution)

        dataset = SinglePointDataset()
        dataloader = DataLoader(dataset, batch_size=10)
        data = next(iter(dataloader))[1]

        log_prob = gf.log_prob(data)

        assert log_prob.size() == torch.Size([10])

    def test_embed_inputs(self):
        """
        Test that the embedding net is used to embed the inputs.
        """
        distribution = StandardNormal((1,))
        transform = GraphCompositeTransform([])

        embedding_net = embeddings["flat_mlp"](input_dim=2, hidden_dim=2, output_dim=2)
        gf = GraphFlow(transform, distribution, embedding_net=embedding_net)

        dataset = SinglePointDataset()
        dataloader = DataLoader(dataset, batch_size=10)
        data = next(iter(dataloader))[1]

        c = data["c"]["x"]
        x = data["x"]["x"]
        data_embed = gf.embed_inputs(data)
        c_embed = data_embed["c"]["x"]
        x_embed = data_embed["x"]["x"]

        # Check c is different from c_embed
        assert not torch.eq(c, c_embed).all()

        # check x is the same
        assert torch.eq(x, x_embed).all()

    def test_embed_inputs_without_context(self):
        """
        Test that the embedding net is used to embed the inputs.
        """
        distribution = StandardNormal((1,))
        transform = GraphCompositeTransform([])
        embedding_net = torch.nn.Linear(2, 2)
        gf = GraphFlow(transform, distribution, embedding_net=embedding_net)

        dataset = SinglePointPriorDataset()
        dataloader = DataLoader(dataset, batch_size=10)
        data = next(iter(dataloader))[1]

        x = data["x"]["x"]
        data_embed = gf.embed_inputs(data)
        x_embed = data_embed["x"]["x"]

        # check x is the same
        assert torch.eq(x, x_embed).all()

    def test_sampling(self):
        """
        Test that the sampling is done correctly.
        """
        distribution = StandardNormal((3,))
        transform = GraphCompositeTransform([])
        gf = GraphFlow(transform, distribution)

        dataset = SinglePointDataset()
        batch_size = 10
        dataloader = DataLoader(dataset, batch_size=batch_size)
        data = next(iter(dataloader))[1]

        n_samples = 2
        samples = gf.sample(n_samples, data)

        assert samples["x"]["x"].size() == torch.Size([batch_size, n_samples, 3])

    def test_transform_to_noise(self):
        """
        Test that the transform_to_noise is done correctly.
        """
        distribution = StandardNormal((3,))
        transform = GraphCompositeTransform([])
        gf = GraphFlow(transform, distribution)

        dataset = SinglePointPriorDataset()
        batch_size = 10
        dataloader = DataLoader(dataset, batch_size=batch_size)
        data = next(iter(dataloader))[1]

        trans = gf.transform_to_noise(data)
        hand_trans, _ = gf._transform(data)

        tt.assert_equal(trans["x"]["x"], hand_trans["x"]["x"])
