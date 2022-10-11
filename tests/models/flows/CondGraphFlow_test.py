from unittest import TestCase

import torch

from propose.models.flows import CondGraphFlow


class TestCondGraphFlow(TestCase):
    def test_smoke(self):
        CondGraphFlow()

    def test_features(self):
        features = 5
        cgf = CondGraphFlow(features=features, num_layers=1)

        assert cgf._distribution._shape == torch.Size([features])

    def test_num_layers(self):
        num_layers = 10
        cgf = CondGraphFlow(num_layers=num_layers)

        assert len(cgf._transform._transforms) == num_layers * 2

    def test_embedding_net_default(self):
        cgf = CondGraphFlow()

        assert isinstance(cgf._embedding_net, torch.nn.Identity)

    def test_embedding_net(self):
        net = torch.nn.Linear(2, 2)
        cgf = CondGraphFlow(embedding_net=net)

        assert cgf._embedding_net is net
