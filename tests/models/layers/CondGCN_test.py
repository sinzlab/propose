from propose.models.layers.CondGCN import CondGCN, FastCondGCN

import torch
import torch.testing as tt

import types

from unittest import TestCase


class CondGCNTests(TestCase):
    @staticmethod
    def test_smoke():
        CondGCN()

    @staticmethod
    def test_layers_constructed():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = CondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        assert len(model.layers) == 5

        assert "x" in model.layers
        assert "c" in model.layers
        assert "x->x" in model.layers
        assert "c->x" in model.layers
        assert "x<-x" in model.layers

        assert model.layers["x"].in_features == in_features
        assert model.layers["x"].out_features == hidden_features

        assert model.layers["c"].in_features == context_features
        assert model.layers["c"].out_features == hidden_features

        assert model.layers["x->x"].in_features == in_features
        assert model.layers["x->x"].out_features == hidden_features

        assert model.layers["x<-x"].in_features == in_features
        assert model.layers["x<-x"].out_features == hidden_features

        assert model.layers["c->x"].in_features == context_features
        assert model.layers["c->x"].out_features == hidden_features

        assert model.pool.in_features == hidden_features
        assert model.pool.out_features == out_features

    @staticmethod
    def test_forward():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = CondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        x = torch.rand(1, in_features)
        c = torch.rand(1, context_features)

        x_dict = dict(x=x, c=c)

        edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0]]).T}

        y_dict, res_edge_index_dict = model(x_dict, edge_index_dict)

        assert "x" in y_dict
        assert "c" in y_dict

        assert y_dict["x"].shape == (1, out_features)
        assert y_dict["c"].shape == (1, hidden_features)

        assert ("c", "->", "x") in res_edge_index_dict
        tt.assert_equal(
            res_edge_index_dict[("c", "->", "x")], edge_index_dict[("c", "->", "x")]
        )

    @staticmethod
    def test_message():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = CondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        x = torch.rand(4, in_features)
        c = torch.rand(4, context_features)

        x_dict = dict(x=x, c=c)

        edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0], [1, 1], [2, 1]]).T}

        message = model.message(x_dict, edge_index_dict)

        assert isinstance(message, types.GeneratorType)

        for m_dst, dst in message:
            true_dst = edge_index_dict[("c", "->", "x")][-1]

            assert m_dst.shape == (true_dst.shape[0], hidden_features)
            tt.assert_equal(dst, edge_index_dict[("c", "->", "x")][-1])

    @staticmethod
    def test_aggregate():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = CondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        x = torch.rand(4, in_features)
        c = torch.rand(4, context_features)

        x_dict = dict(x=x, c=c)

        edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0], [1, 1], [2, 1]]).T}

        message = model.message(x_dict, edge_index_dict)

        aggr = model.aggregate(message, self_x=torch.randn(4, hidden_features))

        assert aggr.shape == (4, hidden_features)

    @staticmethod
    def test_aggregate_options():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        for aggr in ["mean", "max", "add"]:
            model = CondGCN(
                in_features=in_features,
                context_features=context_features,
                out_features=out_features,
                hidden_features=hidden_features,
                aggr=aggr,
            )

            x_dict = dict(
                x=torch.rand(1, in_features), c=torch.rand(1, context_features)
            )

            assert model.aggr == aggr

            edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0]]).T}

            model(x_dict, edge_index_dict)


class FastCondGCNTests(TestCase):
    @staticmethod
    def test_smoke():
        FastCondGCN()

    @staticmethod
    def test_layers_constructed():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = FastCondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        assert len(model.layers) == 2

        assert "x" in model.layers
        assert "c" in model.layers

        assert model.layers["x"].in_features == in_features
        assert model.layers["x"].out_features == hidden_features * 3

        assert model.layers["c"].in_features == context_features
        assert model.layers["c"].out_features == hidden_features * 2

        assert model.pool.in_features == hidden_features
        assert model.pool.out_features == out_features

    @staticmethod
    def test_forward():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = FastCondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        x = torch.rand(1, in_features)
        c = torch.rand(1, context_features)

        x_dict = dict(x=x, c=c)

        edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0]]).T}

        y_dict, res_edge_index_dict = model(x_dict, edge_index_dict)

        assert "x" in y_dict
        assert "c" in y_dict

        assert y_dict["x"].shape == (1, out_features)
        assert y_dict["c"].shape == (1, hidden_features)

        assert ("c", "->", "x") in res_edge_index_dict
        tt.assert_equal(
            res_edge_index_dict[("c", "->", "x")], edge_index_dict[("c", "->", "x")]
        )

    @staticmethod
    def test_message():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = FastCondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        x = torch.rand(4, in_features)
        c = torch.rand(4, context_features)

        x_dict = dict(x=x, c=c)

        edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0], [1, 1], [2, 1]]).T}

        x, c = x_dict["x"], x_dict["c"]

        embeds = {"x": model.act(model.layers["x"](x))}
        if c is not None:
            embeds["c"] = model.act(model.layers["c"](c))
            x_dict["c"] = embeds["c"][:, ..., : model.features["hidden"]]

        message = model.message(embeds, edge_index_dict)

        assert isinstance(message, types.GeneratorType)

        for m_dst, dst in message:
            true_dst = edge_index_dict[("c", "->", "x")][-1]

            assert m_dst.shape == (true_dst.shape[0], hidden_features)
            tt.assert_equal(dst, edge_index_dict[("c", "->", "x")][-1])

    @staticmethod
    def test_aggregate():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        model = FastCondGCN(
            in_features=in_features,
            context_features=context_features,
            out_features=out_features,
            hidden_features=hidden_features,
        )

        x = torch.rand(4, in_features)
        c = torch.rand(4, context_features)

        x_dict = dict(x=x, c=c)

        edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0], [1, 1], [2, 1]]).T}

        x, c = x_dict["x"], x_dict["c"]

        embeds = {"x": model.act(model.layers["x"](x))}
        if c is not None:
            embeds["c"] = model.act(model.layers["c"](c))
            x_dict["c"] = embeds["c"][:, ..., : model.features["hidden"]]

        self_x = embeds["x"][:, ..., : model.features["hidden"]]

        message = model.message(embeds, edge_index_dict)

        aggr = model.aggregate(message, self_x=self_x)

        assert aggr.shape == (4, hidden_features)

    @staticmethod
    def test_aggregate_options():
        in_features = 3
        context_features = 2
        out_features = 3
        hidden_features = 10

        for aggr in ["mean", "max", "add"]:
            model = FastCondGCN(
                in_features=in_features,
                context_features=context_features,
                out_features=out_features,
                hidden_features=hidden_features,
                aggr=aggr,
            )

            x_dict = dict(
                x=torch.rand(1, in_features), c=torch.rand(1, context_features)
            )

            assert model.aggr == aggr

            edge_index_dict = {("c", "->", "x"): torch.tensor([[0, 0]]).T}

            model(x_dict, edge_index_dict)
