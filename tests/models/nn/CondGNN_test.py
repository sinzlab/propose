from propose.models.nn.CondGNN import CondGNN
from propose.models.layers.CondGCN import CondGCN, FastCondGCN

from torch_geometric.data import HeteroData

import torch.testing as tt

import torch

from unittest.mock import MagicMock, patch, call


def test_smoke():
    CondGNN()


def test_use_fast_or_slow():
    model = CondGNN()
    assert model.gcn is CondGCN

    model = CondGNN(gcn_type="fast")
    assert model.gcn is FastCondGCN


def test_architecture_should_have_2_layers():
    model = CondGNN()

    assert len(model.layers) == 2

    assert isinstance(model.layers[0], CondGCN)
    assert isinstance(model.layers[1], CondGCN)


def test_get_x_dict_without_c():
    model = CondGNN()

    x = torch.rand(10, 3)
    data = HeteroData(
        {"x": dict(x=x), ("x", "->", "x"): dict(edge_index=torch.zeros(2, 10))}
    )

    data_dict = model._get_x_dict(data)

    assert isinstance(data_dict, dict)

    tt.assert_equal(data_dict["x"], x)

    assert data_dict["c"] is None

    assert ("x", "->", "x") not in data_dict
    assert ("x", "->", "x") in data.to_dict()


def test_get_x_dict_with_c():
    model = CondGNN()

    x = torch.rand(10, 3)
    c = torch.rand(10, 2)
    data = HeteroData(
        {
            "x": dict(x=x),
            "c": dict(x=c),
            ("x", "->", "x"): dict(edge_index=torch.zeros(2, 10)),
        }
    )

    data_dict = model._get_x_dict(data)

    assert isinstance(data_dict, dict)

    tt.assert_equal(data_dict["x"], x)
    tt.assert_equal(data_dict["c"], c)

    assert ("x", "->", "x") not in data_dict
    assert ("x", "->", "x") in data.to_dict()


def test_get_edge_index():
    model = CondGNN()

    edge_index = torch.zeros(2, 10)
    x = torch.rand(10, 3)

    data = HeteroData({"x": dict(x=x), ("x", "->", "x"): dict(edge_index=edge_index)})

    edge_index_dict = model._get_edge_index(data)

    assert isinstance(edge_index_dict, dict)

    assert ("x", "->", "x") in edge_index_dict
    assert "x" not in edge_index_dict

    tt.assert_equal(edge_index_dict[("x", "->", "x")], edge_index)


def test_get_edge_index_empty():
    model = CondGNN()
    x = torch.rand(10, 3)

    data = HeteroData({"x": dict(x=x)})

    edge_index_dict = model._get_edge_index(data)

    assert isinstance(edge_index_dict, dict)

    assert len(edge_index_dict.keys()) == 0
    assert "x" not in edge_index_dict


@patch("propose.models.nn.CondGNN.nn.ModuleList")
@patch("propose.models.nn.CondGNN.CondGCN")
def test_forward(cond_gcn_mock, module_list_mock):
    x_return_value = torch.randn(10, 3)
    cond_gcn_mock.return_value = (
        dict(x=x_return_value),
        dict(x=2),
    )  # x_dict, edge_index_dict
    module_list_mock.return_value = [cond_gcn_mock, cond_gcn_mock]

    in_features = 3
    context_features = 2
    out_features = 3
    hidden_features = 10

    model = CondGNN(
        in_features=in_features,
        context_features=context_features,
        out_features=out_features,
        hidden_features=hidden_features,
    )

    x = torch.rand(10, 3)

    data = HeteroData({"x": dict(x=x)})

    out = model.forward(data)

    tt.assert_equal(out, x_return_value)

    assert len(cond_gcn_mock.mock_calls) == 4

    assert cond_gcn_mock.mock_calls[0] == call(
        in_features, context_features, hidden_features, hidden_features
    )
    assert cond_gcn_mock.mock_calls[1] == call(
        hidden_features, hidden_features, out_features, hidden_features
    )
