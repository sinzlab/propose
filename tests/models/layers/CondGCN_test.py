from propose.models.layers.CondGCN import CondGCN

import torch
import torch.testing as tt


def test_smoke():
    CondGCN()


def test_layers_constructed():
    in_features = 3
    context_features = 2
    out_features = 3
    hidden_features = 10

    model = CondGCN(
        in_features=in_features,
        context_features=context_features,
        out_features=out_features,
        hidden_features=hidden_features
    )

    assert len(model.layers) == 5

    assert 'x' in model.layers
    assert 'c' in model.layers
    assert 'x->x' in model.layers
    assert 'c->x' in model.layers
    assert 'x<-x' in model.layers

    assert model.layers['x'].in_features == in_features
    assert model.layers['x'].out_features == hidden_features

    assert model.layers['c'].in_features == context_features
    assert model.layers['c'].out_features == hidden_features

    assert model.layers['x->x'].in_features == in_features
    assert model.layers['x->x'].out_features == hidden_features

    assert model.layers['x<-x'].in_features == in_features
    assert model.layers['x<-x'].out_features == hidden_features

    assert model.layers['c->x'].in_features == context_features
    assert model.layers['c->x'].out_features == hidden_features

    assert model.pool.in_features == hidden_features
    assert model.pool.out_features == out_features


def test_forward():
    in_features = 3
    context_features = 2
    out_features = 3
    hidden_features = 10

    model = CondGCN(
        in_features=in_features,
        context_features=context_features,
        out_features=out_features,
        hidden_features=hidden_features
    )

    x = torch.rand(1, in_features)
    c = torch.rand(1, context_features)

    x_dict = dict(
        x=x,
        c=c
    )

    edge_index_dict = {('c', '->', 'x'): torch.tensor([[0, 0]]).T}

    y_dict, res_edge_index_dict = model(x_dict, edge_index_dict)

    assert 'x' in y_dict
    assert 'c' in y_dict

    assert y_dict['x'].shape == (1, out_features)
    assert y_dict['c'].shape == (1, hidden_features)

    assert ('c', '->', 'x') in res_edge_index_dict
    tt.assert_equal(res_edge_index_dict[('c', '->', 'x')], edge_index_dict[('c', '->', 'x')])
