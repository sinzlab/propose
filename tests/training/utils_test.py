import torch

import numpy.testing as npt

from torch_geometric.data import HeteroData
from propose.training.utils import get_x_graph


def test_only_x_is_returned():
    x = torch.rand(10, 3)
    c = torch.rand(10, 2)

    data = HeteroData({
        'x': dict(x=x),
        'c': dict(x=c),
        ('x', '->', 'x'): dict(edge_index=torch.ones(2, 10)),
        ('c', '->', 'x'): dict(edge_index=torch.zeros(2, 10))
    })

    x_graph = get_x_graph(data)
    x_graph_dict = x_graph.to_dict()

    assert 'c' not in x_graph_dict
    assert ('c', '->', 'x') not in x_graph_dict

    assert 'x' in x_graph_dict
    assert ('x', '->', 'x') in x_graph_dict

    npt.assert_array_equal(x_graph_dict['x']['x'], x)
    npt.assert_array_equal(x_graph_dict[('x', '->', 'x')]['edge_index'], torch.ones(2, 10))
