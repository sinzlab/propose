from propose.models.flows.glow import ConditionalGlow

from nflows.transforms.normalization import ActNorm
from nflows.transforms.lu import LULinear
from nflows.transforms.coupling import AdditiveCouplingTransform


def test_smoke():
    ConditionalGlow()


def test_layers():
    BLOCK_SIZE = 3

    num_layers = 2
    hidden_sizes = [10, 10, 10]
    glow = ConditionalGlow(num_layers=num_layers, hidden_sizes=hidden_sizes)

    assert len(glow._transform._transforms) == num_layers * BLOCK_SIZE

    assert isinstance(glow._transform._transforms[0], ActNorm)
    assert isinstance(glow._transform._transforms[1], LULinear)
    assert isinstance(glow._transform._transforms[2], AdditiveCouplingTransform)

    assert len(glow._transform._transforms[2].transform_net._hidden_layers) == len(hidden_sizes) - 1
