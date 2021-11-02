from propose.models.nn.mlp import MLP
import torch


def test_smoke():
    MLP(
        in_shape=(3,),
        out_shape=(3,),
        hidden_sizes=[10, 10]
    )


def test_number_of_layers():
    hidden_sizes = [10, 10, 10]
    mlp = MLP(
        in_shape=(3,),
        out_shape=(3,),
        hidden_sizes=hidden_sizes
    )

    assert len(mlp._hidden_layers) == len(hidden_sizes) - 1


def test_forward_returns_grad():
    mlp = MLP(
        in_shape=(3,),
        out_shape=(3,),
        hidden_sizes=[10, 10]
    )

    x = torch.Tensor([[1, 1, 1]])
    y = mlp(x)

    assert y.grad_fn is not None
