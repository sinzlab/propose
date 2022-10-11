import torch
import torch.testing as tt
from propose.models.nn.embedding import (Embedding, JoinEmbedding,
                                         LinearEmbedding, SplitEmbedding,
                                         SplitLinearEmbedding)


def test_smoke():
    Embedding()
    LinearEmbedding(1, 1)
    SplitEmbedding(1)
    SplitLinearEmbedding(1, 1, 1)


def test_embedding_base():
    embedding = Embedding()
    x = torch.Tensor([[1, 2, 3]])
    y = embedding(x)

    tt.assert_equal(x, y)

    a = torch.Tensor([[1, 1, 1]])
    y = embedding((x, a))

    tt.assert_equal(x, y[0])
    tt.assert_equal(a, y[1])


def test_embedding_linear():
    embedding = LinearEmbedding(3, 10)
    x = torch.Tensor([[1, 2, 3]])
    y = embedding(x)

    assert y.shape == (1, 10)

    a = torch.Tensor([[1, 1, 1]])
    y = embedding((x, a))

    assert y[0].shape == (1, 10)
    tt.assert_equal(a, y[1])


def test_embedding_split():
    # Split 1
    embedding = SplitEmbedding(1)
    x = torch.Tensor([[1, 2, 3, 4]])
    y = embedding(x)

    tt.assert_equal(y[0], torch.Tensor([[1]]), check_stride=False)
    tt.assert_equal(y[1], torch.Tensor([[2, 3, 4]]), check_stride=False)

    # Split 2
    embedding = SplitEmbedding(2)
    x = torch.Tensor([[1, 2, 3, 4]])
    y = embedding(x)

    tt.assert_equal(y[0], torch.Tensor([[1, 2]]), check_stride=False)
    tt.assert_equal(y[1], torch.Tensor([[3, 4]]), check_stride=False)

    # Split 3
    embedding = SplitEmbedding(3)
    x = torch.Tensor([[1, 2, 3, 4]])
    y = embedding(x)

    tt.assert_equal(y[0], torch.Tensor([[1, 2, 3]]), check_stride=False)
    tt.assert_equal(y[1], torch.Tensor([[4]]), check_stride=False)


def test_embedding_join():
    embedding = JoinEmbedding()
    x1 = torch.Tensor([[1, 2, 3, 4]])
    x2 = torch.Tensor([[5, 6, 7, 8]])

    y = embedding((x1, x2))

    tt.assert_equal(y, torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))


def test_embedding_split_linear():
    embedding = SplitLinearEmbedding(1, 1, 10)
    x = torch.Tensor([[1, 2, 3]])
    y = embedding(x)

    assert y.shape == (1, 12)
