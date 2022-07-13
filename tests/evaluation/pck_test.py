import unittest

import torch
import numpy.testing as npt

from propose.evaluation.pck import pck
from propose.utils.reproducibility import set_random_seed


class PCKTests(unittest.TestCase):
    def test_pck_computes_correctly(self):
        set_random_seed(1)
        p_pred = torch.randn((1, 17, 10, 3))
        p_true = torch.randn((1, 17, 10, 3))

        error = pck(p_true, p_pred, threshold=4)

        npt.assert_almost_equal(
            error.numpy(),
            torch.Tensor(
                [
                    [
                        0.9412,
                        0.8824,
                        0.9412,
                        0.8824,
                        0.9412,
                        1.0000,
                        1.0000,
                        1.0000,
                        1.0000,
                        0.9412,
                    ]
                ]
            ).numpy(),
            decimal=4,
        )

    def test_one_joint_wrong(self):
        a = torch.Tensor(
            [
                [
                    [
                        [1, 1, 1],
                    ],
                    [
                        [1, 1, 1],
                    ],
                    [
                        [1, 1, 1],
                    ],
                    [
                        [1, 1, 1],
                    ],
                ]
            ]
        )

        b = torch.Tensor(
            [
                [
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    [
                        [1, 10, 1],
                        [1, 1, 1],
                    ],
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                ]
            ]
        )

        n_joints = 4
        n_batches = 1
        n_samples = 2
        n_dim = 3

        self.assertEqual(a.dim(), 4)
        self.assertEqual(b.dim(), 4)

        self.assertEqual(a.shape, (n_batches, n_joints, 1, n_dim))
        self.assertEqual(b.shape, (n_batches, n_joints, n_samples, n_dim))

        error = pck(a, b, threshold=4)

        self.assertEqual(error.shape, (n_batches, n_samples))

        npt.assert_almost_equal(
            error.numpy(),
            torch.Tensor(
                [
                    [0.7500, 1.0000],
                ]
            ).numpy(),
            decimal=4,
        )
