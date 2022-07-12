import unittest

import torch
import numpy.testing as npt

from propose.evaluation.pck import pck
from propose.utils.reproducibility import set_random_seed


class PCKTests(unittest.TestCase):
    def test_pck_computes_correctly(self):
        set_random_seed(1)
        p_pred = torch.randn((10, 17, 3))
        p_true = torch.randn((10, 17, 3))

        error = pck(p_true, p_pred, threshold=4)

        npt.assert_array_equal(
            error.numpy(),
            torch.BoolTensor(
                [True, True, False, True, False, False, True, True, False, False]
            ).numpy(),
        )
