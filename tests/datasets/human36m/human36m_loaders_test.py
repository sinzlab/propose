from propose.datasets.human36m.loaders import load_poses

from unittest.mock import patch
from unittest import TestCase

import numpy as np


class TestHuman36MPoseLoader(TestCase):
    @patch("propose.datasets.human36m.loaders.cdflib")
    def test_load_poses(self, cdflib_mock):
        cdflib_mock.CDF.return_value = np.random.random((1, 100, 96))

        path = "path/to/cdf/file"
        poses = load_poses(path)

        cdflib_mock.CDF.assert_called_once_with(path)

        self.assertTrue(isinstance(poses, np.ndarray))

    @patch("propose.datasets.human36m.loaders.cdflib")
    def test_check_loaded_poses_integrity(self, cdflib_mock):
        cdflib_mock.CDF.return_value = np.random.random((1, 100, 36))

        path = "path/to/cdf/file"

        with self.assertRaises(AssertionError):
            load_poses(path)

    @patch("propose.datasets.human36m.loaders.cdflib")
    def test_load_poses_output_shape(self, cdflib_mock):
        cdflib_mock.CDF.return_value = np.random.random((1, 100, 96))

        path = "path/to/cdf/file"
        poses = load_poses(path, every_n_frame=10)

        self.assertEqual(poses.shape, (10, 17, 3))
