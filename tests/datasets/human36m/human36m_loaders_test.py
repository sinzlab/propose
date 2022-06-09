from propose.datasets.human36m.loaders import load_poses, pickle_poses

from unittest.mock import patch, MagicMock
from unittest import TestCase

import numpy as np

from pathlib import Path


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
        poses = load_poses(path)

        self.assertEqual(poses.shape, (100, 17, 3))


@patch("propose.datasets.human36m.loaders.load_poses")
@patch("propose.datasets.human36m.loaders.pickle")
class TestHuman36mPickle(TestCase):
    def test(self, pickle_mock, load_poses_mock):
        load_poses_mock.return_value = np.zeros((100, 17, 3))
        input_dir_path = MagicMock()
        input_dir_path.glob.return_value = [Path("path/to/cdf/S1")]
        input_dir_path.__truediv__().__truediv__().__truediv__().glob.return_value = [
            Path("path/to/cdf/action.camera")
        ]

        output_dir_path = MagicMock()

        pickle_poses(input_dir_path, output_dir_path)

        assert str(load_poses_mock.mock_calls[0][1][0]) == "path/to/cdf/action.camera"

        assert output_dir_path.mock_calls[-4:][0][1][0] == "S1"
        assert output_dir_path.mock_calls[-4:][1][1][0] == "camera"
        assert output_dir_path.mock_calls[-4:][2][1][0] == f"100.pkl"

        dataset = pickle_mock.mock_calls[0][1][0]
        assert set(dataset.keys()) == {"poses", "actions"}
        assert dataset["poses"].shape == (100, 17, 3)
        assert dataset["actions"] == ["action"] * 100
