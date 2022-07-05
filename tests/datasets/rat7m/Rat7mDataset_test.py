from propose.datasets.rat7m.Rat7mDataset import Rat7mDataset
from propose.poses.rat7m import Rat7mPose

from neuralpredictors.data.datasets.base import TransformDataset

from unittest.mock import MagicMock, patch, call

import numpy as np


@patch("propose.datasets.rat7m.Rat7mDataset.Rat7mPose")
@patch("propose.datasets.rat7m.Rat7mDataset.pickle")
@patch("builtins.open")
def test_is_a_transform_dataset(open_mock, pickle_mock, Rat7mPose_mock):
    dataset = Rat7mDataset(dirname="")

    assert isinstance(dataset, TransformDataset)


@patch("propose.datasets.rat7m.Rat7mDataset.Rat7mPose")
@patch("propose.datasets.rat7m.Rat7mDataset.pickle")
@patch("builtins.open")
def test_getting_data_key_from_path(*args):
    dataset = Rat7mDataset(dirname="/data_key")

    assert dataset.data_key == "data_key"


@patch("propose.datasets.rat7m.Rat7mDataset.Rat7mPose")
@patch("propose.datasets.rat7m.Rat7mDataset.pickle")
@patch("builtins.open")
def test_pose_and_cameras_loaded(open_mock, pickle_mock, Rat7mPose_mock):
    Rat7mPose_mock.load = MagicMock()

    dirname = "/"
    data_key = "data_key"

    dataset = Rat7mDataset(dirname=dirname + data_key, transforms=[])

    assert dataset.poses == Rat7mPose_mock.load()
    assert dataset.cameras == pickle_mock.load()
    assert Rat7mPose_mock.load.mock_calls[0] == call(
        f"{dirname + data_key}/poses/{data_key}.npy"
    )
    assert open_mock.mock_calls[0] == call(
        f"{dirname + data_key}/cameras/{data_key}.pickle", "rb"
    )


@patch("propose.datasets.rat7m.Rat7mDataset.Rat7mPose")
@patch("propose.datasets.rat7m.Rat7mDataset.pickle")
@patch("builtins.open")
def test_len(open_mock, pickle_mock, Rat7mPose_mock):
    poses = Rat7mPose(np.zeros((10, 20, 3)))
    cameras = {"Camera1": MagicMock(), "Camera2": MagicMock(), "Camera3": MagicMock()}

    Rat7mPose_mock.load = MagicMock(return_value=poses)
    pickle_mock.load = MagicMock(return_value=cameras)

    dirname = ""

    dataset = Rat7mDataset(dirname=dirname, transforms=[])

    assert len(dataset) == len(poses) * len(cameras)


@patch("propose.datasets.rat7m.Rat7mDataset.Rat7mPose")
@patch("propose.datasets.rat7m.Rat7mDataset.pickle")
@patch("builtins.open")
@patch("propose.datasets.rat7m.Rat7mDataset.imageio")
def test_getitem(imageio_mock, open_mock, pickle_mock, Rat7mPose_mock):
    poses = Rat7mPose(np.zeros((7000, 20, 3)))
    cameras = {"Camera1": MagicMock(), "Camera2": MagicMock(), "Camera3": MagicMock()}

    cameras["Camera1"].frames.__getitem__ = MagicMock(side_effect=lambda x: x)
    cameras["Camera2"].frames.__getitem__ = MagicMock(side_effect=lambda x: x)

    Rat7mPose_mock.load = MagicMock(return_value=poses)
    pickle_mock.load = MagicMock(return_value=cameras)

    imageio_mock.imread = MagicMock(return_value=np.zeros((100, 100, 3)))

    dirname = "/"
    data_key = "data_key"

    dataset = Rat7mDataset(dirname=dirname + data_key, transforms=[])

    # Test Cases

    # Selecting object for first camera
    data = dataset[0]
    camera = data.cameras
    pose = data.poses
    image = data.images

    assert camera == cameras["Camera1"]
    assert imageio_mock.imread.mock_calls[0] == call(
        f"{dirname + data_key}/images/{data_key}-camera1-0/{data_key}-camera1-00001.jpg"
    )
    np.testing.assert_array_equal(pose, poses[0])

    # Selecting object for second camera
    data = dataset[7000 * 1]
    camera = data.cameras
    pose = data.poses

    assert camera == cameras["Camera2"]
    assert imageio_mock.imread.mock_calls[1] == call(
        f"{dirname + data_key}/images/{data_key}-camera2-0/{data_key}-camera2-00001.jpg"
    )
    np.testing.assert_array_equal(pose, poses[0])

    # Selecting object second pose for second camera
    data = dataset[7000 * 1 + 1]
    camera = data.cameras
    pose = data.poses

    assert camera == cameras["Camera2"]
    assert imageio_mock.imread.mock_calls[2] == call(
        f"{dirname + data_key}/images/{data_key}-camera2-0/{data_key}-camera2-00002.jpg"
    )
    np.testing.assert_array_equal(pose, poses[1])

    # Selecting object from second chunk for first camera
    data = dataset[3500]
    camera = data.cameras
    pose = data.poses

    assert camera == cameras["Camera1"]
    assert imageio_mock.imread.mock_calls[3] == call(
        f"{dirname + data_key}/images/{data_key}-camera1-3500/{data_key}-camera1-00001.jpg"
    )
    np.testing.assert_array_equal(pose, poses[0])


@patch("propose.datasets.rat7m.Rat7mDataset.Rat7mPose")
@patch("propose.datasets.rat7m.Rat7mDataset.pickle")
@patch("builtins.open")
@patch("propose.datasets.rat7m.Rat7mDataset.imageio")
def test_getitem_misaligned_image_pose(
    imageio_mock, open_mock, pickle_mock, Rat7mPose_mock
):
    poses = np.zeros((7000, 20, 3))
    cameras = {"Camera1": MagicMock(), "Camera2": MagicMock(), "Camera3": MagicMock()}

    cameras["Camera1"].frames.__getitem__ = MagicMock(side_effect=lambda x: x + 1)
    cameras["Camera2"].frames.__getitem__ = MagicMock(side_effect=lambda x: x + 1)

    Rat7mPose_mock.load = MagicMock(return_value=poses)
    pickle_mock.load = MagicMock(return_value=cameras)

    imageio_mock.imread = MagicMock(return_value=np.zeros((100, 100, 3)))

    dirname = "/"
    data_key = "data_key"

    dataset = Rat7mDataset(dirname=dirname + data_key, transforms=[])

    # Selecting object for first camera
    data = dataset[0]
    camera = data.cameras
    pose = data.poses

    assert camera == cameras["Camera1"]
    assert imageio_mock.imread.mock_calls[0] == call(
        f"{dirname + data_key}/images/{data_key}-camera1-0/{data_key}-camera1-00002.jpg"
    )
    np.testing.assert_array_equal(pose, poses[0])
