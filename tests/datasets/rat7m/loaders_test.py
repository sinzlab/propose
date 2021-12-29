from propose.cameras import Camera
from propose.poses import Rat7mPose
from propose.datasets.rat7m.loaders import (
    load_mocap,
    load_cameras,
    temporal_split_dataset,
    static_loader,
)

import propose.datasets.rat7m.transforms as tr
from neuralpredictors.data.transforms import ScaleInputs, ToTensor

from unittest.mock import MagicMock, patch

from torch.utils.data import DataLoader

import numpy as np

path = "./tests/mock/data/mocap-mock.mat"


def test_rat7m_mocap_loaded():
    mocap = load_mocap(path)

    assert isinstance(mocap, Rat7mPose)

    assert mocap.shape[1] == 20
    assert mocap.shape[2] == 3


def test_rat7m_camera_loaded():
    cameras = load_cameras(path)

    assert list(cameras.keys()) == [
        "Camera1",
        "Camera2",
        "Camera4",
        "Camera5",
        "Camera3",
        "Camera6",
    ]
    assert isinstance(cameras["Camera1"], Camera)


def test_temporal_split_dataset():
    dataset = MagicMock()
    dataset.poses = np.arange(0, 10)
    dataset.cameras = np.arange(0, 2)

    train_frac = 0.6
    validation_frac = 0.2

    idx_train = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15])
    idx_val = np.array([6, 7, 16, 17])
    idx_test = np.array([8, 9, 18, 19])

    dat = temporal_split_dataset(
        dataset, train_frac=train_frac, validation_frac=validation_frac
    )

    np.testing.assert_array_equal(dat.train, idx_train)
    np.testing.assert_array_equal(dat.validation, idx_val)
    np.testing.assert_array_equal(dat.test, idx_test)


@patch("propose.datasets.rat7m.loaders.Rat7mDataset")
def test_static_loader(dataset):
    dataset().poses = np.arange(0, 10)
    dataset().cameras = np.arange(0, 2)

    dls = static_loader("", batch_size=1)

    transforms = dataset.mock_calls[-1][2]["transforms"]

    assert isinstance(transforms[-1], ToTensor)
    assert isinstance(transforms[-2], tr.ToGraph)

    assert list(dls.keys()) == ["train", "validation", "test"]
    assert isinstance(dls["train"], DataLoader)
    assert isinstance(dls["validation"], DataLoader)
    assert isinstance(dls["test"], DataLoader)
