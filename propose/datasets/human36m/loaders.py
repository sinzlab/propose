import os
import cdflib
import numpy as np
import pickle

from pathlib import Path
from typing import Union

from propose.poses.utils import load_data_ids


def load_poses(path):
    """
    Loads the poses from the .cdf file.
    :param path: Path to the .cdf file.
    :return: A numpy array with the poses. (N, 17, 3)
    """
    file = cdflib.CDF(path)

    poses_3d = file[0].squeeze()
    assert (
        poses_3d.shape[1] == 96
    ), f"Wrong number of joints, expected 96, got {poses_3d.shape[1]}"

    dirname = os.path.dirname(__file__)
    metadata_path = os.path.join(dirname, "../../poses/metadata/human36m.yaml")
    joints = load_data_ids(metadata_path)

    # Select only the joints we want
    poses_3d = poses_3d.reshape(-1, 32, 3)[:, joints]

    # Reformat the data to be compatible with the propose pose format.
    poses_3d = poses_3d - poses_3d[:, :1]  # center around root
    # data is saved in x, z, y order in the .cdf file, so we need to swap the z, y axes
    poses_3d[..., [1, 2]] = poses_3d[..., [2, 1]]
    poses_3d[..., 2] = -poses_3d[..., 2]  # flip the z axis

    return poses_3d


def pickle_poses(input_dir_path: Union[str, Path], output_dir_path: Union[str, Path]):
    """
    Loads the poses from the .cdf file and saves them as a pickle file.
    :param input_dir_path: Path to the directory containing the .cdf files.
    :param output_dir_path: Path to the directory where the pickle files will be saved.
    """
    if isinstance(input_dir_path, str):
        input_dir_path = Path(input_dir_path)

    subjects = [subject.name for subject in input_dir_path.glob("*")]

    for subject in subjects:
        pose_dir = input_dir_path / subject / "MyPoseFeatures" / "D3_Positions_mono"

        dataset = {}
        for path in pose_dir.glob("*.cdf"):
            poses = load_poses(path)

            action = path.name.split(".")[0]
            camera = path.name.split(".")[1]
            n_frames = len(poses)

            if camera in dataset:
                dataset[camera]["poses"].append(poses)
                dataset[camera]["actions"] += [action] * n_frames
            else:
                dataset[camera] = {"poses": [poses], "actions": [action] * n_frames}

        for camera in dataset:
            dataset[camera]["poses"] = np.concatenate(dataset[camera]["poses"])

            output_dir_path.mkdir(exist_ok=True)
            (output_dir_path / subject).mkdir(exist_ok=True)

            file_path = (
                output_dir_path
                / subject
                / f"poses_cam-{camera}_frames-{len(poses)}.pkl"
            )

            with open(file_path, "wb") as f:
                pickle.dump(dataset[camera], f)
