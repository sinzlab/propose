from propose.cameras import Camera
from propose.poses import Rat7mPose

import scipy.io as sio

import numpy as np


def load_cameras(path: str) -> dict:
    """
    Loads the camera parameters for the Rat7M dataset used for mocap.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: dict of cameras.
    """
    data = sio.loadmat(path, struct_as_record=False)
    camera_data = vars(data["cameras"][0][0])

    camera_names = camera_data['_fieldnames']

    cameras = {}
    for camera_name in camera_names:
        camera_calibration = vars(camera_data[camera_name][0][0])

        camera = Camera(intrinsic_matrix=camera_calibration['IntrinsicMatrix'],
                        rotation_matrix=camera_calibration['rotationMatrix'],
                        translation_vector=camera_calibration['translationVector'],
                        tangential_distortion=camera_calibration['TangentialDistortion'],
                        radial_distortion=camera_calibration['RadialDistortion'],
                        frames=camera_calibration['frame'])

        cameras[camera_name] = camera

    return cameras


def load_mocap(path: str) -> Rat7mPose:
    """
    Loads mocap datafor the Rat7M dataset.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: [Nd array] a Rat7mPose of [frame, joint, xyz]
    """
    d = sio.loadmat(path, struct_as_record=False)
    dataset = vars(d["mocap"][0][0])

    dataset.pop('_fieldnames')

    marker_names = Rat7mPose.marker_names

    n_frames = dataset[marker_names[0]].shape[0]
    n_poses = 20
    n_dims = 3

    poses = np.empty((n_frames, n_poses, n_dims))
    for idx, marker_name in enumerate(marker_names):
         poses[:, idx, :] = dataset[marker_name]

    return Rat7mPose(poses)

