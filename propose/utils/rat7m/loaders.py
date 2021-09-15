from propose.cameras import Camera
from propose.poses import Rat7mPose, PoseSet

import scipy.io as sio


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
                        frame=camera_calibration['frame'])

        cameras[camera_name] = camera

    return cameras


def load_mocap(path: str) -> PoseSet:
    """
    Loads mocap datafor the Rat7M dataset.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: [Nd array] a PoseSet of [frame, joint, xyz]
    """
    d = sio.loadmat(path, struct_as_record=False)
    dataset = vars(d["mocap"][0][0])

    dataset.pop('_fieldnames')

    poses = []
    n_poses = list(dataset.values())[0].shape[0]

    for pose_idx in range(n_poses):
        pose_dict = {marker_name: dataset[marker_name][pose_idx] for marker_name in dataset}
        poses.append(Rat7mPose(**pose_dict))

    return PoseSet(poses)
