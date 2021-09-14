from propose.cameras import Camera

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
                        frame=camera_calibration['frame'])

        cameras[camera_name] = camera

    return cameras


def load_mocap(path: str) -> np.ndarray:
    """
    Loads mocap datafor the Rat7M dataset.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: [Nd array] Mocap positions [frame, xyz, joint]
    """
    d = sio.loadmat(path, struct_as_record=False)
    dataset = vars(d["mocap"][0][0])

    markernames = dataset['_fieldnames']

    mocap = []
    for i in range(len(markernames)):
        mocap.append(dataset[markernames[i]])

    return np.stack(mocap, axis=2)
