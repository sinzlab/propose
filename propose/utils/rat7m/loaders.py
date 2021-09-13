import scipy.io as sio
import numpy as np


def load_cameras(path: str) -> dict:
    """
    Loads the camera parameters for the Rat7M dataset used for mocap.
    :param path: path to the mocap file (e.g. /path/to/mocap-s4-d1.mat)
    :return: dict of cameras.
    """
    d = sio.loadmat(path, struct_as_record=False)
    dataset = vars(d["cameras"][0][0])

    camnames = dataset['_fieldnames']

    cameras = {}
    for i in range(len(camnames)):
        cameras[camnames[i]] = {}
        cam = vars(dataset[camnames[i]][0][0])
        fns = cam['_fieldnames']
        for fn in fns:
            cameras[camnames[i]][fn] = cam[fn]

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


if __name__ == '__main__':
    from pathlib import Path
    home = str(Path.home())
    path = f'{home}/data/rat7m/mocap-s4-d1.mat'
    data = sio.loadmat(path)

    for camera_idx in range(6):
        for i in range(6):
            data['cameras'][0][0][camera_idx][0][0][i] = 0

    for join_idx in range(20):
        data['mocap'][0][0][join_idx] = np.zeros((1, 3))

    sio.savemat('../../../tests/mock_data/mocap-mock.mat', data)

