import numpy as np
from propose.cameras import Camera


def create_mock_camera():
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0.5, 0.5],
        [0, 0.5, 0.5]
    ])

    translation_vector = np.array([
        [1, 2, 3]
    ])

    intrinsic_matrix = np.array([
        [1, 0, 0],
        [2, 3, 0],
        [4, 5, 1]
    ])

    tangential_distortion = np.array([[1, 2]])
    radial_distortion = np.array([[1, 2]])

    frame = np.arange(0, 10)

    return Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)