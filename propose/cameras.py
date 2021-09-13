import numpy as np


class Camera(object):
    """
    Camera class for managing camera related operations and storing camera data.
    """
    def __init__(self, intrinsic_matrix: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray,
                 tangential_distortion: np.ndarray, radial_distortion: np.ndarray):
        assert intrinsic_matrix.shape == (3, 3)
        assert rotation_matrix.shape == (3, 3)
        assert translation_vector.shape == (1, 3)
        assert tangential_distortion.shape == (1, 2)
        assert radial_distortion.shape == (1, 2)

        self.intrinsic_matrix = intrinsic_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.tangential_distortion = tangential_distortion
        self.radial_distortion = radial_distortion

    def camera_matrix(self):
        """
        Computes the camera matrix from the rotation matrix (R) translation vector (t) and intrinsic matrix (C)
        :return: M = [R | t]C
        """
        return np.concatenate((self.rotation_matrix, self.translation_vector), axis=0) @ self.intrinsic_matrix

    def proj2D(self, points: np.ndarray) -> np.ndarray:
        """
        Computes the projection of a 3D point onto the 2D camera space
        :param points: 3D points
        :return: Projected 2D points
        """
        assert points.shape[1] == 3

        camera_matrix = self.camera_matrix()

        extended_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

        projected_points = extended_points @ camera_matrix
        projected_points = projected_points[:, :2] / projected_points[:, 2:]

        return projected_points
