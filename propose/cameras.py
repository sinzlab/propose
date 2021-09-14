import numpy as np

Point2D = np.ndarray
Point3D = np.ndarray


class Camera(object):
    """
    Camera class for managing camera related operations and storing camera data.
    """
    def __init__(self, intrinsic_matrix: np.ndarray, rotation_matrix: np.ndarray, translation_vector: np.ndarray,
                 tangential_distortion: np.ndarray, radial_distortion: np.ndarray):
        """
        :param intrinsic_matrix: 3x3 matrix, transforms the 3D camera coordinates to 2D homogeneous image coordinates.
        :param rotation_matrix: 3x3 matrix, describes the camera's rotation in space.
        :param translation_vector: 1x3 vector, describes the cameras location in space.
        :param tangential_distortion: 1x2 vector, describes the distortion between the lens and the image plane.
        :param radial_distortion: 1x2 or 1x3 vector, describes how light bends near the edges of the lens.
        """

        self.intrinsic_matrix = intrinsic_matrix
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.tangential_distortion = tangential_distortion
        self.radial_distortion = radial_distortion

    def camera_matrix(self) -> np.ndarray:
        """
        Computes the camera matrix (M) from the rotation matrix (R) translation vector (t) and intrinsic matrix (C)
        :return: Camera matrix. M = [R | t]C
        """
        return np.concatenate((self.rotation_matrix, self.translation_vector), axis=0) @ self.intrinsic_matrix

    def proj2D(self, points: Point3D) -> Point2D:
        """
        Computes the projection of a 3D point onto the 2D camera space
        :param points: 3D points (x, y, z)
        :return: Projected 2D points (x, y)
        """
        assert points.shape[1] == 3

        camera_matrix = self.camera_matrix()

        extended_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

        projected_points = extended_points @ camera_matrix  # (u, v, z)
        projected_points = projected_points[:, :2] / projected_points[:, 2:]  # (u/z, v/z)

        return projected_points

    def distort_points(self, points: Point2D) -> Point2D:
        """
        Applies the radial and tangetial distortion to the pixel points.
        :param points: Undistorted 2D points in pixel space.
        :return: Distorted 2D points in pixel space.
        """
        image_points = self._pixel_to_image_points(points)

        kappa = self._radial_distortion(image_points)
        rho = self._tangential_distortion(image_points)

        distorted_image_points = image_points * kappa + rho

        pixel_points = self._image_to_pixel_points(distorted_image_points)

        return pixel_points

    def _unpack_intrinsic_matrix(self) -> tuple[float, float, float, float, float]:
        """
        Unpacks the intrinsic matrix which is of the format
            [ fx   , 0 , 0 ]
        K = [ skew , fy, 0 ]
            [ cx   , cy, 1 ]

        :return: fx, fy, cx, cy, skew
        """
        cx = self.intrinsic_matrix[2, 0]
        cy = self.intrinsic_matrix[2, 1]
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        skew = self.intrinsic_matrix[1, 0]

        return fx, fy, cx, cy, skew

    def _pixel_to_image_points(self, pixel_points: Point2D) -> Point2D:
        """
        Transforms points from pixel space to image space by computing
        x' = (x - cx) / fx
        y' = (y - cy) / fy

        :param pixel_points: 2D points in pixel space
        :return: 2D points in normalised image space
        """
        fx, fy, cx, cy, skew = self._unpack_intrinsic_matrix()

        centered_points = pixel_points - np.array([[cx, cy]])

        y_norm = centered_points[:, 1] / fy
        x_norm = (centered_points[:, 0] - skew * y_norm) / fx

        return np.stack([x_norm, y_norm], axis=1)

    def _image_to_pixel_points(self, image_points: Point2D) -> Point2D:
        """
        Transforms points from image space to pixel space by computing
        x' = x * fx + cx
        y' = y * fy + cy

        :param image_points: 2D points in the normalised image space
        :return: 2D points in pixel space
        """
        fx, fy, cx, cy, skew = self._unpack_intrinsic_matrix()

        pixel_points = np.stack([
            (image_points[:, 0] * fx) + cx + (skew * image_points[:, 1]),
            image_points[:, 1] * fy + cy
        ], axis=1)

        return pixel_points

    def _radial_distortion(self, image_points: Point2D) -> np.ndarray:
        """
        Occurs when light rays bend near the edge of the lens.
        Radial Distortion:
         x_dist = x(1 + k1*r^2 + k2*r^4 + k3*r^6)
         y_dist = y(1 + k1*r^2 + k2*r^4 + k3*r^6)
        where x, y are normalised in image coordinates nad translated to the optical center (x - cx) / fx, (y - cy) / fy.
        ki are the distortion coefficients.
        r^2 = x^2 + y^2
        :param image_points: 2D points in the normalised image space
        :return: (1 + k1*r^2 + k2*r^4 + k3*r^6)
        """
        r2 = image_points.__pow__(2).sum(axis=1)
        r4 = r2 ** 2
        r6 = r2 ** 3

        k = np.zeros(3)
        k[:self.radial_distortion.shape[1]] = self.radial_distortion.squeeze()

        kappa = 1 + (k[0] * r2) + (k[1] * r4) + (k[2] * r6)

        return kappa

    def _tangential_distortion(self, image_points: Point2D) -> np.ndarray:
        """
        Occurs when the lens and image plane are not in parallel.
        Tangential Distortion:
         x_dist = x + [2 * p1 * x * y + p2 * (r^2 + 2 * x^2)]
         y_dist = y + [2 * p2 * x * y + p1 * (r^2 + 2 * y^2)]

        p1 and p2 are tangential distortion coefficients.
        :param image_points: 2D points in the normalised image space
        :return: [dx, dy]
        """
        p = self.tangential_distortion.squeeze()

        r2 = image_points.__pow__(2).sum(axis=1)

        rho = np.array([
            2 * p[0] * image_points[:, 0] * image_points[:, 1] + p[1] * (r2 + 2 * image_points[:, 0] ** 2),
            2 * p[1] * image_points[:, 0] * image_points[:, 1] + p[0] * (r2 + 2 * image_points[:, 1] ** 2)
        ]).T

        return rho
