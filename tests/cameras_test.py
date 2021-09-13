from propose.cameras import Camera
import numpy as np


def set_global_vars():
    global intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion
    intrinsic_matrix = np.random.random(size=(3, 3))
    rotation_matrix = np.random.random(size=(3, 3))
    translation_vector = np.random.random(size=(1, 3))
    tangential_distortion = np.random.random(size=(1, 2))
    radial_distortion = np.random.random(size=(1, 2))


def test_camera_init():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion)

    np.testing.assert_array_equal(camera.intrinsic_matrix, intrinsic_matrix)
    np.testing.assert_array_equal(camera.rotation_matrix, rotation_matrix)
    np.testing.assert_array_equal(camera.translation_vector, translation_vector)
    np.testing.assert_array_equal(camera.tangential_distortion, tangential_distortion)
    np.testing.assert_array_equal(camera.radial_distortion, radial_distortion)


def test_camera_matrix():
    set_global_vars()
    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion)

    camera_matrix = np.concatenate((rotation_matrix, translation_vector), axis=0) @ intrinsic_matrix

    np.testing.assert_array_equal(camera.camera_matrix(), camera_matrix)


def test_camera_projection():
    set_global_vars()
    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion)

    points = np.random.random(size=(1, 3))

    camera_matrix = np.concatenate((rotation_matrix, translation_vector), axis=0) @ intrinsic_matrix
    extended_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)

    projected_points = extended_points @ camera_matrix
    projected_points = projected_points[:, :2] / projected_points[:, 2:]

    np.testing.assert_array_equal(camera.proj2D(points), projected_points)
