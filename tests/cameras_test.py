from propose.cameras import Camera
import numpy as np


def set_global_vars():
    np.random.seed(1)

    global intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, camera_matrix, frame

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

    camera_matrix = np.array([
        [1., 0., 0.],
        [3., 4., 0.5],
        [3., 4., 0.5],
        [17., 21., 3.]
    ])

    frame = np.array([0])


def test_camera_init():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    np.testing.assert_array_equal(camera.intrinsic_matrix, intrinsic_matrix)
    np.testing.assert_array_equal(camera.rotation_matrix, rotation_matrix)
    np.testing.assert_array_equal(camera.translation_vector, translation_vector)
    np.testing.assert_array_equal(camera.tangential_distortion, tangential_distortion)
    np.testing.assert_array_equal(camera.radial_distortion, radial_distortion)


def test_camera_matrix():
    """
    Camera matrix is defined as M = [R | t]C
    """
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    np.testing.assert_array_equal(camera.camera_matrix(), camera_matrix)


def test_camera_proj2D_without_distortion():
    """
    Projection of points is computed given a camera matrix M and a 3D set of points (x, y, z).
    We construct a quaternion Q (x, y, z, w), with w=1.
    The projected quaternion is QM = (u, v, z).
    The 2D points are then p2D = (u/z, v/z).
    """
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    quaternion = np.array([[1, 2, 3, 1]])
    camera_matrix = camera.camera_matrix()

    projected_points = quaternion @ camera_matrix
    projected_points = projected_points[:, :2] / projected_points[:, 2:]

    np.testing.assert_array_equal(camera.proj2D(quaternion[:, :3], distort=False), projected_points)


def test_camera_proj2D_with_distortion():
    """
    Projection of points is computed given a camera matrix M and a 3D set of points (x, y, z).
    We construct a quaternion Q (x, y, z, w), with w=1.
    The projected quaternion is QM = (u, v, z).
    The 2D points are then p2D = (u/z, v/z).
    """
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    quaternion = np.array([[1, 2, 3, 1]])
    camera_matrix = camera.camera_matrix()

    projected_points = quaternion @ camera_matrix
    projected_points = projected_points[:, :2] / projected_points[:, 2:]
    projected_points = camera.distort(projected_points)

    np.testing.assert_array_equal(camera.proj2D(quaternion[:, :3], distort=True), projected_points)


def test_camera_proj2D_with_single_point():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    proj = camera.proj2D(np.array([1, 2, 3]))


def test_unpack_intrinsic_matrix():
    """
    Unpack the intrinsic matrix
        [ fx   , 0 , 0 ]
    K = [ skew , fy, 0 ]
        [ cx   , cy, 1 ]
    should return fx, fy, cx, cy, skew
    """
    set_global_vars()
    fx = 1
    fy = 2
    cx = 3
    cy = 4
    skew = 5

    intrinsic_matrix = np.array([
        [fx, 0, 0],
        [skew, fy, 0],
        [cx, cy, 1]
    ])

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    _fx, _fy, _cx, _cy, _skew = camera._unpack_intrinsic_matrix()

    assert fx == _fx
    assert fy == _fy
    assert cx == _cx
    assert cy == _cy
    assert skew == _skew


def test_pixel_to_image_points():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    points = np.array([[1, 2]])
    normalised_points = np.array([[-1, -1]])

    np.testing.assert_array_equal(camera._pixel_to_image_points(points), normalised_points)


def test_image_to_pixel_points():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    points = np.array([[1, 2]])
    normalised_points = np.array([[-1, -1]])

    np.testing.assert_array_equal(camera._image_to_pixel_points(normalised_points), points)


def test_radial_distortion():
    """
    Radial Distortion:
     x_dist = x(1 + k1*r^2 + k2*r^4 + k3*r^6)
     y_dist = y(1 + k1*r^2 + k2*r^4 + k3*r^6)
    where x, y are normalised in image coordinates nad translated to the optical center (x - cx) / fx, (y - cy) / fy.
    ki are the distortion coefficients.
    r^2 = x^2 + y^2
    """
    set_global_vars()

    global radial_distortion

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    normalised_points = np.array([[1, 1], [1, 1]])
    distortion = np.array([11, 11])

    np.testing.assert_array_equal(camera._radial_distortion(normalised_points), distortion)

    radial_distortion = np.array([[1, 2, 3]])
    normalised_points = np.array([[1, 1]])
    distortion = np.array([35])
    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    np.testing.assert_array_equal(camera._radial_distortion(normalised_points), distortion)


def test_tangential_distortion():
    """
    Tangential Distortion:
     x_dist = x + [2 * p1 * x * y + p2 * (r^2 + 2 * x^2)]
     y_dist = y + [2 * p2 * x * y + p1 * (r^2 + 2 * y^2)]

    p1 and p2 are tangential distortion coefficients.
    """
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    normalised_points = np.array([[1, 1], [1, 1]])
    distortion = np.array([[10, 8], [10, 8]])

    np.testing.assert_array_equal(camera._tangential_distortion(normalised_points), distortion)


def test_distort_single_point():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    points = np.array([[1, 2]])
    pixel_points = np.array([[-3, -4]])

    np.testing.assert_array_equal(camera.distort(points), pixel_points)


def test_distort_multi_points():
    set_global_vars()

    camera = Camera(intrinsic_matrix, rotation_matrix, translation_vector, tangential_distortion, radial_distortion, frame)

    points = np.array([[1, 2], [1, 2], [1, 2]])
    pixel_points = np.array([[-3, -4], [-3, -4], [-3, -4]])

    np.testing.assert_array_equal(camera.distort(points), pixel_points)
