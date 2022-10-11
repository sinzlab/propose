import numpy as np
import numpy.testing as npt
from propose.models.distributions.StandardNormal import StandardNormal


def test_StandardNormal_smoke():
    StandardNormal((1, 2))


def test_StandardNormal_sample_mean_is_0():
    samples = StandardNormal((1,)).sample(1000)
    npt.assert_almost_equal(samples.mean(), 0.0, decimal=1)


def test_StandardNormal_sample_std_is_1():
    samples = StandardNormal((1,)).sample(1000)
    npt.assert_almost_equal(samples.std(), 1.0, decimal=1)


def test_log_prob():
    def log_prob(x):
        return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)

    norm = StandardNormal((1,))

    x = np.linspace(-10, 10, 50)

    npt.assert_almost_equal(norm.log_prob(x.reshape((-1, 1))), log_prob(x))


def test_mean():
    norm = StandardNormal((1,))
    npt.assert_almost_equal(norm.mean(), 0.0)
