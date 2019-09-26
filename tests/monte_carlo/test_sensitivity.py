import pytest
from numpy.testing import assert_almost_equal

from perfana.monte_carlo import sensitivity_m
from perfana.monte_carlo.sensitivity import _setup


@pytest.mark.parametrize("shock", [-0.05, 0.05])
@pytest.mark.parametrize("geometric", [True, False])
@pytest.mark.parametrize("rebalance", [True, False])
@pytest.mark.parametrize("invert, alpha", [(True, 0.95), (False, 0.05)])
@pytest.mark.parametrize("leveraged", [True, False])
@pytest.mark.parametrize("dist", [True, False])
def test_sensitivity_m_coverage(cube_a, weights, freq, shock, geometric, rebalance, invert, alpha, leveraged, dist):
    # This test does not value correctness of numbers, presently, it just tests that there are no errors
    # However, given that `test_sensitivity_setup` is correct, this should be correct as well
    sensitivity_m(cube_a, weights, freq, shock, geometric, rebalance,
                  alpha=alpha, invert=invert, leveraged=leveraged, distribute=dist)


@pytest.mark.parametrize("weights", [[0.99, 0.5, 0.01]])
@pytest.mark.parametrize("shock, leveraged, distribute, expected", [
    (0.05, True, True, [[1.04, 0.45098039, 0.00901961],
                        [0.9405, 0.55, 0.0095],
                        [0.95677852, 0.48322148, 0.06]]),
    (0.05, True, False, [[1.04, 0.5, 0.01],
                         [0.99, 0.55, 0.01],
                         [0.99, 0.5, 0.06]]),
    (0.05, False, True, [[1., 0.49019608, 0.00980392],
                         [0.9405, 0.55, 0.0095],
                         [0.95677852, 0.48322148, 0.06]]),
    (0.05, False, False, [[1., 0.5, 0.01],
                          [0.99, 0.55, 0.01],
                          [0.99, 0.5, 0.06]]),
    (-0.05, True, True, [[0.94, 0.54901961, 0.01098039],
                         [1.0395, 0.45, 0.0105],
                         [1.02322148, 0.51677852, -0.04]]),
    (-0.05, True, False, [[0.94, 0.5, 0.01],
                          [0.99, 0.45, 0.01],
                          [0.99, 0.5, -0.04]]),
    (-0.05, False, True, [[0.94, 0.54901961, 0.01098039],
                          [1.0395, 0.45, 0.0105],
                          [0.9966443, 0.5033557, 0.]]),
    (-0.05, False, False, [[0.94, 0.5, 0.01],
                           [0.99, 0.45, 0.01],
                           [0.99, 0.5, 0.]]),
])
def test_sensitivity_setup(weights, shock, leveraged, distribute, expected):
    res = _setup(weights, shock, leveraged, distribute)
    assert_almost_equal(res, expected, 6)
