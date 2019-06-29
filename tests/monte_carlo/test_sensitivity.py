import pytest

from perfana.monte_carlo import sensitivity_m


@pytest.mark.parametrize("shock", [-0.05, 0.05])
@pytest.mark.parametrize("geometric", [True, False])
@pytest.mark.parametrize("rebalance", [True, False])
@pytest.mark.parametrize("invert, alpha", [(True, 0.95), (False, 0.05)])
@pytest.mark.parametrize("leveraged", [True, False])
@pytest.mark.parametrize("dist", [True, False])
def test_sensitivity_m_coverage(cube_a, weights, freq, shock, geometric, rebalance, invert, alpha, leveraged, dist):
    # TODO verify correctness of number
    # This test does not value correctness of numbers, presently, it just tests that there are no errors
    sensitivity_m(cube_a, weights, freq, shock, geometric, rebalance,
                  alpha=alpha, invert=invert, leveraged=leveraged, distribute=dist)
