from numpy.testing import assert_almost_equal

from perfana.monte_carlo import sharpe_ratio_bmk_m, sharpe_ratio_m


def test_sharpe_ratio_m(cube_a, weights, freq, expected):
    """Tests the risk adjusted return ratio"""
    e = expected["PP risk adjusted return"]
    port_sr = sharpe_ratio_m(cube_a, weights, freq, geometric=True, rebalance=True)
    assert_almost_equal(e, port_sr, 4)


def test_sharpe_ratio_bmk_m(cube, weights, cube_b, bmk_weights, freq, expected):
    """Tests the risk adjusted return ratio of the portfolio against a benchmark"""
    e = expected["PP-RP IR"]
    port_vs_bmk_sr = sharpe_ratio_bmk_m(cube, weights, bmk_weights, freq, geometric=True, rebalance=True)
    assert_almost_equal(e, port_vs_bmk_sr, 4)
