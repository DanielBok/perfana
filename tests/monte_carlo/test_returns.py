from numpy.testing import assert_almost_equal

from perfana.monte_carlo import (annualized_bmk_quantile_returns_m, annualized_bmk_returns_m,
                                 annualized_quantile_returns_m, annualized_returns_m, returns_attr)


def test_annualized_returns_m(cube_a, weights, expected):
    """Test Geometric Returns on cube data is correct"""
    e = expected["PP average return (GR), 20Y"]
    gr = annualized_returns_m(cube_a, weights, freq='quarterly', geometric=True, rebalance=True)

    assert_almost_equal(e, gr, 4)


def test_annualized_bmk_returns_m(cube_a, weights, cube_b, ref_weights, expected):
    """Test the geometric benchmark returns function"""
    e = expected["PP-RP average return (GR), 20Y"]
    gr = annualized_bmk_returns_m(cube_a, weights, cube_b, ref_weights,
                                  freq='quarterly', geometric=True, rebalance=True)

    assert_almost_equal(e, gr, 4)


def test_annualized_quantile_return_m(cube_a, weights, expected):
    """Test the annualized quantile returns function"""
    e = expected["PP return 75th, 20Y"]
    qr = annualized_quantile_returns_m(cube_a, weights, quantile=0.75,
                                       freq='quarterly', geometric=True, rebalance=True)

    assert_almost_equal(e, qr, 4)


def test_annualized_bmk_quantile_returns_m(cube_a, weights, cube_b, ref_weights, expected):
    """Test the annualized benchmark quantile returns function"""
    e = expected["PP-RP return 75th, 20Y"]
    gr = annualized_bmk_quantile_returns_m(cube_a, weights, cube_b, ref_weights, quantile=0.75,
                                           freq='quarterly', geometric=True, rebalance=True)

    assert_almost_equal(e, gr, 4)


def test_returns_attr(cube_a, weights, expected, a_order):
    """Test the returns attribution function"""
    marginal, percentage = returns_attr(cube_a, weights, 'quarterly')

    e_p = [expected[f"PP return attribution - {a}"][0] for a in a_order]
    e_m = [expected[f"PP return attribution - {a}"][1] for a in a_order]

    assert_almost_equal(e_p, percentage, 4)
    assert_almost_equal(e_m, marginal, 4)
