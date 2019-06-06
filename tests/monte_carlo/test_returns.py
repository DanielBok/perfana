from numpy.testing import assert_almost_equal

from perfana.monte_carlo import annualized_returns_m


def test_annualized_returns_m(cube_a, weights, expected):
    """Test Geometric Returns on cube data is correct"""
    e = expected["PP average return (GR), 20Y"]
    gr = annualized_returns_m(cube_a, weights, freq='quarterly', geometric=True, rebalance=True)

    assert_almost_equal(e, gr, 4)
