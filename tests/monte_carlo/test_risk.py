from numpy.testing import assert_almost_equal

from perfana.monte_carlo import beta_m, correlation_m, diversification_m, tracking_error_m, vol_attr, volatility_m


def test_volatility_m(cube_a, weights, freq, expected):
    e = expected["PP vol"]
    vol = volatility_m(cube_a, weights, freq)

    assert_almost_equal(e, vol, 4)


def test_tracking_error_m(cube, weights, bmk_weights, freq, expected):
    e = expected["PP-RP TE"]
    te = tracking_error_m(cube, weights, bmk_weights, freq)

    assert_almost_equal(e, te, 4)


def test_beta_dmeq(cube_a, weights, freq, expected):
    e = expected["PP beta to DMEQ"]
    beta = beta_m(cube_a, weights, freq, aid=0)

    assert_almost_equal(e, beta, 4)


def test_correDMEQ(cube_a, weights, freq, expected):
    e = expected["PP correl to DMEQ"]
    corr = correlation_m(cube_a, weights, freq, aid=0)

    assert_almost_equal(e, corr, 4)


def test_diversification(cube_a, weights, expected):
    e = expected["PP diversification ratio"]
    div_ratio = diversification_m(cube_a, weights, 'quarterly')

    assert_almost_equal(e, div_ratio, 4)


def test_vol_attr(cube_a, weights, freq, expected, a_order):
    vol_rel, vol_abs = vol_attr(cube_a, weights, freq)

    e_rel = [expected[f"PP vol attribution - {a}"][0] for a in a_order]
    e_abs = [expected[f"PP vol attribution - {a}"][1] for a in a_order]

    assert_almost_equal(vol_rel, e_rel, 4)
    assert_almost_equal(vol_abs, e_abs, 4)
