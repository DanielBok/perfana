import pytest
from numpy.testing import assert_almost_equal

from perfana.monte_carlo import (beta_m, correlation_m, cvar_attr, cvar_div_ratio, cvar_m, diversification_m,
                                 drawdown_m, prob_loss, prob_under_perf, tail_loss, tracking_error_m, vol_attr,
                                 volatility_m)


@pytest.fixture
def cvar_cube(cube_a):
    return cube_a[:12]


def test_beta_dmeq(cube_a, weights, freq, expected):
    e = expected["PP beta to DMEQ"]
    beta = beta_m(cube_a, weights, freq, aid=0)

    assert_almost_equal(e, beta, 4)


def test_corr_dmeq(cube_a, weights, freq, expected):
    e = expected["PP correl to DMEQ"]
    corr = correlation_m(cube_a, weights, freq, aid=0)

    assert_almost_equal(e, corr, 4)


def test_cvar_attr(cvar_cube, weights, expected, a_order):
    # TODO fix this illogical step in the test
    cvar_rel = cvar_attr(cvar_cube, weights, rebalance=False).percentage
    cvar = cvar_m(cvar_cube, weights, rebalance=True)

    e_rel = [expected[f"PP CVaR attribution - {a}"][0] for a in a_order]
    e_abs = [expected[f"PP CVaR attribution - {a}"][1] for a in a_order]

    assert_almost_equal(cvar_rel, e_rel, 4)
    assert_almost_equal(cvar_rel * cvar, e_abs, 4)


def test_cvar_m(cvar_cube, weights, expected):
    e_cvar = expected["PP 95% CVaR, 3Y"]
    cvar = cvar_m(cvar_cube, weights)
    assert_almost_equal(cvar, e_cvar, 4)


def test_cvar_div_ratio(cvar_cube, weights, expected):
    e_div_ratio = expected["PP tail diversification ratio"]
    div_ratio = cvar_div_ratio(cvar_cube, weights)

    assert_almost_equal(div_ratio, e_div_ratio, 4)


def test_max_drawdown(cube_a, weights, expected):
    """Test expected max drawdown"""
    average, _ = drawdown_m(cube_a, weights)
    e = expected["PP average max drawdown"]

    assert_almost_equal(average, e, 4)


def test_diversification(cube_a, weights, expected):
    e = expected["PP diversification ratio"]
    div_ratio = diversification_m(cube_a, weights, 'quarterly')

    assert_almost_equal(e, div_ratio, 4)


@pytest.mark.parametrize("year", [10, 20])
@pytest.mark.parametrize("terminal", [True, False])
def test_prob_under_perf(cube, weights, bmk_weights, year, terminal, expected):
    e = expected[f"Prob of PP underperform RP, {year}Y"]

    data = cube[:4 * year]
    prob = prob_under_perf(data, weights, bmk_weights, terminal=terminal)

    assert_almost_equal(prob, e, 4)


@pytest.mark.parametrize("year", [10, 20])
@pytest.mark.parametrize("terminal", [True, False])
def test_prob_loss(cube, weights, year, terminal, expected):
    e = expected[f"Prob of PP less than 0, {year}Y"]

    data = cube[:4 * year]
    prob = prob_loss(data, weights, terminal=terminal)

    assert_almost_equal(prob, e, 4)


def test_tail_loss(cvar_cube, weights, freq, expected):
    e_prob = expected["PP prob of loss great than -30%"]
    e_loss = expected["PP expected loss great than -30%"]

    prob, loss = tail_loss(cvar_cube, weights, threshold=-0.3)

    assert_almost_equal(prob, e_prob, 4)
    assert_almost_equal(loss, e_loss, 4)


def test_tracking_error_m(cube, weights, bmk_weights, freq, expected):
    e = expected["PP-RP TE"]
    te = tracking_error_m(cube, weights, bmk_weights, freq)

    assert_almost_equal(e, te, 4)


def test_vol_attr(cube_a, weights, freq, expected, a_order):
    vol_rel, vol_abs = vol_attr(cube_a, weights, freq)

    e_rel = [expected[f"PP vol attribution - {a}"][0] for a in a_order]
    e_abs = [expected[f"PP vol attribution - {a}"][1] for a in a_order]

    assert_almost_equal(vol_rel, e_rel, 4)
    assert_almost_equal(vol_abs, e_abs, 4)


def test_volatility_m(cube_a, weights, freq, expected):
    e = expected["PP vol"]
    vol = volatility_m(cube_a, weights, freq)

    assert_almost_equal(e, vol, 4)
