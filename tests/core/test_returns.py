import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from ppa.core.returns import *
from ppa.exceptions import TimeIndexError, TimeIndexMismatchError


@pytest.fixture
def retf(etf):
    return etf.ppa.to_returns()


@pytest.mark.parametrize("col_a, col_b, expected", [
    (['VBK', 'BND', 'VTI', 'VWO'], ['VBK', 'BND', 'VTI', 'VWO'],
     [[0, -0.0553853450601067, -0.0104068323363791, -0.0639390330747389],
      [0.0553853450601067, 0, 0.0449785127237277, -0.00855368801463219],
      [0.0104068323363791, -0.0449785127237277, 0, -0.0535322007383598],
      [0.0639390330747389, 0.00855368801463219, 0.0535322007383598, 0]]),
    ('VBK', 'BND', [[0.0553853450601067]])
])
def test_active_premium(retf, col_a, col_b, expected):
    ra, rb = retf[col_a], retf[col_b]

    assert_array_almost_equal(active_premium(ra, rb), expected)

    assert_array_almost_equal(active_premium(ra.values, rb.values, 'daily'), expected)


@pytest.mark.parametrize("geometric, expected", [
    (True, [0.0916094100561804, 0.0362240649960737, 0.0812025777198013, 0.0276703769814415]),
    (False, [0.115486372414379, 0.0365847615014861, 0.0978013892823243, 0.0713152592864056])
])
def test_annualized_returns(retf, geometric, expected):
    assert_array_almost_equal(annualized_returns(retf, geometric=geometric), expected)


@pytest.mark.parametrize("geometric, expected", [
    (True, [0.00962523818462069, -0.0416004490283264, 0, -0.049511721338342]),
    (False, [0.0104068323363791, -0.0449785127237277, 0, -0.0535322007383598])
])
def test_excess_returns(retf, geometric, expected):
    bmk = retf['VTI'].copy()

    assert_array_almost_equal(excess_returns(retf, bmk, geometric=geometric), expected)

    # test that if at least on of them has a date, everything will be okay
    assert_array_almost_equal(excess_returns(retf.values, bmk, geometric=geometric), expected)
    assert_array_almost_equal(excess_returns(retf, bmk.values, geometric=geometric), expected)

    # test that if both are not time series, will be okay if freq is supplied
    assert_array_almost_equal(excess_returns(retf.values, bmk.values, 'daily', geometric), expected)


def test_excess_returns_raises_errors(retf):
    wrong_shape_bmk = retf.loc[:, ('BND', 'VTI')].copy()

    with pytest.raises(ValueError, match='The shapes of the asset and benchmark returns do not match!'):
        excess_returns(retf, wrong_shape_bmk)

    bmk = retf['VTI'].copy()
    arr_retf = retf.values
    arr_bmk = bmk.values
    with pytest.raises(TimeIndexError):
        excess_returns(arr_retf, arr_bmk)

    with pytest.raises(TimeIndexMismatchError):
        freq = 'W' if retf.ppa.frequency == 'daily' else 'D'
        bmk.index = pd.date_range('1680-01-01', periods=len(bmk), freq=freq)
        excess_returns(retf, bmk)


def test_returns_relative(retf, expected_rel_returns):
    rr = relative_returns(retf, retf)
    assert_array_almost_equal(rr, expected_rel_returns)

    # both series and uninformative names
    ra, rb = retf['VTI'].rename(0), retf['BND'].rename(1)
    assert_array_almost_equal(relative_returns(ra, rb), expected_rel_returns['VTI/BND'])

    ra, rb = retf['VTI'].rename(2), retf['BND'].rename(1)
    assert_array_almost_equal(relative_returns(ra, rb), expected_rel_returns['VTI/BND'])
