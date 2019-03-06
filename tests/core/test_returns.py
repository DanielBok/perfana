import pytest
from numpy.testing import assert_array_almost_equal
import numpy as np
import pandas as pd
from ppa.core.returns import *
from ppa.exceptions import TimeIndexError, TimeIndexMismatchError


@pytest.fixture
def retf(etf):
    return etf.ppa.to_returns()


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
