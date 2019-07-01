import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

from perfana.core import drawdown, drawdown_summary
from perfana.datasets import load_hist


@pytest.fixture
def hist() -> pd.DataFrame:
    return load_hist().iloc[:, :7]


@pytest.fixture
def weights():
    return [0.25, 0.18, 0.24, 0.05, 0.04, 0.13, 0.11]


@pytest.mark.parametrize("rebalance, exp_min",
                         [(True, -0.4007984968456346),
                          (False, -0.5103889505008865)])
def test_drawdown(hist, weights, rebalance, exp_min):
    dd = drawdown(hist, weights, rebalance=rebalance)

    assert_almost_equal(dd.min(), exp_min)


def test_drawdown_single(hist):
    assert_almost_equal(drawdown(hist.iloc[:, 0]).min(), -0.5491340503)


def test_drawdown_summary(hist, weights):
    dd = drawdown(hist, weights)

    summary = drawdown_summary(dd, top=None)
    assert drawdown_summary(dd, top=8).shape == (8, 7)
    assert summary.shape == (35, 7)

    expected = {
        "Start": pd.to_datetime("2007-11-30"),
        "Trough": pd.to_datetime("2009-02-28"),
        "End": pd.to_datetime("2014-02-28"),
        "Drawdown": -0.400798,
        "Length": 76,
        "ToTrough": 16,
        "Recovery": 60,
    }
    for key, value in expected.items():
        v = summary.loc[0, key]
        if isinstance(value, (int, float)):
            assert_almost_equal(v, value, 6)
        else:
            assert value == v
