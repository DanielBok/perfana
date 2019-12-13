import pytest
from pandas.testing import assert_frame_equal

from perfana.core import relative_price_index
from perfana.core.utils import days_in_duration
from perfana.datasets import load_etf


@pytest.fixture('module')
def bmk():
    return load_etf().dropna().iloc[:, 0]


@pytest.fixture('module')
def portfolio():
    return load_etf().dropna().iloc[:, 1:]


@pytest.mark.parametrize('duration', ['d', 'w', 'm', 'q', 's', 'y'])
def test_relative_price_index(duration, portfolio, bmk):
    def derive_expected():
        days = days_in_duration(duration)
        p = (portfolio.pct_change() + 1).rolling(days).apply(lambda x: x.prod())
        b = (bmk.pct_change() + 1).rolling(days).apply(lambda x: x.prod())

        return p.subtract(b, axis='rows').dropna()

    actual = relative_price_index(portfolio, bmk, duration)
    expected = derive_expected()

    assert_frame_equal(actual, expected)
