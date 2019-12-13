import pytest
from pandas.testing import assert_frame_equal

from perfana.core import relative_price_index
from perfana.core.utils import days_in_duration
from perfana.datasets import load_etf


@pytest.fixture('module')
def etf():
    return load_etf().dropna()


@pytest.mark.parametrize('duration', ['d', 'w', 'm', 'q', 's', 'y'])
def test_relative_price_index(duration, etf):
    bmk = etf.iloc[:, 0]
    portfolio = etf.iloc[:, 1:]

    def derive_expected():
        days = days_in_duration(duration)
        p = (portfolio.pct_change() + 1).rolling(days).apply(lambda x: x.prod())
        b = (bmk.pct_change() + 1).rolling(days).apply(lambda x: x.prod())

        return p.subtract(b, axis='rows').dropna()

    actual = relative_price_index(portfolio, bmk, duration)
    expected = derive_expected()

    assert_frame_equal(actual, expected)


@pytest.mark.parametrize('duration', ['d', 'w', 'm', 'q', 's', 'y'])
def test_relative_price_index_multiple_benchmark(duration, etf):
    bmk = etf.iloc[:, :2]
    portfolio = etf.iloc[:, 2:]

    def derive_expected(bmk_df):
        days = days_in_duration(duration)
        p = (portfolio.pct_change() + 1).rolling(days).apply(lambda x: x.prod())
        b = (bmk_df.pct_change() + 1).rolling(days).apply(lambda x: x.prod())

        return p.subtract(b, axis='rows').dropna()

    actual = relative_price_index(portfolio, bmk, duration)
    assert isinstance(actual, dict)
    assert all(k in bmk.columns for k in actual.keys())

    for col, _bmk in bmk.iteritems():
        expected = derive_expected(_bmk)
        assert_frame_equal(actual[col], expected)
