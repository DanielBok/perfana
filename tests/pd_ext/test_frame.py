import pytest
from numpy.testing import assert_array_almost_equal


def test_to_time_series(etf_raw):
    # pass in nothing
    ts = etf_raw.ppa.to_time_series()
    assert len(ts) == len(etf_raw)
    assert len(ts.columns) == len(etf_raw.columns) - 1

    # pass in string name
    ts = etf_raw.ppa.to_time_series('Date')
    assert len(ts) == len(etf_raw)
    assert len(ts.columns) == len(etf_raw.columns) - 1

    # pass in date list
    df = etf_raw.copy()
    dates = df.pop('Date')
    ts = df.ppa.to_time_series(dates)
    assert len(ts) == len(etf_raw)
    assert len(ts.columns) == len(etf_raw.columns) - 1


def test_to_time_series_raises_error(etf_raw):
    with pytest.raises(KeyError):
        etf_raw.ppa.to_time_series('NO_SUCH_COLUMN')

    ts = etf_raw.copy()
    ts['A'] = 'A'
    with pytest.raises(ValueError):
        ts.ppa.to_time_series('A')


def test_to_returns_and_prices(etf, expected_returns):
    ret = etf.ppa.to_returns()
    assert_array_almost_equal(ret, expected_returns)

    prices = ret.ppa.to_prices(etf.iloc[0, :])
    assert_array_almost_equal(prices, etf)
