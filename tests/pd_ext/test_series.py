import pytest
from numpy.testing import assert_array_almost_equal

ASSET = 'VTI'


@pytest.fixture()
def series(etf_raw):
    dates = etf_raw.Date
    series = etf_raw[ASSET]
    return series.pa.to_time_series(dates)


def test_to_time_series(etf_raw, series):
    assert len(series) == len(etf_raw)


def test_to_returns_and_prices(series, etf, expected_returns):
    ret = series.pa.to_returns()
    assert_array_almost_equal(ret, expected_returns[ASSET])

    prices = ret.pa.to_prices(etf.iloc[0][ASSET])
    assert_array_almost_equal(prices, series)
