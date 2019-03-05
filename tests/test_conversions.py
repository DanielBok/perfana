from os import path

from numpy.testing import assert_array_almost_equal

from ppa.conversions import to_prices, to_returns, to_time_series

DATA_DIR = path.join(path.dirname(__file__), 'data')


def test_to_returns_and_prices(etf_raw, expected_returns):
    ts = to_time_series(etf_raw)

    ret = to_returns(ts)

    assert_array_almost_equal(ret, expected_returns)

    prices = to_prices(ret, start=ts.iloc[0, :])
    assert_array_almost_equal(prices, ts)


def test_to_log_returns_and_prices(etf_raw, expected_log_returns):
    ts = to_time_series(etf_raw)

    log_ret = to_returns(ts, log=True)

    assert_array_almost_equal(log_ret, expected_log_returns)

    prices = to_prices(log_ret, start=ts.iloc[0, :], log=True)
    assert_array_almost_equal(prices, ts)
