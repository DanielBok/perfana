"""
Relative modules contains functions that are used to compare an asset class or portfolio relative
to a benchmark. Whilst returns have the same sort of functions, they are more specific and are
thus not grouped here.
"""

from typing import Dict, Union

from perfana.conversions import to_time_series
from perfana.types import TimeSeriesData
from .utils import days_in_duration

__all__ = ['relative_price_index']


def relative_price_index(portfolio: TimeSeriesData,
                         benchmark: TimeSeriesData,
                         duration: Union[str, int] = 'monthly',
                         *,
                         is_returns=False) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
    """
    Computes the relative price index through time. The data is assumed to be daily. If the
    benchmark is a single series, a single TimeSeriesData will be returned. Otherwise,
    a dictionary of TimeSeries will be returned where the keys are each individual benchmark

    Notes
    -----
    The relative price index at a particular time :math:`t` for an asset :math:`a` against its
    benchmark :math:`b` is given by

    .. math::
        RP_{a, t} = r_{a, t - d} - r_{b, t - d}

    where `d` is the duration. For example, if the duration is 'monthly', :math:`d` will be
    22 days.

    Parameters
    ----------
    portfolio
        The portfolio values vector or matrix

    benchmark
        The benchmark values vector or matrix

    duration
        Duration to calculate the relative price index with. Either a string or positive integer value
        can be specified. Supported string values are 'day', 'week', 'month', 'quarter', 'semi-annual'
        and 'year'

    is_returns
        Set this to true if the portfolio and benchmark values are in "returns" instead of raw values
        (i.e. prices or raw index value)

    Returns
    -------
    TimeSeriesData or dict of TimeSeriesData:
        A DataFrame of the relative price index between the assets in the portfolio against the benchmark
        If multiple series are included in the benchmark, returns a dictionary where the keys are the
        benchmarks' name and the values are the relative price index of the portfolio against that
        particular benchmark

    Examples
    --------
    >>> from perfana.datasets import load_etf
    >>> from perfana.core import relative_price_index

    >>> etf = load_etf().dropna()
    >>> returns = etf.iloc[:, 1:]
    >>> benchmark = etf.iloc[:, 0]
    >>> relative_price_index(returns, benchmark, 'monthly').head()
    """

    def derive_rolling_returns(values):
        values = to_time_series(values)
        if not is_returns:
            values = values.pct_change() + 1

        return values.rolling(days_in_duration(duration)).apply(lambda x: x.prod())

    r = derive_rolling_returns(portfolio)
    if hasattr(benchmark, 'columns'):
        return {col: r.subtract(derive_rolling_returns(benchmark[col]), axis='rows').dropna()
                for col in benchmark.columns}
    else:
        return r.subtract(derive_rolling_returns(benchmark), axis='rows').dropna()
