"""
Relative modules contains functions that are used to compare an asset class or portfolio relative
to a benchmark. Whilst returns have the same sort of functions, they are more specific and are
thus not grouped here.
"""

from typing import Dict, Union

import pandas as pd

from perfana.conversions import to_time_series
from perfana.types import TimeSeriesData
from .utils import days_in_duration

__all__ = ['correlation_measure', 'relative_price_index']


def correlation_measure(portfolio: TimeSeriesData,
                        benchmark: TimeSeriesData,
                        duration: Union[str, int] = 'monthly',
                        *,
                        is_returns=False,
                        date_as_index=True) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
    """
    Computes the correlation measure through time. The data is assumed to be daily. If the
    benchmark is a single series, a single TimeSeriesData will be returned. Otherwise,
    a dictionary of TimeSeries will be returned where the keys are each individual benchmark

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

    date_as_index
        If true, returns the date as the dataframe's index. Otherwise, the date is placed as a column
        in the dataframe

    Returns
    -------
    TimeSeriesData or dict of TimeSeriesData:
        A DataFrame of the correlation measure between the assets in the portfolio against the benchmark
        If multiple series are included in the benchmark, returns a dictionary where the keys are the
        benchmarks' name and the values are the correlation measure of the portfolio against that
        particular benchmark

    Examples
    --------
    >>> from perfana.datasets import load_etf
    >>> from perfana.core import correlation_measure
    >>> etf = load_etf().dropna()
    >>> returns = etf.iloc[:, 1:]
    >>> benchmark = etf.iloc[:, 0]
    >>> correlation_measure(returns, benchmark, 'monthly').head()
                     BND       VTI       VWO
    Date
    2007-05-10 -0.384576  0.890783  0.846000
    2007-05-11 -0.525299  0.911693  0.857288
    2007-05-14 -0.482180  0.912002  0.855114
    2007-05-15 -0.439073  0.913992  0.842561
    2007-05-16 -0.487110  0.899859  0.837781
    """

    def derive_returns(values):
        values = to_time_series(values)
        return values.pct_change() if not is_returns else values

    portfolio = derive_returns(portfolio)
    benchmark = derive_returns(benchmark)
    days = days_in_duration(duration)

    if hasattr(benchmark, 'columns'):
        return {col: _format_data_frame(portfolio.rolling(days).corr(benchmark[col]), date_as_index)
                for col in benchmark.columns}
    else:
        return _format_data_frame(portfolio.rolling(days).corr(benchmark), date_as_index)


def relative_price_index(portfolio: TimeSeriesData,
                         benchmark: TimeSeriesData,
                         duration: Union[str, int] = 'monthly',
                         *,
                         is_returns=False,
                         date_as_index=True) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
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

    date_as_index
        If true, returns the date as the dataframe's index. Otherwise, the date is placed as a column
        in the dataframe

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
                     BND       VTI       VWO
    Date
    2007-05-10 -0.016000  0.009433  0.000458
    2007-05-11 -0.031772  0.008626  0.013009
    2007-05-14 -0.016945  0.014056  0.008658
    2007-05-15 -0.002772  0.020824  0.018758
    2007-05-16  0.002791  0.025402  0.028448
    """

    def derive_rolling_returns(values):
        values = to_time_series(values)
        if not is_returns:
            values = values.pct_change() + 1

        return values.rolling(days_in_duration(duration)).apply(lambda x: x.prod(), raw=False)

    r = derive_rolling_returns(portfolio)
    if hasattr(benchmark, 'columns'):
        return {col: _format_data_frame(r.subtract(derive_rolling_returns(benchmark[col]), axis='rows'), date_as_index)
                for col in benchmark.columns}
    else:
        return _format_data_frame(r.subtract(derive_rolling_returns(benchmark), axis='rows'), date_as_index)


def _format_data_frame(df: pd.DataFrame, date_as_index: bool):
    df = df.dropna(how='all')
    if not date_as_index:
        df = df.reset_index()

    return df
