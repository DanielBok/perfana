from collections import abc
from typing import Optional, Union

import numpy as np
import pandas as pd

from perfana.types import DateTimes, TimeSeriesData, Vector

__all__ = ['to_prices', 'to_returns', 'to_time_series']


def to_prices(returns: TimeSeriesData, start: Union[Vector, float, int] = 100, log: bool = False) -> TimeSeriesData:
    """
    Calculates the price index given a series of returns

    Parameters
    ----------
    returns
        Price series

    start
        The starting prices

    log
        If series is a log returns series, set this to True

    Returns
    -------
    TimeSeriesData
        Price index series
    """
    returns = to_time_series(returns)

    if isinstance(start, abc.Iterable):
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("<returns> data must be a DataFrame if <start> is a vector")

        start = np.asarray(start)
        if start.size == 1:
            start = float(start)
        elif len(start) != len(returns.columns):
            raise ValueError('<start> vector length not equal to the number of columns in <returns> data')

    returns.replace(np.nan, 0, inplace=True)
    if log:
        return np.exp(returns.cumsum()) * start
    else:
        return (returns + 1).cumprod() * start


def to_returns(prices: TimeSeriesData, log: bool = False) -> TimeSeriesData:
    r"""
    Calculates the returns of a price series

    Parameters
    ----------
    prices
        price series
    log
        If True, calculate the log returns. :math:`\log{\frac{p1}{p0}}`
        Otherwise, calculates the arithmetic returns :math:`\frac{p1}{p0} - 1`.

    Returns
    -------
    TimeSeriesData
        Returns series
    """
    prices = to_time_series(prices)

    if log:
        return np.log(prices / prices.shift(1))
    else:
        return prices / prices.shift(1) - 1


def to_time_series(data, dates: Optional[DateTimes] = None, fail_policy='ignore') -> TimeSeriesData:
    """
    Casts the input data to a time series DataFrame or Series

    Parameters
    ----------
    data
        Input data. If it is 1 dimensional, will be cast into a Series object. If it
        2-dimensional, will be cast to a DataFrame. If data is already a DataFrame and
        has a column named 'date' (case insensitive), that column will
        become the index of the data. All the remaining data must be floats

    dates
        Sets the (datetime) index of the Series or DataFrame. If list of strings, each string
        must be convertible to a datetime object.

    fail_policy
        Policy to use when no dates can be set as index. Use one of [ignore, raise].
        If ignore, function will continue as is. If raise, an error will be raised
        when no dates can be set.

    Returns
    -------
    TimeSeriesData
        a time series Series or DataFrame
    """
    if not isinstance(data, abc.Iterable):
        raise ValueError('data must be a collection of data. Either a vector (Series) or a matrix (DataFrame)')

    dim = np.ndim(data)
    if dim == 1:
        data = pd.Series(data)
    elif dim == 2:
        data = pd.DataFrame(data)
    else:
        raise ValueError('data can only be 1 or 2 dimensional')

    fail_policy = fail_policy.lower()
    if fail_policy not in ('ignore', 'raise'):
        raise ValueError(f"Unknown <fail_policy>: {fail_policy}. Use one of [ignore, raise]")

    if dates is not None:
        dates = pd.to_datetime(dates).rename(None)
        data.index = dates
    elif dates is None and isinstance(data, pd.DataFrame):
        for col in data.columns:
            if str(col).lower() == 'date':
                data = data.copy()
                data.index = pd.to_datetime(data.pop(col)).rename(None)
                break
    elif fail_policy == 'raise':
        raise ValueError(f"unable to add date to data index")

    return data.astype(float)
