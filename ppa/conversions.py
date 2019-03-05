from collections import abc
from typing import Optional, Union

import numpy as np
import pandas as pd

from ppa.types import DateTimes, TimeSeriesData, Vector

__all__ = ['to_prices', 'to_returns', 'to_time_series']


def to_prices(returns: TimeSeriesData, start: Union[Vector, float, int] = 100, log=False):
    """
    Calculates the price index given a series of returns

    :param returns: iterable data
        price series
    :param start: iterable numeric or scalar numeric
        the starting prices
    :param log: boolean, default False
        If series is a log returns series, set this to True
    :return: Series or DataFrame
        price index series
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


def to_returns(prices: TimeSeriesData, log=False):
    """
    Calculates the returns of a price series

    :param prices: iterable data
        price series
    :param log: boolean, default False
        If True, calculate the log returns. [Formula: ln(p1/p0) ]
        Otherwise, calculates the arithmetic returns [Formula: p1/p0 - 1].
    :return: Series or DataFrame
        returns series
    """
    prices = to_time_series(prices)

    if log:
        return np.log(prices / prices.shift(1))
    else:
        return prices / prices.shift(1) - 1


def to_time_series(data, dates: Optional[DateTimes] = None) -> Union[pd.Series, pd.DataFrame]:
    """
    Casts the input data to a time series DataFrame or Series

    :param data: iterable data
        Input data. If it is 1 dimensional, will be cast into a Series object. If it 2 dimensional, will be cast to
        a DataFrame. If data is already a DataFrame and has a column named 'date' (caps insensitive), that column will
        become the index of the data. All the remaining data must be floats
    :param dates: iterable date or str
        Sets the (datetime) index of the Series or DataFrame. If list of strings, each string must be convertable to
        a datetime object.
    :return: Series, DataFrame
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

    if dates is not None:
        dates = pd.to_datetime(dates).rename(None)
        data.index = dates
    elif dates is None and isinstance(data, pd.DataFrame):
        for col in data.columns:
            if col.lower() == 'date':
                data = data.copy()
                data.index = pd.to_datetime(data.pop(col)).rename(None)
                break

    return data.astype(float)
