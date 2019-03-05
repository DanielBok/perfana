from typing import Optional

import pandas as pd

from ppa.conversions import to_time_series
from ppa.types import TimeSeriesData


def infer_scale(data: TimeSeriesData, default_freq: Optional[str] = None) -> int:
    """
    Infers the scale of the time series.

    If data is a pandas DataFrame and a column named 'Date' exists, column will be removed. In turn, the same column
    will be cast as 'datetime' object and set as the index. In any case, if data is a pandas DataFrame or Series,
    the index will be cast as a datetime object.

    :param data: DataFrame, Series, ndarray, iterable
        time series data object
    :param default_freq: str, optional
        default frequency to fall back on if unable to infer data
    :return: int
        scale of data
    """

    if not isinstance(data, (pd.DataFrame, pd.Series)):
        if default_freq is None:
            raise ValueError('<freq> cannot be None when time series data does not have date index')
        return freq_to_scale(default_freq)

    try:
        data = to_time_series(data)
    except (ValueError, TypeError):
        raise ValueError('could not cast data as time series')

    # try reading index and convert to dates. works if index is string dates
    if not data.index.is_all_dates:
        try:
            data.index = pd.to_datetime(data.index).rename(None)
        except TypeError:
            # couldn't cast
            if default_freq is not None:
                return freq_to_scale(default_freq)

    # custom inference
    n = len(data)
    time_delta = (data.index[1:] - data.index[:-1]).value_counts()

    delta = {
        'day': sum(time_delta.get(f'{day} days', 0) for day in range(1, 6)),  # 1 to 5 days - daily data
        'week': time_delta.get('7 days', 0),
        'month': sum(time_delta.get(f'{day} days', 0) for day in (28, 29, 30, 31)),
        'quarter': sum(time_delta.get(f'{day} days', 0) for day in range(89, 93)),
        'semi-annual': sum(time_delta.get(f'{day} days', 0) for day in range(180, 185)),  # 180-184 days - semi-annual
        'year': sum(time_delta.get(f'{day} days', 0) for day in range(360, 367))  # 360-366 - considered annual
    }

    for freq, count in delta.items():
        # if over 85% of the data belongs to frequency category, select frequency
        if count > 0.85 * n:
            return freq_to_scale(freq)

    raise TypeError('could not infer time series index')


def freq_to_scale(freq: str):
    """Converts frequency to scale"""
    freq = freq.lower()
    if freq in ('d', 'day', 'daily'):
        return 252
    elif freq in ('w', 'week', 'weekly'):
        return 52
    elif freq in ('m', 'month', 'monthly'):
        return 12
    elif freq in ('s', 'semi-annual'):
        return 6
    elif freq in ('q', 'quarter', 'quarterly'):
        return 4
    elif freq in ('y', 'year', 'yearly', 'annual'):
        return 1
    else:
        raise ValueError(f"Unknown frequency: {freq}")
