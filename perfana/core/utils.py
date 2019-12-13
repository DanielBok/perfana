from typing import Union

import pandas as pd

from perfana.types import TimeSeriesData

__all__ = ['days_in_duration', 'freq_to_scale', 'infer_frequency']


def days_in_duration(duration: Union[str, int]):
    """Returns the number of day in a specified duration"""
    if isinstance(duration, int):
        assert duration > 0, "duration must be a positive integer"
        return duration

    duration = duration.lower()
    if duration in ('d', 'day', 'daily'):
        return 1
    elif duration in ('w', 'week', 'weekly'):
        return 5
    elif duration in ('m', 'month', 'monthly'):
        return 22
    elif duration in ('q', 'quarter', 'quarterly'):
        return 66
    elif duration in ('s', 'semi-annual', 'semi-annually'):
        return 132
    elif duration in ('a', 'y', 'annual', 'annually', 'year', 'yearly'):
        return 264
    else:
        raise ValueError(f"Unknown period: {duration}")


def freq_to_scale(freq: str):
    """Converts frequency to scale"""
    freq = str(freq).lower()
    if freq in ('d', 'day', 'daily'):
        return 252
    elif freq in ('w', 'week', 'weekly'):
        return 52
    elif freq in ('m', 'month', 'monthly'):
        return 12
    elif freq in ('s', 'semi-annual', 'semi-annually'):
        return 6
    elif freq in ('q', 'quarter', 'quarterly'):
        return 4
    elif freq in ('a', 'y', 'annual', 'annually', 'year', 'yearly'):
        return 1
    else:
        raise ValueError(f"Unknown frequency: {freq}")


def infer_frequency(data: TimeSeriesData, fail_policy='raise'):
    """
    Infers the frequency (periodicity) of the time series

    Function works by taking the difference between the dates of each successive data point.
    It then identifies which is the most common difference, whether 2 dates differ by a day,
    a week or more and use that to determine the periodicity of the data.

    Parameters
    ----------
    data:
        time series pandas data object

    fail_policy:
        Action to do when frequency cannot be inferred. If set to 'raise', a TypeError will be raised.
        If set to 'ignore', a NoneType will be returned on failure.

    Returns
    -------
    str
        Frequency of data. If :code:`fail_policy` is ignore, returns None.

    Raises
    ------
    TypeError:
        If :code:`fail_policy` is set to 'raise' and frequency cannot be inferred
    """
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise ValueError('<data> must be a pandas DataFrame or Series')

    fail_policy = fail_policy.lower()
    if fail_policy not in ('ignore', 'raise'):
        raise ValueError(f"Unknown <fail_policy>: {fail_policy}. Use one of [ignore, raise]")

    # custom inference
    n = len(data)
    time_delta = (data.index[1:] - data.index[:-1]).value_counts()

    delta = {
        'daily': sum(time_delta.get(f'{day} days', 0) for day in range(1, 6)),  # 1 to 5 days - daily data
        'weekly': time_delta.get('7 days', 0),
        'monthly': sum(time_delta.get(f'{day} days', 0) for day in (28, 29, 30, 31)),
        'quarterly': sum(time_delta.get(f'{day} days', 0) for day in range(89, 93)),
        'semi-annually': sum(time_delta.get(f'{day} days', 0) for day in range(180, 185)),  # 180-184 days - semi-annual
        'yearly': sum(time_delta.get(f'{day} days', 0) for day in range(360, 367))  # 360-366 - considered annual
    }

    if n < 30:  # less than 30 observations, just take key with max value
        return max(delta, key=lambda k: delta[k])

    for freq, count in delta.items():
        if count > 0.85 * n:
            # if over 85% of the data belongs to frequency category, select frequency
            return freq

    if fail_policy == 'raise':
        raise TypeError('could not infer periodicity of time series index')
    else:
        return None
