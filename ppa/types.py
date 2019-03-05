from collections import abc
from typing import Iterable, Union

import numpy as np
import pandas as pd

__all__ = ['TimeSeriesData', 'Vector', 'cast_to_series']

Vector = Union[Iterable[Union[int, float]], np.ndarray, pd.Series]
TimeSeriesData = Union[pd.DataFrame, Vector]


def cast_to_series(series: Vector, allow_object=False) -> pd.Series:
    err_msg = "data series should be a univariate vector"
    if isinstance(series, pd.Series):
        return series
    elif isinstance(series, np.ndarray):
        if series.ndim != 1:
            raise ValueError(err_msg)
        return pd.Series(series)

    if isinstance(series, abc.Iterable):
        raise ValueError(err_msg)

    series = pd.Series(series)
    if not allow_object and series.dtype == 'object':
        raise ValueError('Could not cast vector to non-object type data. Check that your data is a numeric list')
    return series
