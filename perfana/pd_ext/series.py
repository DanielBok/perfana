from typing import Union

import pandas as pd
from pandas.api.extensions import register_series_accessor

from perfana.conversions import *
from perfana.core.utils import infer_frequency
from perfana.types import DateTimes


@register_series_accessor('ppa')
class PpaSeriesAccessor:
    def __init__(self, pd_obj: pd.Series):
        self._obj = pd_obj

    def to_prices(self, start: Union[float, int] = 100., log: bool = False) -> pd.Series:
        """
        Creates a price series from a returns series

        Parameters
        ----------
        start
            the starting prices

        log
            If data is a log returns data, set this to True

        Returns
        -------
        Series
            Asset price series
        """
        return to_prices(self._obj, start, log)

    def to_returns(self, log: bool = False) -> pd.Series:
        """
        Creates a returns series from price series

        Parameters
        ----------
        log
            If True, calculate the log returns. Otherwise, calculates the arithmetic returns

        Returns
        -------
        Series
            Asset returns series
        """
        return to_returns(self._obj, log)

    def to_time_series(self, dates: DateTimes) -> pd.Series:
        """
        Creates a time series.

        Parameters
        ----------
        dates
            A list of dates

        Returns
        -------
        Series
            A time series
        """
        return to_time_series(self._obj, dates)

    @property
    def frequency(self):
        """Frequency (periodicity) of the data"""
        return infer_frequency(self._obj, 'ignore')
