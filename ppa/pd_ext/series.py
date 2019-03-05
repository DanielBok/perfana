from typing import Optional, Union

import pandas as pd
from pandas.api.extensions import register_series_accessor

from ppa.conversions import *
from ppa.types import DateTimes


@register_series_accessor('ppa')
class PpaSeriesAccessor:
    def __init__(self, pd_obj: pd.Series):
        self._obj = pd_obj

    def to_prices(self, start: Union[float, int] = 100., log=False):
        """
        Creates a price series from a returns series

        :param start: iterable numeric or scalar numeric
            the starting prices
        :param log: boolean, default False
            If data is a log returns data, set this to True
        :return: Series
            asset prices series
        """
        return to_prices(self._obj, start, log)

    def to_returns(self, log=False):
        """
        Creates a returns series from price series

        :param log: boolean, default False
            If True, calculate the log returns. Otherwise, calculates the arithmetic returns
        :return: Series
            asset returns series
        """
        return to_returns(self._obj, log)

    def to_time_series(self, dates: Optional[DateTimes] = None):
        """
        Creates a time series.

        :param dates: str, iterable dates, optional
            A list of dates
        :return: Series
            a time series
        """
        if dates is None:
            raise ValueError('<dates> cannot be none!')

        return to_time_series(self._obj, dates)