from typing import Iterable, Optional, Union

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from ppa.conversions import *
from ppa.types import DateTimes, Vector


@register_dataframe_accessor('ppa')
class PpaFrameAccessor:
    def __init__(self, pd_obj: pd.DataFrame):
        self._obj = pd_obj

    def to_prices(self, start: Union[Vector, float, int] = 100., log=False):
        """
        Creates a returns data frame to price data frame

        :param start: iterable numeric or scalar numeric
            the starting prices
        :param log: boolean, default False
            If data is a log returns data, set this to True
        :return: DataFrame
            asset prices data
        """
        return to_prices(self._obj, start, log)

    def to_returns(self, log=False):
        """
        Creates a price data frame from returns data frame

        :param log: boolean, default False
            If True, calculate the log returns. Otherwise, calculates the arithmetic returns
        :return: DataFrame
            asset returns data
        """
        return to_returns(self._obj, log)

    def to_time_series(self, dates: Optional[Union[str, Iterable[DateTimes]]] = None):
        """
        Creates a time series data frame. If <dates> is not specified, method will attempt to look for a column named
        'date' to make as the index. Otherwise, you can specify the date column name in the <dates> argument or pass in
        a list of dates

        :param dates: str, iterable dates, optional
            if string, will attempt to look for a column in DataFrame and cast it as date. Otherwise, pass in a list of
            dates. By default, it will look for a column named 'Date' in the data frame
        :return: DataFrame
            a time series DataFrame
        """
        if type(dates) is str:
            col = dates
            dates = self._obj.pop(col)
            try:
                dates = pd.to_datetime(dates)
            except ValueError as e:
                self.obj[col] = dates
                raise ValueError(f"Unable to convert column <{col}> to datetime. ", e)

        return to_time_series(self._obj, dates)
