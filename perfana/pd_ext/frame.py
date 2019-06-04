from typing import Iterable, Optional, Union

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor

from perfana.conversions import *
from perfana.core.utils import infer_frequency
from perfana.types import DateTimes, TimeSeriesData, Vector


@register_dataframe_accessor('pa')
class PaFrameAccessor:
    def __init__(self, pd_obj: pd.DataFrame):
        self._obj = pd_obj

    def to_prices(self, start: Union[Vector, float, int] = 100., log: bool = False) -> pd.DataFrame:
        """
        Creates a returns data frame to price data frame

        Parameters
        ----------
        start:
            the starting prices
        log:
            If data is a log returns data, set this to True

        Returns
        -------
        DataFrame
            Asset price data
        """
        return to_prices(self._obj, start, log)

    def to_returns(self, log: bool = False) -> TimeSeriesData:
        """
        Creates a price data frame from returns data frame

        Parameters
        ----------
        log:
            If True, calculate the log returns. Otherwise, calculates the arithmetic returns

        Returns
        -------
        DataFrame
            Asset returns data
        """
        return to_returns(self._obj, log)

    def to_time_series(self, dates: Optional[Union[str, Iterable[DateTimes]]] = None) -> pd.DataFrame:
        """
        Creates a time series data frame. If <dates> is not specified, method will attempt to look
        for a column named 'date' to make as the index. Otherwise, you can specify the date column
        name in the <dates> argument or pass in a list of dates

        Parameters
        ----------
        dates:
            If string, will attempt to look for a column in DataFrame and cast it as date. Otherwise,
            pass in a list of dates. By default, it will look for a column named 'Date' in the data frame

        Returns
        -------
        DataFrame
            a time series DataFrame
        """
        if type(dates) is str:
            col = dates
            try:
                df = self._obj.copy()
                dates = pd.to_datetime(df.pop(col))
            except KeyError:
                raise KeyError(f"Unable to find column <{col}>")
            except ValueError as e:
                raise ValueError(f"Unable to convert column <{col}> to datetime. ", e)
            return to_time_series(df, dates)

        return to_time_series(self._obj, dates)

    @property
    def frequency(self):
        """Frequency (periodicity) of the data"""
        return infer_frequency(self._obj, 'ignore')
