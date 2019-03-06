from typing import Optional, Union

import pandas as pd

from ppa.conversions import to_time_series
from ppa.core.utils import freq_to_scale
from ppa.exceptions import TimeIndexError, TimeIndexMismatchError
from ppa.types import TimeSeriesData

__all__ = ['annualized_returns', 'excess_returns']


# def active_premium(ra: Vector, rb: Vector, freq='monthly'):
#     pass


def annualized_returns(r: TimeSeriesData, freq: Optional[str] = None, geometric=True) -> Union[float, pd.Series]:
    """
    Calculates the annualized returns from the data

    The formula for annualized geometric returns is formulated by raising the compound return to the number of
    periods in a year, and taking the root to the number of total observations:

        prod(1 + R)^(scale/n) - 1

    where scale is the number of observations in a year, and n is the total number of observations.

    For simple returns (geometric=FALSE), the formula is:

        mean(R)*scale

    :param r: iterable data
        numeric returns series or data frame
    :param freq: str, optional
        frequency of the data. Use one of [daily, weekly, monthly, quarterly, semi-annually, yearly]
    :param geometric: boolean, default True
        If True, calculates the geometric returns. Otherwise, calculates the arithmetic returns

    :return: float, pd.Series
        annualized returns
    """
    r = to_time_series(r).dropna()
    if freq is None:
        freq = r.ppa.frequency

    scale = freq_to_scale(freq)

    if geometric:
        return (r + 1).prod() ** (scale / len(r)) - 1
    else:  # arithmetic mean
        return r.mean() * scale


def excess_returns(ra: TimeSeriesData, rb: TimeSeriesData, freq: Optional[str] = None, geometric=True):
    """
    An average annualized excess return is convenient for comparing excess returns

    Excess returns is calculated by first annualizing the asset returns and benchmark returns stream. See the docs for
    `annualized_returns()` for more details. The geometric returns formula is:

        r_g = (ra - rb) / (1 + rb)

    The arithmetic excess returns formula is:

        r_g = ra - rb

    Returns calculation will be truncated by the one with the shorter length. Also, annualized returns are calculated
    by the geometric annualized returns in both cases

    :param ra: iterable data
        The assets returns vector or matrix
    :param rb: iterable data
        The benchmark returns. If this is a vector and the asset returns is a matrix, then all assets returns (columns)
        will be compared against this single benchmark. Otherwise, if this is a matrix, then assets will be compared
        to each individual benchmark (i.e. column for column)
    :param freq: str, optional
        frequency of the data. Use one of [daily, weekly, monthly, quarterly, semi-annually, yearly]
    :param geometric: boolean, default True
        If True, calculates the geometric excess returns. Otherwise, calculates the arithmetic excess returns
    :return:
    """
    ra = to_time_series(ra).dropna()
    rb = to_time_series(rb).dropna()

    n = min(len(ra), len(rb))
    ra, rb = ra.iloc[:n], rb.iloc[:n]

    if ra.ndim == rb.ndim and ra.shape != rb.shape:
        raise ValueError('The shapes of the asset and benchmark returns do not match!')

    if freq is None:
        fa, fb = ra.ppa.frequency, rb.ppa.frequency
        if fa is None and fb is None:
            raise TimeIndexError
        elif fa is None:
            freq = fb
        elif fb is None:
            freq = fa
        elif fa != fb:
            raise TimeIndexMismatchError(fa, fb)

    ra = annualized_returns(ra, freq)
    rb = annualized_returns(rb, freq)

    return (ra - rb) / (1 + rb) if geometric else ra - rb
