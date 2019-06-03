from typing import Optional, Union

import pandas as pd

from perfana.conversions import to_time_series
from perfana.core.utils import freq_to_scale
from perfana.exceptions import TimeIndexError, TimeIndexMismatchError
from perfana.types import TimeSeriesData

__all__ = ['active_premium', 'annualized_returns', 'excess_returns', 'relative_returns']


def active_premium(ra: TimeSeriesData,
                   rb: TimeSeriesData,
                   freq: Optional[str] = None,
                   geometric=True,
                   prefixes=('AST', 'BMK')) -> TimeSeriesData:
    """
    The return on an investment's annualized return minus the benchmark's annualized return.

    Parameters
    ----------
    ra:
        The assets returns vector or matrix

    rb:
        The benchmark returns

    freq:
        Frequency of the data. Use one of daily, weekly, monthly, quarterly, semi-annually, yearly

    geometric:
        If True, calculates the geometric returns. Otherwise, calculates the arithmetic returns

    prefixes:
        Prefix to apply to overlapping column names in the left and right side, respectively. This is also applied
        when the column name is an integer (i.e. 0 -> AST_0). It is the default name of the Series data if there
        are no name to the Series

    Returns
    -------
    TimeSeriesData
        Active premium of each strategy against benchmark

    Examples
    --------
    >>> from perfana.datasets import load_etf
    >>> from perfana.core import active_premium
    # Get returns starting from the date where all etf has data
    >>> etf = load_etf().dropna().ppa.to_returns().dropna()
    >>> active_premium(etf, etf)
              VBK       BND       VTI       VWO
    VBK  0.000000 -0.055385 -0.010407 -0.063939
    BND  0.055385  0.000000  0.044979 -0.008554
    VTI  0.010407 -0.044979  0.000000 -0.053532
    VWO  0.063939  0.008554  0.053532  0.000000

    >>> active_premium(etf.VBK, etf.BND)
              VBK
    BND  0.055385
    """
    ra = to_time_series(ra)
    rb = to_time_series(rb)

    if isinstance(ra, pd.Series):
        ra = pd.DataFrame(ra.rename(ra.name or prefixes[0]))

    if isinstance(rb, pd.Series):
        rb = pd.DataFrame(rb.rename(rb.name or prefixes[1]))

    freq = _determine_frequency(ra, rb, freq)

    res = pd.DataFrame()
    for ca, a in ra.iteritems():
        premium = {}
        if isinstance(ca, int):
            ca = f'{prefixes[0]}_{ca}'

        for cb, b in rb.iteritems():
            if isinstance(cb, int):
                cb = f'{prefixes[1]}_{cb}'

            premium[cb] = annualized_returns(a, freq, geometric) - annualized_returns(b, freq, geometric)
        res[ca] = pd.Series(premium)

    return res


def annualized_returns(r: TimeSeriesData,
                       freq: Optional[str] = None,
                       geometric=True) -> Union[float, pd.Series]:
    r"""
    Calculates the annualized returns from the data

    The formula for annualized geometric returns is formulated by raising the compound return to the number of
    periods in a year, and taking the root to the number of total observations:

    .. math::
        \prod_i^N(1 + r_i)^{scale/N} - 1

    where scale is the number of observations in a year, and N is the total number of observations.

    For simple returns (geometric=FALSE), the formula is:

    .. math::
        \frac{scale}{N} \sum^N_i r_i


    Parameters
    ----------
    r:
        Numeric returns series or data frame

    freq:
        Frequency of the data. Use one of daily, weekly, monthly, quarterly, semi-annually, yearly

    geometric:
        If True, calculates the geometric returns. Otherwise, calculates the arithmetic returns

    Returns
    -------
    float or Series
        Annualized returns

    Examples
    --------
    >>> from perfana.datasets import load_etf
    >>> from perfana.core import active_premium
    # Get returns starting from the date where all etf has data
    >>> etf = load_etf().dropna().ppa.to_returns().dropna()
    VBK    0.091609
    BND    0.036224
    VTI    0.081203
    VWO    0.027670
    dtype: float64
    >>> annualized_returns(etf.VWO)
    0.02767037698144148
    """
    r = to_time_series(r).dropna()
    if freq is None:
        freq = r.ppa.frequency

    scale = freq_to_scale(freq)

    if geometric:
        return (r + 1).prod() ** (scale / len(r)) - 1
    else:  # arithmetic mean
        return r.mean() * scale


def excess_returns(ra: TimeSeriesData,
                   rb: TimeSeriesData,
                   freq: Optional[str] = None,
                   geometric=True) -> TimeSeriesData:
    r"""
    An average annualized excess return is convenient for comparing excess returns

    Excess returns is calculated by first annualizing the asset returns and benchmark returns stream. See the docs for
    `annualized_returns()` for more details. The geometric returns formula is:

    .. math::
        r_g = (ra - rb) / (1 + rb)

    The arithmetic excess returns formula is:

    .. math::
        r_g = ra - rb

    Returns calculation will be truncated by the one with the shorter length. Also, annualized returns are calculated
    by the geometric annualized returns in both cases

    Parameters
    ----------
    ra
        The assets returns vector or matrix

    rb:
        The benchmark returns. If this is a vector and the asset returns is a matrix, then all assets returns (columns)
        will be compared against this single benchmark. Otherwise, if this is a matrix, then assets will be compared
        to each individual benchmark (i.e. column for column)

    freq:
        Frequency of the data. Use one of [daily, weekly, monthly, quarterly, semi-annually, yearly]

    geometric
        If True, calculates the geometric excess returns. Otherwise, calculates the arithmetic excess returns

    Returns
    -------
    TimeSeriesData
        Excess returns of each strategy against benchmark

    Examples
    --------
    >>> from perfana.datasets import load_etf
    >>> from perfana.core import active_premium
    # Get returns starting from the date where all etf has data
    >>> etf = load_etf().dropna().ppa.to_returns().dropna()
    >>> excess_returns(etf, etf.VBK)
    VBK    0.000000
    BND   -0.050737
    VTI   -0.009533
    VWO   -0.058573
    dtype: float64

    """
    ra = to_time_series(ra).dropna()
    rb = to_time_series(rb).dropna()

    n = min(len(ra), len(rb))
    ra, rb = ra.iloc[:n], rb.iloc[:n]

    if ra.ndim == rb.ndim and ra.shape != rb.shape:
        raise ValueError('The shapes of the asset and benchmark returns do not match!')

    freq = _determine_frequency(ra, rb, freq)

    ra = annualized_returns(ra, freq)
    rb = annualized_returns(rb, freq)

    return (ra - rb) / (1 + rb) if geometric else ra - rb


def relative_returns(ra: TimeSeriesData,
                     rb: TimeSeriesData,
                     prefixes=('AST', 'BMK')) -> TimeSeriesData:
    """
    Calculates the ratio of the cumulative performance for two assets through time

    Parameters
    ----------
    ra:
        The assets returns vector or matrix

    rb:
        The benchmark returns

    prefixes:
        Prefix to apply to overlapping column names in the left and right side, respectively. This is also applied
        when the column name is an integer (i.e. 0 -> AST_0). It is the default name of the Series data if there
        are no name to the Series

    Returns
    -------
    TimeSeriesData
        Returns a DataFrame of the cumulative returns ratio between 2 asset classes.
        Returns a Series if there is only 2 compared classes.

    Examples
    --------
    >>> from perfana.datasets import load_etf
    >>> from perfana.core import active_premium
    # Get returns starting from the date where all etf has data
    >>> etf = load_etf().dropna().ppa.to_returns().dropna()
    >>> relative_returns(etf.tail(), etf.VBK.tail())
                VBK/VBK   BND/VBK   VTI/VBK   VWO/VBK
    Date
    2019-02-25      1.0  0.996027  0.997856  1.009737
    2019-02-26      1.0  1.004013  1.002591  1.013318
    2019-02-27      1.0  0.997005  0.997934  1.000389
    2019-02-28      1.0  1.001492  1.001461  0.998348
    2019-03-01      1.0  0.987385  0.997042  0.988521
    """
    ra = to_time_series(ra)
    rb = to_time_series(rb)

    if isinstance(ra, pd.Series):
        ra = pd.DataFrame(ra.rename(ra.name or prefixes[0]))

    if isinstance(rb, pd.Series):
        rb = pd.DataFrame(rb.rename(rb.name or prefixes[1]))

    res = pd.DataFrame()
    for ca, a in ra.iteritems():
        for cb, b in rb.iteritems():
            df = (pd.merge(a, b, 'outer', left_index=True, right_index=True).dropna() + 1).cumprod()
            rel = df.iloc[:, 0] / df.iloc[:, 1]

            if isinstance(ca, int):
                ca = f'{prefixes[0]}_{ca}'
            if isinstance(cb, int):
                cb = f'{prefixes[1]}_{cb}'

            res = pd.merge(res, rel.rename(f'{ca}/{cb}'), 'outer', left_index=True, right_index=True)

    if res.shape[1] == 1:
        return res.iloc[:, 0]
    return res


def _determine_frequency(ra, rb, freq):
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
        else:
            freq = fa

    return freq
