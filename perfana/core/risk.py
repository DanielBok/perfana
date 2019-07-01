from typing import Optional

import numpy as np
import pandas as pd

from perfana.conversions import to_time_series
from perfana.types import TimeSeriesData, Vector

__all__ = ["drawdown", "drawdown_summary"]


def drawdown(data: TimeSeriesData, weights: Vector = None, geometric=True, rebalance=True) -> pd.Series:
    """
    Calculates the drawdown at each time instance.

    If data is DataFrame-like, weights must be specified. If data is Series-like, weights
    can be left empty.

    Parameters
    ----------
    data
        The assets returns vector or matrix

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    Series
        Drawdown at each time instance

    Examples
    --------
    >>> from perfana.datasets import load_hist
    >>> from perfana.core import drawdown
    >>> hist = load_hist().iloc[:, :7]
    >>> weights = [0.25, 0.18, 0.24, 0.05, 0.04, 0.13, 0.11]
    >>> drawdown(hist, weights).min()
    -0.4007984968456346
    >>> drawdown(hist.iloc[:, 0]).min()
    -0.5491340502573534

    .. plot:: plots/core_drawdown.py
        :include-source:
    """
    data = to_time_series(data)
    weights = np.ravel(weights)

    assert data.ndim in (1, 2), "returns can only be a series or a dataframe"
    if data.ndim == 2:
        assert weights is not None, "weight must be specified if returns is a dataframe"
        weights = np.ravel(weights)

        if rebalance:
            cum_ret = (data @ weights + 1).cumprod() if geometric else (data @ weights).cumsum() + 1
        else:
            cum_ret = (data + 1).cumprod() @ weights if geometric else data.cumsum() @ weights + 1
    else:
        cum_ret = (data + 1).cumprod() if geometric else data.cumsum() + 1

    dd = cum_ret / cum_ret.expanding().max() - 1
    setattr(dd, "is_drawdown", True)
    return dd


def drawdown_summary(data: TimeSeriesData, weights: Vector = None, geometric=True, rebalance=True, *,
                     top: Optional[int] = 5) -> pd.DataFrame:
    """
    A summary of each drawdown instance. Output is ranked by depth of the drawdown.

    If data is DataFrame-like, weights must be specified. If data is Series-like, weights
    can be left empty.

    Parameters
    ----------
    data
        The assets returns vector or matrix

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    top
        If None, returns all episodes. If specified, returns the top `n` episodes ranked by the depth
        of drawdown.

    Returns
    -------
    DataFrame
        A data frame summarizing each drawdown episode

    Examples
    --------
    >>> from perfana.datasets import load_hist
    >>> from perfana.core import drawdown_summary
    >>> hist = load_hist().iloc[:, :7]
    >>> weights = [0.25, 0.18, 0.24, 0.05, 0.04, 0.13, 0.11]
    >>> drawdown_summary(hist, weights)
           Start     Trough        End  Drawdown  Length  ToTrough  Recovery
    0 2007-11-30 2009-02-28 2014-02-28 -0.400798      76        16        60
    1 2000-04-30 2003-03-31 2004-02-29 -0.203652      47        36        11
    2 1990-01-31 1990-11-30 1991-05-31 -0.150328      17        11         6
    3 1998-04-30 1998-10-31 1999-06-30 -0.149830      15         7         8
    4 1994-02-28 1995-03-31 1996-01-31 -0.132766      24        14        10
    >>> drawdown_summary(hist.iloc[:, 0])
           Start     Trough        End  Drawdown  Length  ToTrough  Recovery
    0 2007-11-30 2009-02-28 2014-05-31 -0.549134      79        16        63
    1 2000-04-30 2003-03-31 2006-12-31 -0.474198      81        36        45
    2 1990-01-31 1990-09-30 1994-01-31 -0.286489      49         9        40
    3 1998-08-31 1998-09-30 1999-01-31 -0.148913       6         2         4
    4 2018-10-31 2018-12-31 2019-03-31 -0.130014       6         3         3
    """
    if not hasattr(data, 'is_drawdown') or not data.is_drawdown:
        data = drawdown(data, weights, geometric, rebalance)

    info = {
        "Drawdown": [],
        "Start": [],
        "Trough": [],
        "End": [],
    }

    # use indices to store date diff initially as it isn't easy to determine the unit of time
    # (days, months, etc). Also, heuristics like round(days / 30) breaks easily when date lengths
    # are long (month day diff and leap years break)
    prior_sign = 1 if data[0] >= 0 else 0
    min_, from_, current_ = 0, 0, data[0]

    # index, drawdown
    for i, dd in enumerate(data):
        sign = 0 if dd < 0 else 1

        if sign == prior_sign:
            if dd < current_:
                current_ = dd
                min_ = i
        else:
            info["Drawdown"].append(current_)
            info["Start"].append(from_)
            info["Trough"].append(min_)
            info["End"].append(i)
            min_, from_, current_ = i, i, dd
            prior_sign = sign

    info["Drawdown"].append(current_)
    info["Start"].append(from_)
    info["Trough"].append(min_)
    info["End"].append(len(data) - 1)

    # calculate various drawdown lengths
    df = pd.DataFrame(info)
    df["Length"] = df["End"] - df["Start"] + 1
    df["ToTrough"] = df["Trough"] - df["Start"] + 1
    df["Recovery"] = df["End"] - df["Trough"]

    # sort by drawdown
    df = df.sort_values('Drawdown').reset_index(drop=True)

    # convert indices to date
    df["Start"] = data.index[df["Start"]]
    df["Trough"] = data.index[df["Trough"]]
    df["End"] = data.index[df["End"]]

    # reorder columns
    df = df[df.Drawdown < 0][['Start', 'Trough', 'End', 'Drawdown', 'Length', 'ToTrough', 'Recovery']]

    assert top is None or (isinstance(top, int) and top > 0), "If top is not None, it must be a positive integer"

    if top is None:
        return df

    return df.head(top)
