from typing import List

import numpy as np
import pandas as pd

from perfana.monte_carlo._utility import infer_frequency
from perfana.types import Vector
from ._types import Frequency
from .returns import annualized_returns_m
from .risk import cvar_m, volatility_m

__all__ = ["sensitivity_m"]


def sensitivity_m(data: np.ndarray,
                  weights: Vector,
                  freq: Frequency,
                  shock: float = 0.05,
                  geometric: bool = True,
                  rebalance: bool = True,
                  cov: np.ndarray = None,
                  cvar_cutoff: int = 3,
                  alpha=0.95,
                  invert=True,
                  names: List[str] = None,
                  leveraged=False,
                  distribute=True) -> pd.DataFrame:
    """
    Calculates the sensitivity of adding and removing from the asset class on the portfolio.

    Notes
    -----
    When given a positive shock and a "proportionate" distribution strategy, each asset class is given an
    additional amount by removing from the other asset classes proportionately. For example, given a portfolio
    with weights :code:`[0.1, 0.2, 0.3, 0.4]`, a shock of 5% to the first asset in the portfolio will result
    in weights :code:`[0.15, 0.19, 0.28, 0.38]`. A negative shock works by removing from the asset class and
    adding to the other asset classes proportionately.

    If the distribution strategy is set to :code:`False`, the asset class' weight is increased without removing
    from the other asset classes. Thus the sum of the portfolio weights will not equal 1.

    By default, the portfolio is **not** leveraged. This means that the asset class be shorted (negative shock) to
    go below 0 and levered (positive shock) to go above 1. The asset class weight is thus capped between 0 and 1
    by default. If the :code:`leverage` option is set to :code:`True`, then this value is no longer capped.

    Parameters
    ----------
    data: ndarray
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights: array_like
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    shock: float
        The amount to shock each asset class by. A positive number represents adding to the asset class by
        proportionately removing from the other asset class. A negative number represents removing from the
        asset class and adding to the other asset class proportionately.

    freq: Frequency
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    geometric: bool
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    cov: ndarray
        Asset covariance matrix

    cvar_cutoff: int
        Number of years to trim the data cube by for cvar calculation.

    alpha: float
        Confidence level for calculation.

    invert: bool
        Whether to invert the confidence interval level

    rebalance: bool
        If True, portfolio is assumed to be rebalanced at every step.

    names: list of str
        Asset class names

    leveraged: bool
        If True, asset weights are allowed to go below 0 and above 1. This represents that the
        asset class can be shorted or levered.

    distribute: bool
        If True, asset value changes are distributed proportionately to all other asset classes. See Notes
        for more information.

    Returns
    -------
    DataFrame
        A dataframe with the asset names as the indices and with columns (ret, vol, cvar) representing
        returns, volatility and CVaR respectively.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import sensitivity_m
    >>> data = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> freq = 'quarterly'
    >>> shock = 0.05  # 5% absolute shock
    >>> sensitivity_m(data, weights, freq)
                  ret       vol      cvar
    Asset_1  0.022403  0.113284 -0.485220
    Asset_2  0.020484  0.121786 -0.542988
    Asset_3  0.022046  0.113964 -0.492411
    Asset_4  0.020854  0.109301 -0.478581
    Asset_5  0.020190  0.104626 -0.459786
    Asset_6  0.020335  0.106652 -0.467798
    Asset_7  0.020220  0.106140 -0.468692
    """
    assert -1 <= shock <= 1, "shock must be between [-1, 1]"
    assert isinstance(cvar_cutoff, int) and cvar_cutoff > 0, "cvar_cutoff must be a positive integer"
    weights = np.ravel(weights)
    freq = infer_frequency(freq)

    if leveraged:
        min_shock = shock
    else:
        if shock < 0:
            # prevents shocks beyond the asset's current allocation. That is if the shock is
            # -5% and the asset only has 2% allocation, then the shock is effectively -2%.
            min_shock = np.minimum(weights, -shock)
        else:
            # prevent shocks that bring the assets beyond 100% allocation.
            min_shock = np.minimum(1 - weights, shock)

    n = len(weights)

    if distribute:
        e, o = np.eye(n), np.ones((n, n))
        r = (o - e) * weights

        rows = e * min_shock - (min_shock / r.sum(1) * r.T).T
    else:
        rows = min_shock * np.eye(n)

    cov_or_data = data if cov is None else cov
    res = {
        "ret": [],
        "vol": [],
        "cvar": [],
    }

    for r in rows:
        w = weights + r
        res["ret"].append(annualized_returns_m(data, w, freq, geometric, rebalance))
        res["vol"].append(volatility_m(cov_or_data, w, freq))
        res["cvar"].append(cvar_m(data[:cvar_cutoff * freq], w, alpha, rebalance, invert))

    if names is None:
        names = [f"Asset_{i + 1}" for i in range(n)]

    return pd.DataFrame(res, names)
