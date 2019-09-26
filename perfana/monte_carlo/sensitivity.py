from typing import List

import numpy as np
import pandas as pd

from perfana.monte_carlo._utility import infer_frequency
from perfana.types import Vector
from ._types import Frequency
from .returns import annualized_returns_m
from .risk import cvar_m, volatility_m

__all__ = ["sensitivity_m", "sensitivity_cvar_m", "sensitivity_returns_m", "sensitivity_vol_m"]


def sensitivity_m(data: np.ndarray,
                  weights: Vector,
                  freq: Frequency,
                  shock: float = 0.05,
                  geometric: bool = True,
                  rebalance: bool = True,
                  cov: np.ndarray = None,
                  cvar_cutoff: int = 3,
                  cvar_data: np.ndarray = None,
                  alpha=0.95,
                  invert=True,
                  names: List[str] = None,
                  leveraged=False,
                  distribute=True) -> pd.DataFrame:
    """
    Calculates the sensitivity of adding and removing from the asset class on the portfolio.

    This is a wrapper function for the 3 sensitivity calculations. For more granular usages, use the base
    functions instead.

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

    freq: Frequency
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    shock: float
        The amount to shock each asset class by. A positive number represents adding to the asset class by
        proportionately removing from the other asset class. A negative number represents removing from the
        asset class and adding to the other asset class proportionately.

    geometric: bool
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    cov: ndarray
        Asset covariance matrix

    cvar_cutoff: int
        Number of years to trim the data cube by for cvar calculation.

    cvar_data: np.ndarray
        If specified, will use this data cube instead of the main data cube for cvar calculations.

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
    >>> sensitivity_m(data, weights, freq, shock)
                  ret       vol      cvar
    Asset_1  0.022403  0.113284 -0.485220
    Asset_2  0.020484  0.121786 -0.542988
    Asset_3  0.022046  0.113964 -0.492411
    Asset_4  0.020854  0.109301 -0.478581
    Asset_5  0.020190  0.104626 -0.459786
    Asset_6  0.020335  0.106652 -0.467798
    Asset_7  0.020220  0.106140 -0.468692
    """
    assert isinstance(cvar_cutoff, int) and cvar_cutoff > 0, "cvar_cutoff must be a positive integer"
    cov_or_data = data if cov is None else cov

    if cvar_data is None:
        cvar_data = data
    cvar_data = cvar_data[:cvar_cutoff * infer_frequency(freq)]

    ret = sensitivity_returns_m(data, weights, freq, shock, geometric, rebalance, names, leveraged, distribute)
    vol = sensitivity_vol_m(cov_or_data, weights, freq, shock, names, leveraged, distribute)
    cvar = sensitivity_cvar_m(cvar_data, weights, shock, alpha, rebalance, invert, names, leveraged, distribute)

    return pd.merge(ret, vol, left_index=True, right_index=True).merge(cvar, left_index=True, right_index=True)


def sensitivity_cvar_m(data: np.ndarray,
                       weights: Vector,
                       shock: float = 0.05,
                       alpha=0.95,
                       rebalance: bool = True,
                       invert=True,
                       names: List[str] = None,
                       leveraged=False,
                       distribute=True) -> pd.Series:
    """
    Calculates the sensitivity of a shock to the CVaR of the portfolio

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
    Series
        A series with asset names as the index and CVaR as its value

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import sensitivity_cvar_m
    >>> data = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> freq = 'quarterly'
    >>> shock = 0.05  # 5% absolute shock
    >>> sensitivity_cvar_m(data, weights, shock)
    Asset_1   -0.485220
    Asset_2   -0.542988
    Asset_3   -0.492411
    Asset_4   -0.478581
    Asset_5   -0.459786
    Asset_6   -0.467798
    Asset_7   -0.468692
    Name: cvar, dtype: float64
    """
    weight_matrix = _setup(weights, shock, leveraged, distribute)
    names = _setup_names(weights, names)
    cvar = [cvar_m(data, w, alpha, rebalance, invert) for w in weight_matrix]

    return pd.Series(cvar, names, name="cvar")


def sensitivity_returns_m(data: np.ndarray,
                          weights: Vector,
                          freq: Frequency,
                          shock: float = 0.05,
                          geometric: bool = True,
                          rebalance: bool = True,
                          names: List[str] = None,
                          leveraged=False,
                          distribute=True) -> pd.Series:
    """
    Calculates the sensitivity of a shock to the annualized returns of the portfolio

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
    Series
        A series with asset names as the index and annualized returns as its value

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import sensitivity_returns_m
    >>> data = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> freq = 'quarterly'
    >>> shock = 0.05  # 5% absolute shock
    >>> sensitivity_returns_m(data, weights, freq, shock)
    Asset_1    0.022403
    Asset_2    0.020484
    Asset_3    0.022046
    Asset_4    0.020854
    Asset_5    0.020190
    Asset_6    0.020335
    Asset_7    0.020220
    Name: ret, dtype: float64
    """
    weight_matrix = _setup(weights, shock, leveraged, distribute)
    names = _setup_names(weights, names)
    ret = [annualized_returns_m(data, w, freq, geometric, rebalance) for w in weight_matrix]

    return pd.Series(ret, names, name="ret")


def sensitivity_vol_m(cov_or_data: np.ndarray,
                      weights: Vector,
                      freq: Frequency = None,
                      shock: float = 0.05,
                      names: List[str] = None,
                      leveraged=False,
                      distribute=True) -> pd.Series:
    """
    Calculates the sensitivity of a shock to the annualized volatility of the portfolio

    Parameters
    ----------
    cov_or_data
        Monte carlo simulation data or covariance matrix. If simulation cube, this must be 3 dimensional with
        the axis representing time, trial and asset respectively and frequency will also need to be specified.

    weights: array_like
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    freq: Frequency
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    shock: float
        The amount to shock each asset class by. A positive number represents adding to the asset class by
        proportionately removing from the other asset class. A negative number represents removing from the
        asset class and adding to the other asset class proportionately.

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
    Series
        A series with asset names as the index and annualized volatility as its value

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import sensitivity_vol_m
    >>> data = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> freq = 'quarterly'
    >>> shock = 0.05  # 5% absolute shock
    >>> sensitivity_vol_m(data, weights, freq, shock)
    Asset_1    0.113284
    Asset_2    0.121786
    Asset_3    0.113964
    Asset_4    0.109301
    Asset_5    0.104626
    Asset_6    0.106652
    Asset_7    0.106140
    Name: vol, dtype: float64
    """
    weight_matrix = _setup(weights, shock, leveraged, distribute)
    names = _setup_names(weights, names)
    vol = [volatility_m(cov_or_data, w, freq) for w in weight_matrix]

    return pd.Series(vol, names, name="vol")


def _setup(weights: Vector,
           shock: float = 0.05,
           leveraged=False,
           distribute=True,
           names: List[str] = None):
    """Common setup for sensitivity analytics"""
    assert -1 <= shock <= 1, "shock must be between [-1, 1]"
    weights = np.ravel(weights)

    # if leverage is True:
    # Prevents shocks beyond the asset's current allocation. That is if the shock is
    # -5% and the asset only has 2% allocation, then the shock is effectively -2%.
    # And prevent shocks that bring the assets beyond 100% allocation.
    shocks = np.array([
        shock if (0 <= w + shock <= 1) or leveraged else
        -w if shock < 0 else 1 - w
        for w in weights
    ])

    n = len(weights)

    matrix = np.tile(weights, (n, 1)) + np.diag(shocks)
    if distribute:
        weight_matrix = np.tile(weights, (n, 1))
        np.fill_diagonal(weight_matrix, 0)

        matrix -= weight_matrix * (shocks / weight_matrix.sum(1))[:, None]

    if names is None:
        names = [f"Asset_{i + 1}" for i in range(len(weights))]

    return matrix, names


def _setup_names(weights: np.ndarray, names: List[str] = None):
    if names is None:
        return [f"Asset_{i + 1}" for i in range(len(weights))]

    names = list(names)
    assert len(names) == len(weights), "number of names given is not equal to length of weight vector"
    return names
