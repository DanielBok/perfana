from typing import Iterable, Union

import numpy as np

from perfana.types import Vector
from ._types import Attribution, Frequency
from ._utility import infer_frequency

__all__ = [
    "annualized_returns_m",
    "annualized_bmk_returns_m",
    "annualized_quantile_returns_m",
    "annualized_bmk_quantile_returns_m",
    "returns_attr",
]


def annualized_returns_m(data: np.ndarray,
                         weights: Vector,
                         freq: Frequency,
                         geometric: bool = True,
                         rebalance: bool = True) -> float:
    r"""
    Calculates the annualized returns from the Monte Carlo simulation

    The formula for annualized geometric returns is formulated by raising the compound return to the number of
    periods in a year, and taking the root to the number of total observations. For the rebalance, geometric
    returns, the annualized returns is derived by:

    .. math::

        y = M / s

        \frac{1}{N}\sum^N_i [\prod_j^T(1 + \sum^A_k (r_{ijk} \cdot w_k))]^{\frac{1}{y}} - 1

    where `s` is the number of observations in a year, and `M` is the total number of observations, `N` is
    the number of trials in the simulation, `T` is the number of trials in the simulation and `A` is the
    number of assets in the simulation.

    For simple returns (geometric=FALSE), the formula for the rebalanced case is:

    .. math::

        \frac{\text{scale}}{NM} [\sum^N_i \sum^T_j \sum^A_k (r_{ijk} \cdot w_k)]

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    float
        Annualized returns of the portfolio

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import annualized_returns_m
    >>> cube = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> annualized_returns_m(cube, weights, 'month')
    0.02111728739277985
    """

    freq = infer_frequency(freq)
    w = np.ravel(weights)  # force weights to be a 1D vector
    y = len(data) / freq  # number of years

    if rebalance:
        if geometric:
            d = ((data @ w) + 1).prod(0)
            return (np.sign(d) * np.abs(d) ** (1 / y)).mean() - 1
        else:
            return (data @ w).mean() * freq
    else:
        d = ((data + 1).prod(0) @ w)
        if geometric:
            return (np.sign(d) * np.abs(d) ** (1 / y)).mean() - 1
        else:
            return (d - 1).mean() * freq


def annualized_bmk_returns_m(data: np.ndarray,
                             weights: Vector,
                             bmk: np.ndarray,
                             bmk_weights: Vector,
                             freq: Frequency,
                             geometric: bool = True,
                             rebalance: bool = True) -> float:
    """
    Calculates the returns of the portfolio relative to a benchmark portfolio.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    bmk
        Benchmark data. This must also be 3 dimensional like the `data` object with the axis representing
        time, trial and asset respectively.

    bmk_weights
        Weights of the benchmark portfolio.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    float
        The portfolio returns relative to the benchmark
    """

    freq = infer_frequency(freq)
    w1 = np.ravel(weights)  # force weights to be a 1D vector
    w2 = np.ravel(bmk_weights)
    y = len(data) / freq  # number of years

    assert len(data) == len(bmk), \
        f"data length ({len(data)}) != bmk length ({len(bmk)}). Both must have the same period length. "

    # TODO check implementation of the arithmetic mean
    if rebalance:
        if geometric:
            d = ((data @ w1) + 1).prod(0) / ((bmk @ w2) + 1).prod(0)
            return (np.sign(d) * np.abs(d) ** (1 / y)).mean() - 1
        else:
            return ((data @ w1) / (bmk @ w2)).mean() * freq
    else:
        d = ((data + 1).prod(0) @ w1) / ((bmk + 1).prod(0) @ w2)
        if geometric:
            return (np.sign(d) * np.abs(d) ** (1 / y)).mean() - 1
        else:
            return (d - 1).mean() * freq


def annualized_quantile_returns_m(data: np.ndarray,
                                  weights: Vector,
                                  quantile: Union[float, Iterable[float]],
                                  freq: Frequency,
                                  geometric: bool = True,
                                  rebalance: bool = True,
                                  interpolation="midpoint") -> Union[float, np.ndarray]:
    """
    Compute the q-th quantile of the returns in the simulated data cube.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    quantile
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    interpolation
        This optional parameter specifies the interpolation method to use when the desired quantile
        lies between two data points ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.
    Returns
    -------
    float
        The returns of the portfolio relative to the benchmark at the specified quantile
    """

    freq = infer_frequency(freq)
    w = np.ravel(weights)  # force weights to be a 1D vector
    y = len(data) / freq  # number of years

    if rebalance:
        if geometric:
            d = ((data @ w) + 1).prod(0)
            return np.quantile((np.sign(d) * np.abs(d) ** (1 / y)), quantile, interpolation=interpolation) - 1
        else:
            return np.quantile((data @ w), quantile, interpolation=interpolation) * freq
    else:
        d = ((data + 1).prod(0) @ w)
        if geometric:
            return np.quantile((np.sign(d) * np.abs(d)), quantile, interpolation=interpolation) ** (1 / y) - 1
        else:
            return np.quantile((d - 1), quantile, interpolation=interpolation) * freq


def annualized_bmk_quantile_returns_m(data: np.ndarray,
                                      weights: Vector,
                                      bmk: np.ndarray,
                                      bmk_weights: Vector,
                                      quantile: Union[float, Iterable[float]],
                                      freq: Frequency,
                                      geometric: bool = True,
                                      rebalance: bool = True,
                                      interpolation="midpoint") -> float:
    """


    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    bmk
        Benchmark data. This must also be 3 dimensional like the `data` object with the axis representing
        time, trial and asset respectively.

    bmk_weights
        Weights of the benchmark portfolio.

    quantile
        Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    interpolation
        This optional parameter specifies the interpolation method to use when the desired quantile
        lies between two data points ``i < j``:

            * linear: ``i + (j - i) * fraction``, where ``fraction``
              is the fractional part of the index surrounded by ``i``
              and ``j``.
            * lower: ``i``.
            * higher: ``j``.
            * nearest: ``i`` or ``j``, whichever is nearest.
            * midpoint: ``(i + j) / 2``.

    Returns
    -------

    """
    freq = infer_frequency(freq)
    w1 = np.ravel(weights)  # force weights to be a 1D vector
    w2 = np.ravel(bmk_weights)
    y = len(data) / freq  # number of years

    if rebalance:
        if geometric:
            d = ((data @ w1) + 1).prod(0) / ((bmk @ w2) + 1).prod(0)
            return np.quantile((np.sign(d) * np.abs(d) ** (1 / y)), quantile, interpolation=interpolation) - 1
        else:
            return np.quantile(((data @ w1) / (bmk @ w2)), quantile, interpolation=interpolation) * freq
    else:
        d = ((data + 1).prod(0) @ w1) / ((bmk + 1).prod(0) @ w2)
        if geometric:
            return np.quantile((np.sign(d) * np.abs(d) ** (1 / y)), quantile, interpolation=interpolation) - 1
        else:
            return np.quantile((d - 1), quantile, interpolation=interpolation) * freq


def returns_attr(data: np.ndarray,
                 weights: Vector,
                 freq: Frequency,
                 geometric: bool = True,
                 rebalance: bool = True) -> Attribution:
    """
    Derives the returns attribution given a data cube and weights.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    Attribution
        A named tuple of marginal and percentage returns attribution respectively. The marginal attribution
        is the returns of the simulated data over time multiplied by the percentage attribution.
    """

    data = np.asarray(data)
    t, n, a = data.shape
    assert a == len(weights), "length of weights must match number of assets in data cube"

    freq = infer_frequency(freq)

    # calculate annualized returns before reshape
    gr = annualized_returns_m(data, weights, freq, geometric, rebalance)
    data = (data + 1).reshape(t // freq, freq, n, a).prod(1) - 1
    ar = data.mean((0, 1))  # arithmetic returns

    # TODO Check if non-rebalanced attribution are still calculated similarly
    attr_p = (ar * weights) / (ar @ weights)
    attr_m = attr_p * gr

    return Attribution(attr_m, attr_p)
