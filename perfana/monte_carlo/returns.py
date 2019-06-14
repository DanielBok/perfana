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
                             bmk_weights: Vector,
                             freq: Frequency,
                             geometric: bool = True,
                             rebalance: bool = True) -> float:
    """
    Calculates the returns of the portfolio relative to a benchmark portfolio.

    The benchmark components must be placed after the portfolio components in the simulated returns cube.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

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

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import annualized_bmk_returns_m

    >>> cube = load_cube()
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> bmk_weights = [0.65, 0.35]
    >>> freq = "quarterly"
    >>> annualized_bmk_returns_m(cube, weights, bmk_weights, freq)
    -0.006819613944426206
    """

    freq = infer_frequency(freq)
    w1, w2 = np.ravel(weights), np.ravel(bmk_weights)
    y = len(data) / freq  # number of years
    n = len(w1)
    port, bmk = data[..., :n], data[..., n:]

    # TODO check implementation of the arithmetic mean
    if rebalance:
        if geometric:
            d = ((port @ w1) + 1).prod(0) / ((bmk @ w2) + 1).prod(0)
            return (np.sign(d) * np.abs(d) ** (1 / y)).mean() - 1
        else:
            return ((port @ w1) / (bmk @ w2)).mean() * freq
    else:
        d = ((port + 1).prod(0) @ w1) / ((bmk + 1).prod(0) @ w2)
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

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import annualized_quantile_returns_m

    >>> cube = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> freq = "quarterly"
    >>> q = 0.25
    >>> annualized_quantile_returns_m(cube, weights, q, freq)
    0.005468353416130167
    >>> q = [0.25, 0.75]
    >>> annualized_quantile_returns_m(cube, weights, q, freq)
    array([0.00546835, 0.03845033])
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
                                      bmk_weights: Vector,
                                      quantile: Union[float, Iterable[float]],
                                      freq: Frequency,
                                      geometric: bool = True,
                                      rebalance: bool = True,
                                      interpolation="midpoint") -> Union[float, np.ndarray]:
    """
    Compares the annualized returns against a benchmark at the specified quantiles.

    The benchmark components must be placed after the portfolio components in the simulated returns cube.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

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
    float or array_like of floats
        The returns of the portfolio over the benchmark at the specified quantiles

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import annualized_bmk_quantile_returns_m

    >>> cube = load_cube()
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> bmk_weights = [0.65, 0.35]
    >>> freq = "quarterly"
    >>> q = 0.25
    >>> annualized_bmk_quantile_returns_m(cube, weights, bmk_weights, q, freq)
    -0.010792419409674459
    >>> q = [0.25, 0.75]
    >>> annualized_bmk_quantile_returns_m(cube, weights, bmk_weights, q, freq)
    array([-0.01079242, -0.0025487 ])
    """
    freq = infer_frequency(freq)
    w1, w2 = np.ravel(weights), np.ravel(bmk_weights)
    y = len(data) / freq  # number of years
    n = len(w1)
    port, bmk = data[..., :n], data[..., n:]

    if rebalance:
        if geometric:
            d = ((port @ w1) + 1).prod(0) / ((bmk @ w2) + 1).prod(0)
            return np.quantile((np.sign(d) * np.abs(d) ** (1 / y)), quantile, interpolation=interpolation) - 1
        else:
            return np.quantile(((port @ w1) / (bmk @ w2)), quantile, interpolation=interpolation) * freq
    else:
        d = ((port + 1).prod(0) @ w1) / ((bmk + 1).prod(0) @ w2)
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

    Notes
    -----
    The return values are defined as follows:

    **marginal**
        The absolute marginal contribution of the asset class towards the portfolio returns.
        It is essentially the percentage attribution multiplied by the portfolio returns.

    **percentage**
        The percentage contribution of the asset class towards the portfolio returns. This number
        though named in percentage is actually in decimals. Thus 0.01 represents a 1% contribution.

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

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import returns_attr

    >>> cube = load_cube()[..., :3]
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = "quarterly"
    >>> attr = returns_attr(cube, weights, freq)
    >>> attr.marginal
    array([0.00996204, 0.00733369, 0.00963802])
    >>> attr.percentage
    array([0.36987203, 0.27228623, 0.35784174])
    >>> attr.marginal is attr[0]
    True
    >>> attr.percentage is attr[1]
    True
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
