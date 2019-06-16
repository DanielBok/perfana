import numpy as np
from copulae.core import is_psd

from perfana.monte_carlo.returns import annualized_bmk_returns_m, annualized_returns_m
from perfana.monte_carlo.risk import tracking_error_m, volatility_m
from perfana.types import Vector
from ._types import Frequency

__all__ = ["sharpe_ratio_m", "sharpe_ratio_bmk_m"]


def sharpe_ratio_m(data: np.ndarray,
                   weights: Vector,
                   freq: Frequency,
                   cov: np.ndarray = None,
                   geometric: bool = True,
                   rebalance: bool = True) -> float:
    """
    Calculates the Sharpe Ratio of portfolio

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

    cov
        Covariance matrix. If not specified, the empirical covariance matrix from the joined data and
        benchmark data will be derived.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    float
        Sharpe ratio of the portfolio

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import sharpe_ratio_m
    >>> cube = load_cube()[..., :3]
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = 'quarterly'
    >>> sharpe_ratio_m(cube, weights, freq)
    0.14242462120295252
    """
    gr = annualized_returns_m(data, weights, freq, geometric, rebalance)
    vol = volatility_m(cov if cov is not None else data, weights, freq)
    return gr / vol


def sharpe_ratio_bmk_m(data: np.ndarray,
                       weights: Vector,
                       bmk_weights: Vector,
                       freq: Frequency,
                       cov: np.ndarray = None,
                       geometric: bool = True,
                       rebalance: bool = True):
    """
    Calculates the Sharpe Ratio of the portfolio over the benchmark.

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

    cov
        Covariance matrix. If not specified, the empirical covariance matrix from the joined data and
        benchmark data will be derived.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    float
        Sharpe ratio of the portfolio over the benchmark

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import sharpe_ratio_bmk_m
    >>> cube = load_cube()
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> bmk_weights = [0.65, 0.35]
    >>> freq = 'quarterly'
    >>> sharpe_ratio_bmk_m(cube, weights, bmk_weights, freq)
    -0.2186945589389277
    """
    gr = annualized_bmk_returns_m(data, weights, bmk_weights, freq, geometric, rebalance)

    if cov is not None:
        n = len(weights) + len(bmk_weights)
        assert cov.shape == (n, n), "covariance matrix shape incorrect"
        assert is_psd(cov), "covariance matrix is not positive semi-definite"
        data = cov

    te = tracking_error_m(data, weights, bmk_weights, freq)
    return gr / te
