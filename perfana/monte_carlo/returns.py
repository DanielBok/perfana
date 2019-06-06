import numpy as np
from ._types import Frequency
from ._utility import infer_frequency

__all__ = ["annualized_returns_m"]


def annualized_returns_m(data: np.ndarray,
                         weights: np.ndarray,
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
    data: array
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
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean

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
