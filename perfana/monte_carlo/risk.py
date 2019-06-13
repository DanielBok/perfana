import numpy as np
from copulae.core import cov2corr, is_psd

from perfana.types import Vector
from ._types import Attribution, Frequency
from ._utility import infer_frequency

__all__ = [
    "beta_m",
    "correlation_m",
    "diversification_m",
    "portfolio_cov",
    "tracking_error_m",
    "vol_attr",
    "volatility_m",
]


def beta_m(cov_or_data: np.ndarray,
           weights: Vector,
           freq: Frequency = None,
           aid=0) -> float:
    """
    Derives the portfolio beta with respect to the specified asset class

    Notes
    -----
    The asset is identified by its index (aid) on the covariance matrix / simulated
    returns cube / weight vector. If a simulated returns data cube is given,
    the frequency of the data must be specified. In this case, the empirical covariance
    matrix would be used to derive the volatility.

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the
        covariance matrix shape or the simulated data's last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    aid
        Asset index

    Returns
    -------
    float
        Portfolio beta with respect to asset class.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov, beta_m

    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = 'quarterly'
    >>> dm_eq_id = 0  # calculate correlation with respect to developing markets equity
    >>> beta_m(data, weights, freq, dm_eq_id)
    1.3047194776321622
    """
    cov = _portfolio_volatility_structure(cov_or_data, weights, freq, aid)
    return cov[0, 1] / cov[1, 1]


def correlation_m(cov_or_data: np.ndarray,
                  weights: Vector,
                  freq: Frequency = None,
                  aid=0) -> float:
    """
    Derives the portfolio correlation with respect to the specified asset class

    Notes
    -----
    The asset is identified by its index (aid) on the covariance matrix / simulated
    returns cube / weight vector. If a simulated returns data cube is given,
    the frequency of the data must be specified. In this case, the empirical covariance
    matrix would be used to derive the volatility.

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the
        covariance matrix shape or the simulated data's last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    aid
        Asset index

    Returns
    -------
    float
        Portfolio correlation with respect to asset class

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov, correlation_m

    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = 'quarterly'
    >>> dm_eq_id = 0  # calculate correlation with respect to developing markets equity
    >>> correlation_m(data, weights, freq, dm_eq_id)
    0.9642297301278216
    """
    cov = _portfolio_volatility_structure(cov_or_data, weights, freq, aid)
    return cov2corr(cov)[0, 1]


def diversification_m(cov_or_data: np.ndarray,
                      weights: Vector,
                      freq: Frequency) -> float:
    """
    Derives the diversification ratio of the portfolio

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    Returns
    -------
    float
        Tracking error of the portfolio

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov, diversification_m

    >>> data = load_cube()[..., :7]  # first 7 asset classes
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> freq = 'quarterly'
    >>> diversification_m(data, weights, freq)
    """
    cov = _get_covariance_matrix(cov_or_data, freq)
    vol = volatility_m(cov, weights)
    _, asset_vol = cov2corr(cov, True)

    return (weights * asset_vol).sum() / vol


def portfolio_cov(data: np.ndarray,
                  freq: Frequency) -> np.ndarray:
    """
    Forms the empirical portfolio covariance matrix from the simulation data cube

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    Returns
    -------
    array_like of float
        Empirical portfolio covariance matrix

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov

    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> portfolio_cov(data, 'quarterly').round(4)
    array([[0.0195, 0.0356, 0.021 ],
           [0.0356, 0.0808, 0.0407],
           [0.021 , 0.0407, 0.0239]])
    """
    freq = infer_frequency(freq)
    data = np.asarray(data)
    t, n, a = data.shape
    y = t // freq

    # convert to annual returns then calculate the average covariance across time
    ar = (data + 1).reshape(y, freq, n, a).prod(1) - 1
    return np.mean([np.cov(ar[i], rowvar=False) for i in range(y)], 0)


def tracking_error_m(cov_or_data: np.ndarray,
                     weights: Vector,
                     bmk_weights: Vector,
                     freq: Frequency) -> float:
    """
    Calculates the tracking error with respect to the benchmark.

    If a covariance matrix is used as the data, the benchmark components must be placed after the
    portfolio components. If a simulated returns cube is used as the data, the benchmark components
    must be placed after the portfolio components.

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    bmk_weights
        Weights of the benchmark portfolio.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    Returns
    -------
    float:
        Tracking error of the portfolio

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov, tracking_error_m

    >>> data = load_cube()
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> bmk_weights = [0.65, 0.35]
    >>> freq = 'quarterly'
    >>> tracking_error_m(data, weights, bmk_weights, freq)
    0.031183281273726802
    """
    weights = np.ravel(weights)
    bmk_weights = np.ravel(bmk_weights)
    w = np.hstack((weights, (-1) * bmk_weights))

    cov = _get_covariance_matrix(cov_or_data, freq)
    return (w @ cov @ w.T) ** 0.5


def vol_attr(cov_or_data: np.ndarray,
             weights: Vector,
             freq: Frequency) -> Attribution:
    """
    Derives the volatility attribution given a data cube and weights.

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    Returns
    -------
    Attribution
        A named tuple of relative and absolute volatility attribution respectively. The absolute attribution
        is the volatility of the simulated data over time multiplied by the relative attribution.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov, vol_attr

    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = 'quarterly'
    >>> attr = vol_attr(data, weights, freq)
    >>> attr.marginal.round(4)
    array([0.2352, 0.5006, 0.2643])
    >>> attr.percentage.round(4)
    array([0.0445, 0.0947, 0.05  ])
    """
    weights = np.ravel(weights)
    cov = _get_covariance_matrix(cov_or_data, freq)
    vol = volatility_m(cov, weights)
    attr = (weights * (cov @ weights)) / (weights @ cov @ weights.T)

    return Attribution(attr, vol * attr)


def volatility_m(cov_or_data: np.ndarray,
                 weights: Vector,
                 freq: Frequency = None) -> float:
    """
    Calculates the portfolio volatility given a simulated returns cube or a covariance matrix

    Notes
    -----
    If a simulated returns data cube is given, the frequency of the data must be specified.
    In this case, the empirical covariance matrix would be used to derive the volatility.

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the
        covariance matrix shape or the simulated data's last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    Returns
    -------
    float
        Portfolio volatility.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import portfolio_cov, volatility_m

    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = 'quarterly'
    >>> cov_mat = portfolio_cov(data, freq).round(4)  # empirical covariance matrix

    >>> # Using covariance matrix
    >>> volatility_m(cov_mat, weights)
    0.1891091219375734

    >>> # Using the simulated returns data cube
    >>> volatility_m(data, weights, freq)
    0.1891091219375734
    """
    s = _portfolio_volatility_structure(cov_or_data, weights, freq)
    return s[0, 0] ** 0.5


def _get_covariance_matrix(cov_or_data: np.ndarray, freq: Frequency):
    """
    Helper to derive a covariance matrix.

    If simulated cube is put in, derive empirical covariance matrix. Otherwise, check that matrix
    is a covariance matrix and return it.
    """

    assert cov_or_data.ndim in (2, 3), "only 2D covariance matrix or a 3D simulation returns tensor is allowed"
    cov_or_data = np.asarray(cov_or_data)
    n = cov_or_data.shape[-1]  # number of assets must be last dimension of cube or covariance matrix

    if cov_or_data.ndim == 3:
        # if 3D, get empirical covariance matrix
        assert n == cov_or_data.shape[2], "number of weights does not match simulation cube asset quantity"
        assert freq is not None, "frequency must be specified if deriving covariance structure from simulated data"

        cov_or_data = portfolio_cov(cov_or_data, freq)

    assert is_psd(cov_or_data), "covariance matrix is not positive semi-definite"
    return cov_or_data  # this will be a covariance matrix


def _portfolio_volatility_structure(cov_or_data: np.ndarray,
                                    weights: Vector,
                                    freq: Frequency = None,
                                    aid: int = 0) -> np.ndarray:
    """
    Derive the portfolio volatility structure

    Parameters
    ----------
    cov_or_data
        Covariance matrix or simulated returns data cube.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the
        covariance matrix shape or the simulated data's last axis.

    freq
        Frequency of the data. Can either be a string ('week', 'month', 'quarter', 'semi-annual', 'year') or
        an integer specifying the number of units per year. Week: 52, Month: 12, Quarter: 4, Semi-annual: 6,
        Year: 1.

    aid
        Asset index

    Returns
    -------
    covariance matrix
        Covariance matrix of portfolio volatility structure
    """
    cov_or_data = np.asarray(cov_or_data)
    weights = np.ravel(weights)
    n = len(weights)

    cov = _get_covariance_matrix(cov_or_data, freq)
    w = np.vstack([weights, np.zeros(n)])

    assert 0 <= aid <= len(w), f"asset id must be between [0, {len(w)}]"
    w[1, aid] = 1
    return w @ cov @ w.T
