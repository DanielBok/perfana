import numpy as np
from copulae.core import cov2corr, is_psd

from perfana.types import Vector
from ._types import Attribution, Drawdown, Frequency, TailLoss
from ._utility import infer_frequency

__all__ = [
    "beta_m",
    "correlation_m",
    "cvar_attr",
    "cvar_m",
    "cvar_div_ratio",
    "diversification_m",
    "drawdown_m",
    "portfolio_cov",
    "prob_loss",
    "prob_under_perf",
    "tail_loss",
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


def cvar_attr(data: np.ndarray,
              weights: Vector,
              alpha=0.95,
              rebalance=True,
              invert=True) -> Attribution:
    """
    Calculates the CVaR (Expected Shortfall) attribution for each asset class in the portfolio.

    Notes
    -----
    From a mathematical point of view, the alpha value (confidence level for calculation)
    should be taken at the negative extreme of the distribution. However, the default is
    set to ease the practitioner.

    The return values are defined as follows:

    - **marginal**
        The absolute marginal contribution of the asset class towards the portfolio CVaR.
        It is essentially the percentage attribution multiplied by the portfolio CVaR.

    - **percentage**
        The percentage contribution of the asset class towards the portfolio CVaR. This number
        though named in percentage is actually in decimals. Thus 0.01 represents a 1% contribution.


    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    alpha
        Confidence level for calculation.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    invert
        Whether to invert the confidence interval level. See Notes.

    Returns
    -------
    Attribution
        A named tuple of relative and absolute CVaR (expected shortfall) attribution respectively.
        The absolute attribution is the CVaR of the simulated data over time multiplied by the
        percentage attribution.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import cvar_attr
    >>> cube = load_cube()[..., :3]
    >>> weights = [0.33, 0.34, 0.33]
    >>> attr = cvar_attr(cube, weights, alpha=0.95)
    >>> attr.marginal
    array([-0.186001  , -0.35758411, -0.20281477])
    >>> attr.percentage
    array([0.24919752, 0.47907847, 0.27172401])
    >>> attr.marginal is attr[0]
    True
    >>> attr.percentage is attr[1]
    True
    """
    data = np.asarray(data)
    port_cvar = cvar_m(data, weights, alpha, rebalance, invert)

    assert isinstance(alpha, float) and 0 <= alpha <= 1, "alpha must be a float between 0 and 1"
    alpha = 1 - alpha if invert else alpha
    t, n, a = data.shape
    k = int(alpha * n)

    if rebalance:
        # calculate the CVaR quantile for each time period for the rebalanced portfolio
        port = data @ weights
        # calculate alpha quantile value at each time period
        quantiles = np.quantile(port, alpha, 1)

        cvar_decomp = np.zeros((t, a))  # cvar decomposition at each time period for each asset class
        for i, q, p, d in zip(range(t), quantiles, port, data):
            # for each time period, get the mean of the values that are less than the quantile for that
            # time period along the trials axis. Multiply resulting vector with the weights
            es = d[p <= q].mean(0) * weights
            cvar_decomp[i] = es / es.sum()

        attr = cvar_decomp.mean(0)
    else:
        data = (data + 1).prod(0) - 1  # portfolio drift

        # sort data according to the portfolio returns then truncate at the alpha value
        sorted_data = data[np.argsort(data @ weights)][:k]

        # take the mean along each trials axis multiplied by weight
        es = sorted_data.mean(0) * weights
        attr = es / es.sum()

    return Attribution(port_cvar * attr, attr)


def cvar_div_ratio(data: np.ndarray,
                   weights: Vector,
                   alpha=0.95,
                   rebalance=True,
                   invert=True) -> float:
    """
    Calculates the CVaR (Expected Shortfall) tail diversification ratio of the portfolio

    Notes
    -----
    From a mathematical point of view, the alpha value (confidence level for calculation)
    should be taken at the negative extreme of the distribution. However, the default is
    set to ease the practitioner.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    alpha
        Confidence level for calculation.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    invert
        Whether to invert the confidence interval level. See Notes.

    Returns
    -------
    float
        CVaR (Expected Shortfall) tail diversification ratio

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import cvar_div_ratio
    >>> cube = load_cube()[..., :3]
    >>> weights = [0.33, 0.34, 0.33]
    >>> cvar_div_ratio(cube, weights)
    0.8965390850633622
    """
    data = np.asarray(data)
    port_cvar = cvar_m(data, weights, alpha, rebalance, invert)

    assert isinstance(alpha, float) and 0 <= alpha <= 1, "alpha must be a float between 0 and 1"
    alpha = 1 - alpha if invert else alpha

    asset_cvar = np.zeros_like(weights)
    for i in range(len(weights)):
        d = (data[..., i] + 1).prod(0) - 1
        q = np.quantile(d, alpha)
        asset_cvar[i] = d[d <= q].mean()

    return (asset_cvar @ weights) / port_cvar


def cvar_m(data: np.ndarray,
           weights: Vector,
           alpha=0.95,
           rebalance=True,
           invert=True):
    """
    Calculates the Conditional Value at Risk (Expected Shortfall) of the portfolio.

    Notes
    -----
    From a mathematical point of view, the alpha value (confidence level for calculation)
    should be taken at the negative extreme of the distribution. However, the default is
    set to ease the practitioner.

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    alpha
        Confidence level for calculation.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    invert
        Whether to invert the confidence interval level. See Notes.

    Returns
    -------
    float
        CVaR (Expected Shortfall) of the portfolio

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import cvar_m
    >>> cube = load_cube()[..., :3]
    >>> weights = [0.33, 0.34, 0.33]
    >>> cvar_m(cube, weights)
    -0.7463998716846179
    """

    data = np.asarray(data)
    assert isinstance(alpha, float) and 0 <= alpha <= 1, "alpha must be a float between 0 and 1"
    alpha = 1 - alpha if invert else alpha

    if rebalance:
        data = (data @ weights + 1).prod(0) - 1
    else:
        data = (data + 1).prod(0) @ weights - 1

    return data[data <= np.quantile(data, alpha)].mean()


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


def drawdown_m(data: np.ndarray,
               weights: Vector,
               geometric=True,
               rebalance=True) -> Drawdown:
    """
    Calculates the drawdown statistics

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    geometric
        If True, calculates the geometric mean, otherwise, calculates the arithmetic mean.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    Drawdown
        A named tuple containing the average maximum drawdown and the drawdown path for each simulation
        instance.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import drawdown_m
    >>> data = load_cube()[..., :7]
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> dd = drawdown_m(data, weights)
    >>> dd.average
    -0.3198714473717889
    >>> dd.paths.shape
    (80, 1000)
    """
    data = np.asarray(data)
    weights = np.ravel(weights)

    # cumulative returns per asset class
    if rebalance:
        if geometric:
            cum_ret = (data @ weights + 1).cumprod(0)
        else:
            cum_ret = (data @ weights).cumsum(0) + 1
    else:
        if geometric:
            cum_ret = ((data + 1).cumprod(0) - 1) @ weights + 1
        else:
            cum_ret = data.cumsum(1) @ weights + 1

    # get cumulative maximum along path
    cum_max = np.maximum.accumulate(np.vstack([np.ones(data.shape[1]), cum_ret]), 0)[1:]
    # drawdown path formula
    dd_paths = cum_ret / cum_max - 1
    average_max_dd = dd_paths.min(0).mean()

    return Drawdown(average_max_dd, dd_paths)


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


def prob_loss(data: np.ndarray,
              weights: Vector,
              rebalance=True,
              terminal=False) -> float:
    """
    Calculates the probability of the portfolio suffering a loss。

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    terminal
        If True, this only compares the probability of a loss at the last stage. If False (default),
        the calculation will take into account if the portfolio was "ruined" and count it as a loss
        even though the terminal value is positive.

    Returns
    -------
    float
        A named tuple containing the probability of underperformance and loss

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import prob_loss
    >>> data = load_cube()
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> prob_loss(data, weights)
    0.198
    """
    weights = np.ravel(weights)
    n = len(weights)

    port_data = data[..., :n]
    if rebalance:
        port_cum_ret = (port_data @ weights + 1).cumprod(0) - 1
    else:
        port_cum_ret = ((port_data + 1).cumprod(0) - 1) @ weights

    loss = np.asarray(port_cum_ret < 0)[-1]

    if terminal:
        return loss.mean()
    else:
        return ((port_cum_ret < -1).any(0) | loss).mean()


def prob_under_perf(data: np.ndarray,
                    weights: Vector,
                    bmk_weights: Vector,
                    rebalance=True,
                    terminal=False) -> float:
    """
    Calculates the probability of the portfolio underperforming the benchmark at the terminal state

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

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    terminal
        If True, this only compares the probability of underperformance at the last stage.
        If False (default), the calculation will take into account if the portfolio was "ruined"
        and count it as an underperformance against the benchmark even though the terminal value
        is higher than the benchmark. If both portfolios are "ruined", then it underperforms if
        it is ruined earlier.

    Returns
    -------
    float
        A named tuple containing the probability of underperformance and loss

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import prob_under_perf
    >>> data = load_cube()
    >>> weights = [0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04]
    >>> bmk_weights = [0.65, 0.35]
    >>> prob_under_perf(data, weights, bmk_weights)
    0.863
    """
    weights, bmk_weights = np.ravel(weights), np.ravel(bmk_weights)
    n = len(weights)

    port_data, bmk_data = data[..., :n], data[..., n:]

    if rebalance:
        port_cum_ret = (port_data @ weights + 1).cumprod(0) - 1
        bmk_cum_ret = (bmk_data @ bmk_weights + 1).cumprod(0) - 1
    else:
        port_cum_ret = ((port_data + 1).cumprod(0) - 1) @ weights
        bmk_cum_ret = ((bmk_data + 1).cumprod(0) - 1) @ bmk_weights

    under_perf = np.asarray(port_cum_ret[-1] < bmk_cum_ret[-1])  # wrapped for autocomplete
    if terminal:
        return under_perf.mean()

    m = data.shape[1]
    for i, prow, brow in zip(range(m), port_cum_ret.T, bmk_cum_ret.T):
        for p, b in zip(prow, brow):
            if b <= -1:
                # can ignore the fact that p <= -1, cause there would be no underperformance either way
                under_perf[i] = False
                break
            elif p <= -1:
                under_perf[i] = True
                break

    return under_perf.mean()


def tail_loss(data: np.ndarray,
              weights: Vector,
              threshold=-0.3,
              rebalance=True) -> TailLoss:
    """
    Calculates the probability and expectation of a tail loss beyond a threshold

    Threshold by default is set at -0.3, which means find the probability that the portfolio loses more than
    30% of its value and the expected loss.

    Notes
    -----
    The return values are defined as follows:

    - **prob**
        Probability of having a tail loss exceeding the threshold

    - **expected_loss**
        Value of the expected loss for the portfolio at the threshold

    Parameters
    ----------
    data
        Monte carlo simulation data. This must be 3 dimensional with the axis representing time, trial
        and asset respectively.

    weights
        Weights of the portfolio. This must be 1 dimensional and must match the dimension of the data's
        last axis.

    threshold
        Portfolio loss threshold.

    rebalance
        If True, portfolio is assumed to be rebalanced at every step.

    Returns
    -------
    TailLoss
        A named tuple containing the probability and expected loss of the portfolio exceeding the threshold.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import tail_loss
    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> weights = [0.33, 0.34, 0.33]
    >>> loss = tail_loss(data, weights, -0.3)
    >>> loss.prob
    0.241
    >>> loss.expected_loss
    -0.3978210273894446
    """
    data = np.asarray(data)
    weights = np.ravel(weights)

    if rebalance:
        port = (data @ weights + 1).prod(0) - 1
    else:
        port = ((data + 1).prod(0) - 1) @ weights

    mask = port <= threshold
    prob = mask.mean()
    exp = 0 if sum(mask) == 0 else port[mask].mean()

    return TailLoss(prob, exp)


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
    w = np.hstack((weights, -bmk_weights))

    cov = _get_covariance_matrix(cov_or_data, freq)
    return (w @ cov @ w.T) ** 0.5


def vol_attr(cov_or_data: np.ndarray,
             weights: Vector,
             freq: Frequency) -> Attribution:
    """
    Derives the volatility attribution given a data cube and weights.

    Notes
    -----
    The return values are defined as follows:

    **marginal**
        The absolute marginal contribution of the asset class towards the portfolio volatility.
        It is essentially the percentage attribution multiplied by the portfolio volatility.

    **percentage**
        The percentage contribution of the asset class towards the portfolio volatility. This number
        though named in percentage is actually in decimals. Thus 0.01 represents a 1% contribution.

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
        is the volatility of the simulated data over time multiplied by the percentage attribution.

    Examples
    --------
    >>> from perfana.datasets import load_cube
    >>> from perfana.monte_carlo import vol_attr
    >>> data = load_cube()[..., :3]  # first 3 asset classes only
    >>> weights = [0.33, 0.34, 0.33]
    >>> freq = 'quarterly'
    >>> attr = vol_attr(data, weights, freq)
    >>> attr.marginal.round(4)
    array([0.2352, 0.5006, 0.2643])
    >>> attr.percentage.round(4)
    array([0.0445, 0.0947, 0.05  ])
    >>> attr.marginal is attr[0]
    True
    >>> attr.percentage is attr[1]
    True
    """
    weights = np.ravel(weights)
    cov = _get_covariance_matrix(cov_or_data, freq)
    vol = volatility_m(cov, weights)
    attr = (weights * (cov @ weights)) / (weights @ cov @ weights.T)

    return Attribution(vol * attr, attr)


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
