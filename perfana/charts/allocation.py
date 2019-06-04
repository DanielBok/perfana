from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from ._plot import plot
from ._type import GRAPH_OUTPUT

__all__ = ['allocation_chart']


def allocation_chart(weights: np.ndarray,
                     asset_names: Optional[List[str]] = None,
                     portfolio_names: Optional[List[str]] = None) -> GRAPH_OUTPUT:
    """
    Plots the allocation chart

    Parameters
    ----------
    weights
        Portfolio weights. If multiple portfolios are compared, rows (axis 0) must
        represent the assets and columns (axis 1) must represent the portfolio

    asset_names
        The asset class names

    portfolio_names
        The portfolio names

    Returns
    -------
    Graph Description
        The allocation chart object if on a python console or Jupyter notebook.
        If used in app, returns the dictionary to form the graphs on the frontend
    DataFrame
        The underlying dataset in a dataframe
    """
    weights = np.asarray(weights)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)

    # checks
    assert np.alltrue((weights >= 0) & (weights <= 1)), "all weights must be between [0, 1]"

    n_assets, n_port = weights.shape
    if asset_names is None:
        asset_names = [f"Asset {i + 1}" for i in range(n_assets)]

    if portfolio_names is None:
        portfolio_names = [f"Portfolio {i + 1}" for i in range(n_port)]

    assert len(asset_names) == n_assets, "asset_names length does not match the number of assets"
    assert len(portfolio_names) == n_port, "portfolio_names length does not match the number of portfolios"

    data = [go.Bar(
        x=portfolio_names,
        y=w,
        name=a,
    ) for w, a in zip(weights, asset_names)]

    layout = go.Layout(
        title="Allocation Chart",
        barmode="stack",
        xaxis={
            "title": "Portfolio"
        },
        yaxis={
            "title": "Asset"
        }
    )

    fig = go.Figure(data, layout)
    table = pd.DataFrame({p: weights[:, i] for i, p in enumerate(portfolio_names)}, index=asset_names)

    return plot(fig), table
