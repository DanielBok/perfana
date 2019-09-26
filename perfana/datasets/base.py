import numpy as np
import pandas as pd

from ._file_handler import filepath

__all__ = ["load_cube", "load_etf", "load_hist", "load_smi"]


def load_cube(*, download=False) -> np.ndarray:
    """
    Loads a sample Monte Carlo simulation of 9 asset classes.

    The dimension of the cube is 80 * 1000 * 9. The first axis represents the time, the second
    represents the number of trials (simulations) and the third represents each asset class.

    Parameters
    ----------
    download: bool
        If True, forces the data to be downloaded again from the repository. Otherwise, loads the data from the
        stash folder

    Returns
    -------
    ndarray
        A data cube of simulated returns
    """
    return np.load(filepath('cube.npy', download))


def load_etf(*, date_as_index: bool = True, download=False) -> pd.DataFrame:
    """
    Dataset contains prices of 4 ETF ranging from 2001-06-15 to 2019-03-01.

    Parameters
    ----------
    date_as_index:
        If True, sets the first column as the index of the DataFrame

    download: bool
        If True, forces the data to be downloaded again from the repository. Otherwise, loads the data from the
        stash folder

    Returns
    -------
    DataFrame
        A data frame containing the prices of 4 ETF
    """
    fp = filepath('etf.csv', download)

    if date_as_index:
        df = pd.read_csv(fp, index_col=0, parse_dates=[0])
        df.index.name = df.index.name.strip()
    else:
        df = pd.read_csv(fp, parse_dates=[0])

    df.columns = df.columns.str.strip()
    for c in 'VBK', 'BND':
        df[c] = pd.to_numeric(df[c].str.strip())

    return df


def load_hist(*, date_as_index: bool = True, download=False) -> pd.DataFrame:
    """
    Dataset containing 20-years returns data from different asset classes spanning from 1988 to 2019.

    Parameters
    ----------
    date_as_index:
        If True, sets the first column as the index of the DataFrame

    download: bool
        If True, forces the data to be downloaded again from the repository. Otherwise, loads the data from the
        stash folder

    Returns
    -------
    DataFrame
        A data frame containing the prices of 4 ETF
    """
    fp = filepath('hist.csv', download)

    if date_as_index:
        df = pd.read_csv(fp, index_col=0, parse_dates=[0])
        df.index.name = df.index.name.strip()
    else:
        df = pd.read_csv(fp, parse_dates=[0])

    df.columns = df.columns.str.strip()
    return df


def load_smi(*, as_returns=False, download=False) -> pd.DataFrame:
    """
    Dataset contains the close prices of all 20 constituents of the Swiss Market Index (SMI) from
    2011-09-09 to 2012-03-28.

    Parameters
    ----------
    as_returns: bool
        If true, transforms the price data to returns data

    download: bool
        If True, forces the data to be downloaded again from the repository. Otherwise, loads the data from the
        stash folder

    Returns
    -------
    DataFrame
        A data frame of the closing prices of all 20 constituents of the Swiss Market Index
    """

    df = pd.read_csv(filepath('smi.csv', download), index_col=0, parse_dates=[0])
    if as_returns:
        df = df.pct_change().dropna()
    return df
