import os

import numpy as np
import pandas as pd
import pytest

from perfana.datasets import load_cube


@pytest.fixture
def cube_a() -> np.ndarray:
    """Assets"""
    return load_cube()[..., :7]


@pytest.fixture
def cube_b() -> np.ndarray:
    """Benchmark, reference"""
    return load_cube()[..., 7:]


@pytest.fixture
def expected():
    fp = os.path.join(os.path.dirname(__file__), "data", "expected.txt")
    df = pd.read_csv(fp, sep='\t', index_col=0)
    res = {}

    for key, row in df.iterrows():
        a, b = row
        res[key] = a if np.isnan(b) else (a, b)

    return res


@pytest.fixture
def weights():
    return np.array([0.25, 0.18, 0.13, 0.11, 0.24, 0.05, 0.04])
