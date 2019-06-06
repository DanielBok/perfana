from datetime import datetime
from typing import Iterable, Union

import numpy as np
import pandas as pd

__all__ = ['DateTimes', 'TimeSeriesData', 'Vector']

DateTimes = Union[Iterable[str], Iterable[datetime]]
Vector = Union[Iterable[Union[int, float]], np.ndarray, pd.Series]
TimeSeriesData = Union[pd.DataFrame, Vector]
