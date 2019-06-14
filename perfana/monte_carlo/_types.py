from typing import NamedTuple, Union

import numpy as np

__all__ = ["Attribution", "Drawdown", "Frequency", "RiskPerf", "TailLoss"]

Attribution = NamedTuple("Attribution", [
    ("marginal", np.ndarray),
    ("percentage", np.ndarray),
])

Drawdown = NamedTuple("Drawdown", [
    ("average", float),
    ("paths", np.ndarray),
])

Frequency = Union[str, int]

RiskPerf = NamedTuple("RiskPerformance", [
    ("prob_under_performance", float),
    ("prob_loss", float),
])

TailLoss = NamedTuple("TailLoss", [
    ("prob", float),
    ("expected_loss", float)
])
