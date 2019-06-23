from typing import NamedTuple, Union

import numpy as np

__all__ = ["Attribution", "Drawdown", "Frequency", "TailLoss"]

Attribution = NamedTuple("Attribution", [
    ("marginal", np.ndarray),
    ("percentage", np.ndarray),
])

Drawdown = NamedTuple("Drawdown", [
    ("average", float),
    ("paths", np.ndarray),
])

Frequency = Union[str, int]

TailLoss = NamedTuple("TailLoss", [
    ("prob", float),
    ("expected_loss", float)
])
