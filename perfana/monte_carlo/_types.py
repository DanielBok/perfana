from typing import NamedTuple, Union

import numpy as np

Attribution = NamedTuple("Attribution",
                         [
                             ("marginal", np.ndarray),
                             ("percentage", np.ndarray),
                         ])

Frequency = Union[str, int]
