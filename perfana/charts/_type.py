from typing import Optional, Tuple, Union

import pandas as pd

__all__ = ['GRAPH_DESC', 'GRAPH_OUTPUT']

GRAPH_DESC = Optional[Union[dict, str]]
GRAPH_OUTPUT = Tuple[GRAPH_DESC, pd.DataFrame]
