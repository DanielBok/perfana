import pandas as pd
import pytest

from ppa.core.utils import *


@pytest.mark.parametrize('freq, expected', [
    (('d', 'day', 'daily'), 252),
    (('w', 'week', 'weekly'), 52),
    (('m', 'month', 'monthly'), 12),
    (('s', 'semi-annual', 'semi-annually'), 6),
    (('q', 'quarter', 'quarterly'), 4),
    (('y', 'year', 'yearly', 'annual'), 1),
    pytest.param(('anything', 123, 0.15, [1, 'asd']), 0, marks=pytest.mark.raises(exception=ValueError))
])
def test_freq_to_scale(freq, expected):
    for f in freq:
        assert freq_to_scale(f) == expected


def test_infer_frequency_raises_errors(etf):
    with pytest.raises(ValueError, match=r'<data> must be a pandas DataFrame or Series'):
        infer_frequency(etf.values)

    with pytest.raises(ValueError, match=r'Unknown <fail_policy>:'):
        infer_frequency(etf, 'WRONG_FAIL_POLICY')

    with pytest.raises(TypeError, match='could not infer periodicity of time series index'):
        df = etf.head(30).copy()
        df.index = [*pd.date_range('2000-01-01', periods=15, freq='Q'),
                    *pd.date_range('2010-01-01', periods=15, freq='D')]

        infer_frequency(df, 'raise')

    # no error
    df = etf.head(29).copy()
    assert infer_frequency(df) == 'daily'