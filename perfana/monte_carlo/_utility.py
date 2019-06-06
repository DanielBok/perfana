from ._types import Frequency

__all__ = ['infer_frequency']


def infer_frequency(f: Frequency) -> int:
    assert isinstance(f, (str, int)), "Frequency must be either a string"

    if isinstance(f, str):
        f = f.lower()
        if f in ('week', 'weekly'):
            return 52
        if f in ('month', 'monthly'):
            return 12
        elif f in ('quarter', 'quarterly'):
            return 4
        elif f in ('semi-annual', 'semi-annually', 'semi-year', 'semi-yearly', 'half-year', 'half-yearly'):
            return 2
        elif f in ('annual', 'annually', 'year', 'yearly'):
            return 1

    assert f in (1, 2, 4, 12), \
        f"Unknown time frequency: {f}. For reference, 1 - year, 2 - semi-annual, 4 - quarter, 12 - month, 52: week"

    return f
