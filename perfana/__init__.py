from perfana.pd_ext import PaFrameAccessor, PaSeriesAccessor
from ._version import get_versions

v = get_versions()
__version__ = v.get('closest-tag', v['version']).split('+')[0]

del v, get_versions
