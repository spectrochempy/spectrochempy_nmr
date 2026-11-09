# ======================================================================================
# Vendored NMRGlue reading functions (BSD 3-Clause, Jonathan J. Helmus).
#
# This package contains remaining read-only subsets of nmrglue.
# See individual submodules for attribution.
# ======================================================================================

from ._base import create_blank_udic
from ._base import uc_from_udic
from ._simpson import read as read_simpson
from ._tecmag import guess_udic as guess_udic_tecmag
from ._tecmag import read as read_tecmag

__all__ = [
    "create_blank_udic",
    "guess_udic_tecmag",
    "read_simpson",
    "read_tecmag",
    "uc_from_udic",
]
