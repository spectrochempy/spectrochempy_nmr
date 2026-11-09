# ======================================================================================
# Vendored NMRGlue reading functions (BSD 3-Clause, Jonathan J. Helmus).
#
# This package contains remaining read-only subsets of nmrglue.
# See individual submodules for attribution.
# ======================================================================================

from ._base import create_blank_udic
from ._base import uc_from_udic
from ._simpson import read as read_simpson

__all__ = [
    "create_blank_udic",
    "read_simpson",
    "uc_from_udic",
]
