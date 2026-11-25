"""Native reference intensity scaling must not become a fitted JCAMP scale."""

import runpy
from pathlib import Path

import pytest


_scale = runpy.run_path(
    str(Path(__file__).parents[1] / "tools/qualify_public_tppi.py")
)["_native_raw_scale"]


@pytest.mark.parametrize(
    "dtype,nc,expected", [(0, -2, 0.25), (0, 0, 1), (0, 2, 4), (2, 0, 1)]
)
def test_native_reference_nc_scale(dtype, nc, expected):
    assert _scale({"DTYPA": dtype, "NC": nc}) == (nc, expected)


@pytest.mark.parametrize(
    "dtype,nc",
    [(0, None), (0, True), (0, 2.5), (0, 2.0), (0, 1024), (0, -1023), (2, 2), (1, 0)],
)
def test_native_reference_rejects_unqualified_storage_scale(dtype, nc):
    with pytest.raises(ValueError, match="NC scaling"):
        _scale({"DTYPA": dtype, "NC": nc})
