"""Qualify the public JEOL fallback against the previous decoder's output."""

from hashlib import sha256
from importlib import import_module
import os
from pathlib import Path

from nmrglue.fileio import jeol
import pytest
import spectrochempy as scp


@pytest.mark.parametrize(
    "source, shape, digest",
    [
        (
            "NMR_JEOL_REFERENCE",
            (65536,),
            "701fefcf04230a322c500f12ba43472480942d193081f68ded096eb170f8715d",
        ),
        (
            "NMR_JEOL_COSY",
            (256, 1280),
            "604db7fb80709eb8dc496247387f737007f8524692fa65e85e9e4e1d911745af",
        ),
        (
            "NMR_JEOL_HSQC",
            (64, 1024),
            "7988254b85c3013776ce3a4fecdcebaf851c150e96e5b5e6dc3a219175d8efb0",
        ),
    ],
)
def test_public_jeol_uses_released_decoder(source, shape, digest, monkeypatch):
    path = os.environ.get(source)
    if not path:
        pytest.skip(f"Set {source}")
    from spectrochempy_nmr import interop

    def unexpected_conversion(*args):
        pytest.fail("the released provider must use the compatibility reader")

    monkeypatch.setattr(interop, "from_interop", unexpected_conversion)
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    assert reader._read_jeol_raw is jeol.read
    dataset = scp.nmr.read_jeol(Path(path))
    assert dataset.shape == shape
    assert (
        sha256(dataset.data.astype("<c16", copy=False).tobytes()).hexdigest() == digest
    )
    assert dataset.origin == "jeol"
    assert dataset.x.units == scp.ur.us
    assert dataset.fft(dim="x").shape == shape
