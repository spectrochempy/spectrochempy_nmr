"""Qualify the public JEOL fallback against the previous decoder's output."""

from hashlib import sha256
from importlib import import_module
import os
from pathlib import Path
import struct

from nmrglue.fileio import jeol
import numpy as np
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
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    assert reader._read_jeol_raw is jeol.read
    monkeypatch.setattr(reader, "_read_normalized_jeol", lambda path: None)
    dataset = scp.nmr.read_jeol(Path(path))
    assert dataset.shape == shape
    assert (
        sha256(dataset.data.astype("<c16", copy=False).tobytes()).hexdigest() == digest
    )
    assert dataset.origin == "jeol"
    assert dataset.x.units == scp.ur.us
    assert dataset.fft(dim="x").shape == shape


def test_public_jeol_luteolin_float32_fallback(monkeypatch):
    source = os.environ.get("NMR_JEOL_LUTEOLIN_COSY")
    if not source:
        pytest.skip("Set NMR_JEOL_LUTEOLIN_COSY to nmrXiv D1155")
    path = Path(source)
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == (
        "2cceb9c5e19ef6569c113cd6b24ca913b628ccd8dc6eb582b69dee084b6d7dfa"
    )
    y, x = np.ogrid[:512, :5120]
    indices = ((y // 32 * (5120 // 32) + x // 32) * 32 + y % 32) * 32 + x % 32
    (offset,) = struct.unpack_from(">I", raw, 1284)
    words = np.frombuffer(raw, "<f4", count=2 * indices.size, offset=offset)
    real, imag = words.reshape(2, -1)[:, indices]

    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    monkeypatch.setattr(reader, "_read_normalized_jeol", lambda path: None)
    dataset = scp.nmr.read_jeol(path)
    assert dataset.shape == (512, 5120)
    np.testing.assert_array_equal(dataset.data.real, real)
    np.testing.assert_array_equal(-dataset.data.imag, imag)
    np.testing.assert_allclose(
        dataset.y.data, np.arange(512) * 227.38, rtol=0, atol=1e-8
    )
    np.testing.assert_allclose(
        dataset.x.data, np.arange(5120) * 181.76, rtol=0, atol=1e-8
    )
    assert dataset.y.units == dataset.x.units == scp.ur.us
    assert dataset.meta.normalization_state == "storage-decoded"
