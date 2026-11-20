"""Opt-in qualification of the common QF workflow on public CENAPTNMR data."""

from hashlib import sha256
import os
from pathlib import Path

from nmrglue import jeol
import numpy as np
import pytest
import spectrochempy as scp


def test_epicatechin_common_qf_signed_fft_and_export():
    root = os.environ.get("NMR_JEOL_COMMON_DIR")
    if not root:
        pytest.skip("Set NMR_JEOL_COMMON_DIR to the pinned epicatechin JDF/JDX pair")
    root = Path(root)
    (path,) = root.glob("*EpiCatechin*COSY*.jdf")
    (reference_path,) = root.glob("*EpiCatechin*COSY*.jdx")
    assert (
        sha256(path.read_bytes()).hexdigest()
        == "f109351f2102435faa7d302659e073b77ecc1184f1c3ef8817087ca91b9df3aa"
    )
    assert (
        sha256(reference_path.read_bytes()).hexdigest()
        == "b77cd1fe72cea1f6d820e40279d91cfdff3424333ca4d669e89b129c566c1152"
    )
    before = path.read_bytes()
    _, stored_fid = jeol.read(path)
    raw = scp.nmr.read(path, quadrature="qf")
    np.testing.assert_array_equal(raw.data, stored_fid)
    weighted = raw.sp(ssb=0, pow=2, dim="x").sp(ssb=0, pow=2, dim="y")
    spectrum = scp.nmr.Experiment(weighted).process(size=(2048, 2048))

    # Independent negative-sign FFT2 oracle, converted to the common QF signs.
    n1, n2 = stored_fid.shape
    windows = np.sin(np.pi * np.arange(n1) / (n1 - 1))[:, None] ** 2
    windows = windows * np.sin(np.pi * np.arange(n2) / (n2 - 1)) ** 2
    oracle = np.fft.fftshift(np.fft.fft2(stored_fid * windows, s=(2048, 2048)))
    oracle = oracle[:, (-np.arange(2048)) % 2048].conj()
    oracle = np.roll(oracle, -1, axis=0)
    np.testing.assert_allclose(spectrum.data, oracle, rtol=1e-11, atol=1e-9)

    reference = scp.nmr.read(
        reference_path, protocol="interop", profile="jcampdx-spectrum-2d"
    )
    result = scp.nmr.Experiment(abs(spectrum)).symmetrize(
        frequency_grid=(reference.y.data, reference.x.data)
    )
    np.testing.assert_array_equal(result.y.data, reference.y.data)
    np.testing.assert_array_equal(result.x.data, reference.x.data)
    assert np.corrcoef(result.data.ravel(), reference.data.ravel())[0, 1] > 0.94
    i = np.flatnonzero(abs(result.y.data - 2269.92) < 17.2)
    j = np.flatnonzero(abs(result.x.data - 2338.07) < 5.4)
    a, b = np.unravel_index(result.data[np.ix_(i, j)].argmax(), (i.size, j.size))
    assert abs(result.y.data[i[a]] - 2269.92) <= 6.0
    assert abs(result.x.data[j[b]] - 2338.07) <= 6.0
    assert path.read_bytes() == before


def test_luteolin_float32_common_workflow():
    root = os.environ.get("NMR_JEOL_FLOAT32_DIR")
    if not root:
        pytest.skip("Set NMR_JEOL_FLOAT32_DIR to the pinned luteolin JDF/JDX pair")
    root = Path(root)
    (path,) = root.glob("*Luteolin*COSY*.jdf")
    (reference_path,) = root.glob("*Luteolin*COSY*.jdx")
    assert (
        sha256(path.read_bytes()).hexdigest()
        == "2cceb9c5e19ef6569c113cd6b24ca913b628ccd8dc6eb582b69dee084b6d7dfa"
    )
    assert (
        sha256(reference_path.read_bytes()).hexdigest()
        == "37683c4ca03ba3a10e4f0a6cfc6103dd0cad859f4ab0381d3903e62dd9a44b09"
    )
    metadata, fid = jeol.read(path)
    assert metadata["header"]["data_type"] == "float32"
    raw = scp.nmr.read(path, quadrature="qf")
    np.testing.assert_array_equal(raw.data, fid)
    weighted = raw.sp(ssb=0, pow=2, dim="x").sp(ssb=0, pow=2, dim="y")
    spectrum = scp.nmr.Experiment(weighted).process(size=(512, 8192))
    reference = scp.nmr.read(
        reference_path, protocol="interop", profile="jcampdx-spectrum-2d"
    )
    result = scp.nmr.Experiment(abs(spectrum)).symmetrize(
        frequency_grid=(reference.y.data, reference.x.data)
    )
    assert result.shape == (512, 6554)
    assert np.corrcoef(result.data.ravel(), reference.data.ravel())[0, 1] > 0.92
    i = np.flatnonzero(abs(result.y.data - 2924.11) < 17.2)
    j = np.flatnonzero(abs(result.x.data - 2687.70) < 5.4)
    a, b = np.unravel_index(result.data[np.ix_(i, j)].argmax(), (i.size, j.size))
    assert abs(result.y.data[i[a]] - 2924.11) <= 9.0
    assert abs(result.x.data[j[b]] - 2687.70) <= 5.0
