"""Qualify JEOL time axes and section order on authentic JDF data."""

from hashlib import sha256
import os
from pathlib import Path
import struct

from nmrglue import interop
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.processing.fft_validation import _require_fft_prevalidation


@pytest.mark.parametrize(
    "name, shape, count, dwell, nominal_y, digest",
    [
        (
            "epicatechin_cosy",
            (256, 1280),
            2,
            (166.74e-6, 133.12e-6),
            256,
            "f109351f2102435faa7d302659e073b77ecc1184f1c3ef8817087ca91b9df3aa",
        ),
        (
            "epicatechin_hsqc",
            (32, 1024),
            4,
            (58.48e-6, 133.44e-6),
            128,
            "eaa28b4b1fe05f95a41ace4f69f33ee6f31f26f392d3aea1d94bf3978b153625",
        ),
        (
            "epicatechin_hmbc",
            (64, 2048),
            4,
            (39.76e-6, 133.44e-6),
            256,
            "578db932e166b04beeb0e9a5db443ff7a042b50e24af2789e535c06c1f6af859",
        ),
        (
            "rutin_hsqc",
            (256, 4096),
            4,
            (58.48e-6, 181.76e-6),
            256,
            "6918f30526488f2657f627af880ca62770d834b0939b2b8f9c00c83005889342",
        ),
    ],
)
def test_public_jeol_2d_interop(
    name, shape, count, dwell, nominal_y, digest, monkeypatch
):
    env = "NMR_JEOL_RUTIN_HSQC" if name == "rutin_hsqc" else "NMR_JEOL_JDF4_DIR"
    source = os.environ.get(env)
    if not source:
        pytest.skip(f"Set {env} to the pinned JEOL qualification data")
    try:
        _require_fft_prevalidation()
    except NotImplementedError as exc:
        pytest.skip(str(exc))

    path = Path(source) if name == "rutin_hsqc" else Path(source) / f"{name}.jdf"
    raw = path.read_bytes()
    assert sha256(raw).hexdigest() == digest
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    index = ((y // 32 * (nx // 32) + x // 32) * 32 + y % 32) * 32 + x % 32
    (start,) = struct.unpack_from(">I", raw, 1284)
    sections = np.frombuffer(raw, "<f8", count=count * ny * nx, offset=start)
    sections = sections.reshape(count, -1)[:, index]
    branches = sections[::2] + 1j * sections[1::2]

    dataset = scp.nmr.read(
        path, protocol="interop", normalize_options={"conjugation": "none"}
    )
    historic = scp.nmr.read_jeol(path)
    assert dataset.meta.acquisition["nominal_shape"] == (nominal_y, nx)
    assert dataset.meta.encoding[-2] == historic.meta.encoding[-2] == "UNVALIDATED:F1"
    assert (dataset.y.units, dataset.x.units) == (scp.ur.s, scp.ur.s)
    np.testing.assert_allclose(
        dataset.x.data, np.arange(nx) * dwell[1], rtol=0, atol=1e-15
    )

    listed = nominal_y != ny
    sampling = "listed" if listed else "regular"
    assert dataset.meta.acquisition["sampling"] == (sampling, "regular")
    if listed:
        (list_start,) = struct.unpack_from(">I", raw, 1224)
        (list_len,) = struct.unpack_from(">I", raw, 1256)
        assert list_len == 8 * ny
        original_list = raw[list_start : list_start + list_len]
        times = np.frombuffer(original_list, ">f8") * 1e-3
        np.testing.assert_array_equal(dataset.y.data, times)
        np.testing.assert_array_equal(
            dataset.meta.source_metadata["axis_lists_raw"]["y"],
            np.frombuffer(original_list, np.uint8),
        )
    else:
        times = np.arange(ny) * dwell[0]
        np.testing.assert_allclose(dataset.y.data, times, rtol=0, atol=1e-15)

    if count == 4:
        assert dataset.dims == ["c", "y", "x"]
        assert dataset.meta.component_labels == ("section-0/1", "section-2/3")
        np.testing.assert_array_equal(dataset.data, branches)
        assert historic.meta.acquisition["sampling"] == (
            f"{sampling}-paired",
            "regular",
        )
        np.testing.assert_array_equal(historic.data[::2], branches[0].conj())
        np.testing.assert_array_equal(historic.data[1::2], -branches[1].conj())
        times = np.repeat(times, 2)
    else:
        assert dataset.dims == ["y", "x"]
        np.testing.assert_array_equal(dataset.data, branches[0])
        np.testing.assert_array_equal(historic.data, branches[0].conj())
    np.testing.assert_allclose(historic.y.to("s").data, times, rtol=0, atol=1e-15)

    def provider_unavailable(*args, **kwargs):
        pytest.fail("Processing must use the imported dataset")

    monkeypatch.setattr(interop, "read", provider_unavailable)
    monkeypatch.setattr(interop, "normalize", provider_unavailable)
    with pytest.raises(NotImplementedError, match="UNVALIDATED:F1"):
        dataset.fft(dim="y")
    f2 = dataset.fft(dim="x")
    assert f2.shape == dataset.shape
    with pytest.raises(NotImplementedError, match="UNVALIDATED:F1"):
        f2.fft(dim="y")
