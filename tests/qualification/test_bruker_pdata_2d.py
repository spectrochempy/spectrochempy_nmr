"""Qualify the Bruker four-plane bridge against tiled binary words."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop


NAMES = ("2rr", "2ri", "2ir", "2ii")
REFERENCE_HASHES = {
    "procs": "6be827f1dc8d4a17745d4fec105b4d0fc272a66c9018db98a718341ff53a4a05",
    "proc2s": "0198df8e261f344dcf84e15738e9f38b9a16079937b27b4376ad5ae1238392ef",
    "2rr": "dde508a1fc1d3534dee73aea15764d54d0d05ddcaebfc8d6242d747e8dc88241",
    "2ri": "7b13cbaa813a71166cd487656efbecacbdac6510c9657671c7c1e9a41a9e1492",
    "2ir": "2664c4092aef51f5b360e67db5d5a59b4b19fecb8521a02d1d6f147016eb9d54",
    "2ii": "7b9a2f8a18f946cfb1d7504e52d67da1fd591ebcdbe122de1e7038f76d608e85",
}


@pytest.fixture(scope="module")
def provider():
    return pytest.importorskip("nmrglue", reason="optional external provider")


def _check_plane(provider, path, scaled, expected, axes, exponent, tmp_path):
    stored = provider.interop.read_bruker_pdata_2d(path, scale_data=scaled)
    dataset = from_interop(provider.interop.normalize(stored))
    if scaled:
        expected = expected.astype(np.float64) * 2.0**exponent
    assert dataset.dtype == expected.dtype
    np.testing.assert_array_equal(dataset.data, expected)
    assert tuple(dataset.c.labels) == NAMES
    assert dataset.meta.axis_roles == ("component", "F1", "F2")
    assert dataset.meta.interop_encoding == ("stored", "stored")
    scaling = {"scale_data": scaled}
    if scaled:
        scaling["NC_proc"] = exponent
    assert dataset.meta.acquisition["storage_scaling"] == scaling
    for coord, (sf, width, offset), size in zip(
        (dataset.y, dataset.x), axes, expected.shape[1:], strict=True
    ):
        hz = offset * sf - np.arange(size) * (width / size)
        assert coord.units == scp.ur.Hz
        np.testing.assert_allclose(coord.data, hz, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(
            coord.to("ppm").data, hz / sf, rtol=1e-12, atol=1e-10
        )
    target = tmp_path / "four-bruker-planes.scp"
    dataset.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, expected)
    assert tuple(restored.c.labels) == NAMES
    assert tuple(restored.meta.axis_roles) == ("component", "F1", "F2")
    for before, after in zip(
        (dataset.y, dataset.x), (restored.y, restored.x), strict=True
    ):
        np.testing.assert_array_equal(after.data, before.data)
        assert after.units == before.units
        assert after.meta.acquisition_frequency == before.meta.acquisition_frequency


@pytest.mark.parametrize("scaled", [False, True])
def test_synthetic_tiled_bruker_planes(provider, scaled, tmp_path):
    axes = ((100, 800, 3), (500, 2400, 7))
    for name, size, tile, (sf, sw, offset) in zip(
        ("proc2s", "procs"), (4, 6), (2, 3), axes, strict=True
    ):
        parameters = dict(
            SI=size,
            XDIM=tile,
            STSI=size,
            STSR=0,
            FTSIZE=size,
            FT_mod=6,
            BYTORDP=1,
            DTYPP=0,
            NC_proc=-3,
            SF=sf,
            SW_p=sw,
            OFFSET=offset,
        )
        lines = "".join(f"##${key}= {value}\n" for key, value in parameters.items())
        (tmp_path / name).write_text("##TITLE= synthetic pdata\n" + lines + "##END=\n")
    expected = []
    for index, name in enumerate(NAMES):
        plane = np.arange(-12, 12, dtype=np.int32).reshape(4, 6) - 31 * index
        tiles = [plane[y : y + 2, x : x + 3] for y in (0, 2) for x in (0, 3)]
        (tmp_path / name).write_bytes(np.stack(tiles).astype(">i4").tobytes())
        expected.append(plane)
    _check_plane(provider, tmp_path, scaled, np.stack(expected), axes, -3, tmp_path)


@pytest.mark.parametrize("scaled", [False, True])
def test_local_bruker_proton_exchange_planes(provider, scaled, tmp_path):
    configured = os.environ.get("NMR_PDATA_2D_REFERENCE")
    if not configured:
        pytest.skip("configure the checksum-pinned local 2D pdata")
    path = Path(configured)

    def check_hashes():
        for name, digest in REFERENCE_HASHES.items():
            assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest

    check_hashes()
    # One full-plane tile per file, little-endian int32, as recorded by procs.
    expected = np.stack(
        [np.fromfile(path / name, dtype="<i4").reshape(512, 4096) for name in NAMES]
    )
    axes = (
        (500.070370850006, 40000.001010485, 45.42811),
        (500.070370847839, 10000.0, 15.43233),
    )
    _check_plane(provider, path, scaled, expected, axes, -7, tmp_path)
    check_hashes()
