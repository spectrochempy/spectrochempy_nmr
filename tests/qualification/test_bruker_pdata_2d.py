"""Qualify the Bruker four-plane bridge against tiled binary words."""

import hashlib
import importlib
import os
from pathlib import Path

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.utils.jsonutils import json_encoder

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


@pytest.fixture
def tiled_pdata(tmp_path):
    experiment = tmp_path / "bruker" / "1"
    path = experiment / "pdata" / "1"
    path.mkdir(parents=True)
    for name in ("acqus", "acqu2s"):
        (experiment / name).write_text(
            "##TITLE= synthetic acquisition\n"
            "##$PARMODE= 1\n##$TD= 12\n##$SW= 24\n##$SFO1= 501\n"
            "##$RG= 3\n##$NS= 9\n##$NUC1= <1H>\n##END=\n"
        )
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
        (path / name).write_text("##TITLE= synthetic pdata\n" + lines + "##END=\n")
    expected = []
    for index, name in enumerate(NAMES):
        plane = np.arange(-12, 12, dtype=np.int32).reshape(4, 6) - 31 * index
        tiles = [plane[y : y + 2, x : x + 3] for y in (0, 2) for x in (0, 3)]
        (path / name).write_bytes(np.stack(tiles).astype(">i4").tobytes())
        expected.append(plane)
    return path, np.stack(expected), axes


@pytest.mark.parametrize("scaled", [False, True])
def test_synthetic_tiled_bruker_planes(provider, scaled, tmp_path, tiled_pdata):
    path, expected, axes = tiled_pdata
    _check_plane(provider, path, scaled, expected, axes, -3, tmp_path)


def _check_public_reader(provider, path, proc_axis, monkeypatch, *, component="2ri"):
    reader = importlib.import_module("spectrochempy_nmr.readers.read_topspin")

    def unsupported(*args, **kwargs):
        raise ValueError("unqualified profile")

    with monkeypatch.context() as legacy:
        legacy.setattr(provider.interop, "read_bruker_pdata_2d", unsupported)
        expected = scp.read_topspin(path / component, proc_axis=proc_axis)

    def reject_legacy(*args, **kwargs):
        pytest.fail("qualified pdata should not use the vendored binary decoder")

    with monkeypatch.context() as migrated:
        migrated.setattr(reader, "read_pdata", reject_legacy)
        actual = scp.read_topspin(path / component, proc_axis=proc_axis)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.dims == expected.dims
    assert actual.units == expected.units
    assert actual.origin == expected.origin
    assert actual.name == expected.name
    assert json_encoder(actual.meta) == json_encoder(expected.meta)
    np.testing.assert_array_equal(actual.data, expected.data)
    for dim in ("c", "y", "x"):
        got, want = actual.coordset[dim], expected.coordset[dim]
        np.testing.assert_array_equal(got.data, want.data)
        np.testing.assert_array_equal(got.labels, want.labels)
        assert got.units == want.units
        assert got.title == want.title
        assert json_encoder(got.meta) == json_encoder(want.meta)
    return actual


@pytest.mark.parametrize("proc_axis", [False, True])
def test_synthetic_tiled_bruker_public_reader(
    provider, tiled_pdata, proc_axis, monkeypatch
):
    path, expected, _ = tiled_pdata
    result = _check_public_reader(provider, path, proc_axis, monkeypatch)
    rr, ri, ir, ii = expected
    channels = np.stack((rr + 1j * ir, ri + 1j * ii)) * 2.0**-3 / 3
    np.testing.assert_array_equal(result.data, channels)
    assert tuple(result.c.labels) == ("S_R", "S_I")


@pytest.mark.parametrize("proc_axis", [False, True])
def test_local_bruker_public_reader(provider, proc_axis, monkeypatch):
    configured = os.environ.get("NMR_PDATA_2D_REFERENCE")
    if not configured:
        pytest.skip("configure the checksum-pinned local 2D pdata")
    path = Path(configured)
    for name, digest in REFERENCE_HASHES.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    _check_public_reader(provider, path, proc_axis, monkeypatch)


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
