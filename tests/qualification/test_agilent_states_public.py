"""Qualify the public Agilent 2D reader against its compatibility path."""

import hashlib
import os
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from nmrglue import interop
from nmrglue import varian

import spectrochempy as scp
import spectrochempy_nmr.interop as adapter


def _check_public_2d(path, monkeypatch, profile="states"):
    convert = adapter.from_interop
    calls = []

    def spy(result):
        dataset = convert(result)
        calls.append(dataset)
        return dataset

    def reject_fallback(*args, **kwargs):
        pytest.fail("qualified 2D acquisition called the historical decoder")

    reader = import_module("spectrochempy_nmr.readers.read_agilent")
    with monkeypatch.context() as baseline:
        baseline.setattr(interop, f"normalize_agilent_{profile}_2d", None)
        old = scp.read_agilent(path, origin="qualification")

    monkeypatch.setattr(adapter, "from_interop", spy)
    monkeypatch.setattr(reader, "read_varian", reject_fallback)
    actual = scp.read_agilent(path, origin="qualification")
    assert len(calls) == 1 and actual is calls[0]
    assert actual.dims == old.dims
    assert actual.dtype == old.dtype
    assert actual.origin == old.origin == "qualification"
    assert actual.units == old.units
    assert actual.meta.normalization_state == "storage-normalized"
    for key, expected in old.meta.items():
        if key != "normalization_state":
            np.testing.assert_equal(actual.meta[key], expected, err_msg=key)
    np.testing.assert_array_equal(actual.data, old.data)
    for dim in actual.dims:
        got, expected = actual.coordset[dim], old.coordset[dim]
        assert got.title == expected.title
        assert got.units == expected.units
        assert got.meta == expected.meta
        np.testing.assert_array_equal(got.data, expected.data)
        np.testing.assert_array_equal(got.labels, expected.labels)

    direct, baseline = actual.fft(), old.fft()
    np.testing.assert_array_equal(direct.data, baseline.data)
    np.testing.assert_array_equal(direct.x.data, baseline.x.data)
    assert direct.x.units == baseline.x.units
    assert actual.meta.data_domain == "time"
    assert direct.meta.data_domain == "mixed"
    for current, historical, domains in (
        (actual, old, ("time", "time")),
        (direct, baseline, ("time", "frequency")),
    ):
        experiment = scp.nmr.Experiment(current)
        previous = scp.nmr.Experiment(historical)
        assert experiment.ndim == previous.ndim == 2
        assert experiment.domains == previous.domains == domains
        assert experiment.encoding == previous.encoding
        assert experiment.nuclei == previous.nuclei
        assert experiment.source_kind == previous.source_kind
    with pytest.raises(
        NotImplementedError, match=f"UNVALIDATED:AGILENT-{profile.upper()}"
    ):
        direct.fft(dim="y")
    if profile == "tppi":
        assert scp.nmr.Experiment(actual[0].squeeze()).domains == (
            scp.nmr.Experiment(old[0].squeeze()).domains
        )
    return actual


@pytest.fixture
def synthetic_states(tmp_path, request):
    tppi = getattr(request, "param", "states") == "tppi"
    params = {
        "array": [""] if tppi else ["phase"],
        "phase": ["3"] if tppi else ["1", "2"],
        "ni": ["4"] if tppi else ["2"],
        "arraydim": ["4"],
        "np": ["8"],
        "sw": ["10775.862069"],
        "sw1": ["500"],
        "sw2": ["831.91923"],
        "sfrq": ["125"],
        "dfrq2": ["50"],
        "ns": ["3"],
        "ne": ["2"],
    }
    procpar = {
        key: varian.create_pdic_param(key, values) for key, values in params.items()
    }
    procpar["array"]["basictype"] = "2"
    varian.write_procpar(tmp_path / "procpar", procpar)
    words = np.arange(32, dtype=">i4").reshape(4, 8)
    words[:, 1::2] *= -7
    with (tmp_path / "fid").open("wb") as stream:
        varian.put_fileheader(stream, (4, 1, 8, 4, 32, 60, 0, 21, 1))
        for index, trace in enumerate(words, 1):
            varian.put_blockheader(stream, (0, 21, index, 0, 3, 0, 0, 0, 0))
            stream.write(trace.tobytes())
    return tmp_path / "fid", words[:, ::2] + 1j * words[:, 1::2]


@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_states_2d"),
    reason="provider does not support States normalization",
)
def test_synthetic_states_public(synthetic_states, monkeypatch):
    path, samples = synthetic_states
    actual = _check_public_2d(path, monkeypatch)
    np.testing.assert_array_equal(actual.data[0], samples[::2] / 6)
    np.testing.assert_array_equal(actual.data[1], samples[1::2] / 6)
    scales = [
        operation
        for operation in actual.meta.applied_operations
        if operation["name"] == "normalize-agilent-ns-ne"
    ]
    assert len(scales) == 1 and scales[0]["divisor"] == 6


def test_synthetic_states_older_provider(synthetic_states, monkeypatch):
    path, samples = synthetic_states
    monkeypatch.delattr(interop, "normalize_agilent_states_2d", raising=False)
    actual = scp.read_agilent(path)
    assert actual.meta.normalization_state == "storage-decoded"
    np.testing.assert_array_equal(actual.data[0], samples[::2] / 6)
    np.testing.assert_array_equal(actual.data[1], samples[1::2] / 6)


@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_states_2d"),
    reason="provider does not support States normalization",
)
def test_synthetic_imaging_topology_uses_compatibility(synthetic_states, monkeypatch):
    path, _ = synthetic_states
    procpar_path = path.with_name("procpar")
    procpar = varian.read_procpar(procpar_path)
    procpar.update(
        seqcon=varian.create_pdic_param("seqcon", ["nnnnn"]),
        nv=varian.create_pdic_param("nv", ["2"]),
        nv2=varian.create_pdic_param("nv2", ["2"]),
    )
    varian.write_procpar(procpar_path, procpar, overwrite=True)
    reader = import_module("spectrochempy_nmr.readers.read_agilent")
    legacy = reader.read_varian
    shapes = []

    def spy(*args, **kwargs):
        metadata, data = legacy(*args, **kwargs)
        shapes.append(data.shape)
        return metadata, data

    monkeypatch.setattr(reader, "read_varian", spy)
    assert scp.read_agilent(path) is None
    assert shapes == [(2, 2, 4)]


@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_states_2d"),
    reason="provider does not support States normalization",
)
@pytest.mark.parametrize(
    ("variable", "shape", "width", "checksums"),
    [
        (
            "NMR_AGILENT_STATES_DATA",
            (2, 166, 1500),
            (5555.55555556, 50000.0),
            (
                "4a35f4b98f9116ba67760d1c0c83cc48a17b4e53897b5dbe1e25f85f28b1eac8",
                "54cf12841987ec3a05f5ca76f2b09dff09b7648d84b2936627f2c9757472c817",
            ),
        ),
        (
            "NMR_AGILENT_NOESY_DATA",
            (2, 200, 1616),
            (10775.862069, 10775.862069),
            (
                "730290aaa33a1d7c7d394ec0d63af01a6a32c01c39819ffcce0a1b0dc677cef6",
                "3d387937d66af98aaf1356893ef0bae7d34d290d4fa32920374eb88c8bcbc471",
            ),
        ),
    ],
)
def test_real_states_public(monkeypatch, variable, shape, width, checksums):
    directory = os.environ.get(variable)
    if not directory:
        pytest.skip(f"set {variable} to the real acquisition directory")
    path = Path(directory)
    files = (path / "fid", path / "procpar")
    before = tuple(hashlib.sha256(file.read_bytes()).hexdigest() for file in files)
    assert before == checksums
    actual = _check_public_2d(path, monkeypatch)
    assert actual.shape == shape
    assert actual.y.units is None
    assert actual.meta.acquisition["spectral_widths_hz"] == width
    assert actual.meta.interop_axes[0]["reference"] is None
    assert before == tuple(
        hashlib.sha256(file.read_bytes()).hexdigest() for file in files
    )


@pytest.mark.parametrize("synthetic_states", ["tppi"], indirect=True)
@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_tppi_2d"),
    reason="provider does not support TPPI normalization",
)
def test_synthetic_tppi_public(synthetic_states, monkeypatch):
    path, samples = synthetic_states
    actual = _check_public_2d(path, monkeypatch, "tppi")
    np.testing.assert_array_equal(actual.data, samples / 6)
    scales = [
        operation
        for operation in actual.meta.applied_operations
        if operation["name"] == "normalize-agilent-ns-ne"
    ]
    assert len(scales) == 1 and scales[0]["divisor"] == 6


@pytest.mark.parametrize("synthetic_states", ["tppi"], indirect=True)
def test_synthetic_tppi_older_provider(synthetic_states, monkeypatch):
    path, samples = synthetic_states
    monkeypatch.delattr(interop, "normalize_agilent_tppi_2d", raising=False)
    actual = scp.read_agilent(path)
    assert actual.meta.encoding == ["UNVALIDATED:AGILENT-TPPI", "QSIM"]
    assert actual.meta.get("normalization_state") is None
    np.testing.assert_array_equal(actual.data, samples / 6)


@pytest.mark.parametrize("synthetic_states", ["tppi"], indirect=True)
@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_tppi_2d"),
    reason="provider does not support TPPI normalization",
)
def test_synthetic_tppi_secondary_dimension_uses_compatibility(
    synthetic_states, monkeypatch
):
    path, _ = synthetic_states
    procpar_path = path.with_name("procpar")
    procpar = varian.read_procpar(procpar_path)
    procpar["ni2"] = varian.create_pdic_param("ni2", ["0"])
    procpar["phase2"] = varian.create_pdic_param("phase2", ["1"])
    varian.write_procpar(procpar_path, procpar, overwrite=True)

    def reject_normalized(result):
        pytest.fail("ambiguous indirect dimensions reached from_interop")

    monkeypatch.setattr(adapter, "from_interop", reject_normalized)
    assert scp.read_agilent(path) is None


@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_tppi_2d"),
    reason="provider does not support TPPI normalization",
)
def test_real_tppi_public(monkeypatch):
    directory = os.environ.get("NMR_AGILENT_TPPI_DATA")
    if not directory:
        pytest.skip("set NMR_AGILENT_TPPI_DATA to the real TPPI acquisition directory")
    path = Path(directory)
    files = (path / "fid", path / "procpar")
    checksums = (
        "db3a4b42140aec536917b5fe4369d1a517dd9208c2c1cc88d8e17f6ae1d0ee13",
        "cbe683a6463bdc34b389aedc18438edf32968984f5f8599da2ed82cef73b0199",
    )
    assert (
        tuple(hashlib.sha256(file.read_bytes()).hexdigest() for file in files)
        == checksums
    )
    actual = _check_public_2d(path, monkeypatch, "tppi")
    assert actual.shape == (600, 1400)
    assert actual.y.units is None
    assert actual.meta.acquisition["spectral_widths_hz"] == (66666.6666667, 50000.0)
    assert (
        tuple(hashlib.sha256(file.read_bytes()).hexdigest() for file in files)
        == checksums
    )
