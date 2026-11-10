"""Qualify the public Agilent 2D reader against its compatibility path."""

import hashlib
import os
from importlib import import_module
from inspect import signature
from pathlib import Path

import numpy as np
import pytest
from nmrglue import interop
from nmrglue import varian

import spectrochempy as scp
from spectrochempy.utils.exceptions import SpectroChemPyError
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
        name = "fid_series" if profile == "series" else f"{profile}_2d"
        baseline.setattr(interop, f"normalize_agilent_{name}", None)
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
    if profile != "series":
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


@pytest.mark.skipif(
    "quadrature" not in signature(interop.normalize_agilent_states_2d).parameters,
    reason="provider does not support Agilent quadrature selection",
)
def test_synthetic_states_quadrature_auto_and_manual(synthetic_states, monkeypatch):
    path, _ = synthetic_states
    adapter = import_module("nmrglue.interop.varian")
    procpar_path = path.with_name("procpar")
    procpar = varian.read_procpar(procpar_path)
    procpar["seqfil"] = varian.create_pdic_param("seqfil", ["NOESY"])
    procpar["seqfil"]["basictype"] = "2"
    procpar["d2"] = varian.create_pdic_param("d2", ["0"])
    varian.write_procpar(procpar_path, procpar, overwrite=True)
    source = path.with_name("NOESY.c")
    source.write_bytes(b"verified source fixture")
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        adapter,
        "_STATES_PROGRAMS",
        {("NOESY", fingerprint): ("states-tppi", "reference")},
    )
    auto = scp.nmr.read_agilent(path)
    manual = scp.nmr.read_agilent(path, quadrature="states-tppi")
    assert auto.meta.acquisition["quadrature_selection"]["source"] == "pulse-program"
    assert manual.meta.acquisition["quadrature_selection"]["source"] == "manual"
    np.testing.assert_array_equal(auto.data, manual.data)
    auto_spectrum = scp.nmr.Experiment(auto).process()
    manual_spectrum = scp.nmr.Experiment(manual).process()
    np.testing.assert_array_equal(auto_spectrum.data, manual_spectrum.data)
    assert auto_spectrum.meta.tppi_alternation_state == "applied"
    np.testing.assert_allclose(auto.y.data, [0, 0.002])
    pure = scp.nmr.read_agilent(path, quadrature="states")
    pure_spectrum = scp.nmr.Experiment(pure).process()
    np.testing.assert_array_equal(
        manual_spectrum.data, np.roll(pure_spectrum.data, 1, axis=1)
    )

    source.write_bytes(b"unverified same-name variant")
    unresolved = scp.nmr.read_agilent(path)
    assert unresolved.meta.acquisition["quadrature_selection"]["mode"] is None
    np.testing.assert_array_equal(unresolved.data, auto.data)
    with pytest.raises(NotImplementedError, match="UNVALIDATED"):
        unresolved.fft(dim="y")

    source.write_bytes(bytes(256 * 1024 + 1))
    unresolved = scp.nmr.read_agilent(path)
    assert unresolved.meta.quadrature_mode == "states-stored-2d"
    assert unresolved.meta.acquisition["quadrature_selection"]["mode"] is None
    core_fft = import_module("spectrochempy.processing.fft.fft")
    monkeypatch.setattr(core_fft, "FFT_PREVALIDATION_API", 0)
    with pytest.raises(SpectroChemPyError, match="FFT prevalidation"):
        scp.nmr.read_agilent(path, quadrature="states")


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


@pytest.fixture
def synthetic_multitrace_series(tmp_path):
    procpar = {
        key: varian.create_pdic_param(key, values)
        for key, values in {
            "np": ["8"],
            "nf": ["3"],
            "arraydim": ["2"],
            "sw": ["5000"],
            "sfrq": ["125"],
            "ns": ["3"],
            "ne": ["2"],
        }.items()
    }
    varian.write_procpar(tmp_path / "procpar", procpar)
    words = np.arange(48, dtype=">i4").reshape(2, 3, 8)
    words[..., 1::2] *= -3
    path = tmp_path / "fid"
    with path.open("wb") as stream:
        varian.put_fileheader(stream, (2, 3, 8, 4, 32, 124, 0, 21, 1))
        for index, block in enumerate(words, 1):
            varian.put_blockheader(stream, (0, 21, index, 0, 3, 0, 0, 0, 0))
            stream.write(block.tobytes())
    expected = words[..., ::2] + 1j * words[..., 1::2]
    return path, expected.reshape(6, 4) / 6


@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_fid_series"),
    reason="provider does not support flat FID series",
)
@pytest.mark.parametrize("prevalidation_api", [0, 1])
def test_synthetic_multitrace_series_public(
    synthetic_multitrace_series, monkeypatch, prevalidation_api
):
    fft_module = import_module("spectrochempy.processing.fft.fft")
    monkeypatch.setattr(
        fft_module, "FFT_PREVALIDATION_API", prevalidation_api, raising=False
    )
    path, expected = synthetic_multitrace_series
    dataset = _check_public_2d(path, monkeypatch, "series")
    np.testing.assert_array_equal(dataset.data, expected)
    assert dataset.meta.nblocks == 2
    assert dataset.meta.ntraces == 3
    assert dataset.meta.interop_domains == ("unknown", "time")
    if prevalidation_api:
        stored = interop.read_agilent_fid_storage(path.parent, series_only=True)
        canonical = adapter.from_interop(interop.normalize(stored))
        assert scp.nmr.Experiment(canonical).domains == ("unknown", "time")
        assert scp.nmr.Experiment(canonical.fft().T).domains == ("frequency", "unknown")


def test_synthetic_inconsistent_3d_uses_external_varian_fallback(
    synthetic_multitrace_series, monkeypatch
):
    path, expected = synthetic_multitrace_series
    procpar_path = path.with_name("procpar")
    procpar = varian.read_procpar(procpar_path)
    for key in ("ni", "ni2"):
        procpar[key] = varian.create_pdic_param(key, ["2"])
    varian.write_procpar(procpar_path, procpar, overwrite=True)

    reader = import_module("spectrochempy_nmr.readers.read_agilent")
    assert reader.read_varian is varian.read
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return varian.read(*args, **kwargs)

    monkeypatch.setattr(reader, "read_varian", spy)
    with pytest.warns(UserWarning, match="returning raw 2D data"):
        dataset = scp.read_agilent(path)
    assert calls == [{"legacy_layout": True}]
    np.testing.assert_array_equal(dataset.data, expected)
    np.testing.assert_array_equal(dataset.y.data, np.arange(6))
    np.testing.assert_array_equal(dataset.x.data, np.arange(4) * 200)
    assert dataset.meta.encoding == ["QF", "QSIM"]


@pytest.mark.skipif(
    not hasattr(interop, "normalize_agilent_fid_series"),
    reason="provider does not support flat FID series",
)
def test_real_fid_series_public(monkeypatch):
    directory = os.environ.get("NMR_AGILENT_SERIES_DATA")
    if not directory:
        pytest.skip("set NMR_AGILENT_SERIES_DATA to the OpenVnmrJ CRAFT array")
    path = Path(directory)
    files = (path / "fid", path / "procpar")
    checksums = (
        "a806ff0f6864f9c521f397510bbcec47d1a297f97b76beed31391d218ea9b760",
        "96b651b8faea84ce9466b82a5137f13da6e0978b2f0667b23e7d5b3cac13c12a",
    )
    assert (
        tuple(hashlib.sha256(file.read_bytes()).hexdigest() for file in files)
        == checksums
    )
    dataset = _check_public_2d(path, monkeypatch, "series")
    assert dataset.shape == (4, 16384)
    assert dataset.meta.nblocks == 4
    assert dataset.meta.ntraces == 1
    assert dataset.y.units is None
    assert (
        tuple(hashlib.sha256(file.read_bytes()).hexdigest() for file in files)
        == checksums
    )
