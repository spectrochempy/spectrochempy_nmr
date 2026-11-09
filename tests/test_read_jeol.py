# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for JEOL JDF NMR reader."""

import os
from importlib import import_module
from inspect import signature
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
JEOL_DIR = Path(
    os.environ.get("NMR_JEOL_DIR", EXTRA_DATADIR / "testdata" / "nmrdata" / "jeol")
)


def _has_jeol_data():
    return (JEOL_DIR / "1H.jdf").exists()


def _read_jeol_or_skip(*args, **kwargs):
    try:
        result = scp.nmr.read_jeol(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"JEOL test data incomplete: {exc}")
    if result is None:
        pytest.skip("JEOL test data could not be read in this environment")
    return result


# ---------------------------------------------------------------------------
# nmrglue-ng provider layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestJeolNMRGlue:
    """Tests for the nmrglue-ng JEOL reading functions."""

    def test_read_jeol_1d(self):
        from nmrglue.fileio.jeol import read as read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "1H.jdf"))
        assert data.ndim == 1
        assert data.shape == (32768,)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_read_jeol_2d(self):
        from nmrglue.fileio.jeol import read as read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "COSY.jdf"))
        assert data.ndim == 2
        assert data.shape[0] == 512
        assert data.shape[1] == 1280
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_guess_udic_1d(self):
        from nmrglue.fileio.jeol import guess_udic
        from nmrglue.fileio.jeol import read as read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "1H.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 1
        assert udic[0]["label"] == "1H"
        assert udic[0]["sw"] > 0
        assert udic[0]["obs"] > 0

    def test_guess_udic_2d(self):
        from nmrglue.fileio.jeol import guess_udic
        from nmrglue.fileio.jeol import read as read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "COSY.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 2
        assert udic[0]["label"] == "1H"
        assert udic[1]["label"] == "1H"

    def test_guess_udic_hsqc(self):
        from nmrglue.fileio.jeol import guess_udic
        from nmrglue.fileio.jeol import read as read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "HSQC.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 2
        assert udic[0]["label"] == "1H"
        assert udic[1]["label"] == "13C"

    def test_header_fields(self):
        from nmrglue.fileio.jeol import read as read_jeol

        dic, _ = read_jeol(str(JEOL_DIR / "1H.jdf"))
        assert "header" in dic
        assert "parameters" in dic
        assert dic["header"]["data_format"] == "one_d"
        assert dic["header"]["endian"] in ("big_endian", "little_endian")


# ---------------------------------------------------------------------------
# Public reader API
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestReadJeol:
    """Tests for the public read_jeol() function."""

    def test_read_1d(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.shape == (32768,)
        assert ds.origin == "jeol"
        assert ds.history_entries[-1]["operation"] is None
        assert ds.history_entries[-1]["message"] == "Imported JEOL JDF file 1H.jdf"

    def test_read_13c(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "13C.jdf"))
        assert ds.shape == (32768,)
        assert ds.origin == "jeol"

    def test_read_cosy(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.ndim == 2
        assert ds.origin == "jeol"

    def test_read_hsqc(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "HSQC.jdf"))
        assert ds.ndim == 2
        assert ds.origin == "jeol"

    def test_read_hmbc(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "HMBC.jdf"))
        assert ds.ndim == 2
        assert ds.origin == "jeol"

    def test_1d_metadata_sw(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        sw = ds.meta.sw
        assert sw[0] is not None
        assert abs(sw[0] - 10016.0) < 1.0  # approximately 10016 Hz

    def test_1d_metadata_sfrq(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        sfrq = ds.meta.sfrq
        assert sfrq[0] is not None
        assert 399 < sfrq[0] < 401  # 400 MHz spectrometer

    def test_1d_metadata_nucleus(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        nuc = ds.meta.nucleus
        assert nuc[0] == "1H"

    def test_1d_metadata_solvent(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.solvent == "DMSO-D6"

    def test_1d_metadata_ns(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.ns == 128

    def test_1d_complex_data(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert np.issubdtype(ds.data.dtype, np.complexfloating)

    def test_1d_encoding_is_direct_complex(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.encoding == ["QSIM"]

    def test_1d_has_coords(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.x is not None

    def test_2d_shape(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.ndim == 2
        assert ds.shape[0] == 512
        assert ds.shape[1] == 1280

    def test_2d_encoding(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        enc = ds.meta.encoding
        assert len(enc) == 2
        assert enc[0] is not None
        assert enc[1] is not None

    def test_2d_iscomplex(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        ic = ds.meta.iscomplex
        assert len(ic) == 2

    def test_2d_has_coords(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.x is not None
        assert ds.y is not None

    def test_2d_origin(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "COSY.jdf"))
        assert ds.origin == "jeol"

    def test_meta_readonly(self):
        ds = _read_jeol_or_skip(str(JEOL_DIR / "1H.jdf"))
        assert ds.meta.readonly is True

    def test_hsqc_different_nuclei(self):
        """HSQC has 1H indirect and 13C direct dimensions."""
        ds = _read_jeol_or_skip(str(JEOL_DIR / "HSQC.jdf"))
        nuc = ds.meta.nucleus
        assert "1H" in nuc
        assert "13C" in nuc

    def test_auto_detect_protocol(self):
        """Auto-detect JEOL from .jdf extension."""
        ds = scp.nmr.read(str(JEOL_DIR / "1H.jdf"))
        assert ds.shape == (32768,)
        assert ds.origin == "jeol"

    def test_auto_detect_generic_read(self):
        """Auto-detect JEOL from .jdf extension via generic read."""
        ds = scp.nmr.read(str(JEOL_DIR / "1H.jdf"))
        assert ds.origin == "jeol"


@pytest.mark.parametrize("filename", ["1H.jdf", "13C.jdf"])
def test_public_jeol_fid_interop_parity(filename, monkeypatch):
    path = JEOL_DIR / filename
    if not path.is_file():
        pytest.skip("JEOL qualification JDF is unavailable")
    _assert_public_jeol_fid_interop_parity(path, monkeypatch)


@pytest.mark.skipif(
    not os.environ.get("NMR_JEOL_REFERENCE"), reason="Set NMR_JEOL_REFERENCE"
)
def test_public_jeol_reference_interop_parity(monkeypatch):
    path = Path(os.environ["NMR_JEOL_REFERENCE"])
    for origin in (None, "custom"):
        with monkeypatch.context() as patch:
            _assert_public_jeol_fid_interop_parity(
                path, patch, origin=origin, require_provider=True
            )


@pytest.mark.skipif(
    not os.environ.get("NMR_JEOL_REFERENCE"), reason="Set NMR_JEOL_REFERENCE"
)
@pytest.mark.parametrize("mode", ["automatic", "explicit", "core"])
def test_public_jeol_reference_stack_retains_legacy_contract(mode, monkeypatch):
    from spectrochempy_nmr import Experiment

    def unexpected_normalization(*args):
        pytest.fail("stacked JDFs require the legacy presentation")

    path = Path(os.environ["NMR_JEOL_REFERENCE"])
    if mode == "core":
        assert scp.nmr.read_jeol(path) is not None
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    monkeypatch.setattr(reader, "_read_normalized_jeol", unexpected_normalization)
    if mode == "automatic":
        merged = scp.nmr.read_jeol(path, path)
    elif mode == "explicit":
        merged = scp.nmr.read_jeol([path, path], merge=True)
    else:
        merged = scp.read(path, path)
    assert merged.shape[0] == 2
    assert Experiment(merged).domain == "time"


@pytest.mark.skipif(
    not os.environ.get("NMR_JEOL_REFERENCE"), reason="Set NMR_JEOL_REFERENCE"
)
def test_public_jeol_reference_preserves_custom_dataset_type(monkeypatch):
    class CustomDataset(scp.NDDataset):
        pass

    def unexpected_normalization(*args):
        pytest.fail("a custom dataset type requires the historical importer")

    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    monkeypatch.setattr(reader, "_read_normalized_jeol", unexpected_normalization)
    path = Path(os.environ["NMR_JEOL_REFERENCE"])
    assert isinstance(scp.nmr.read_jeol(path, objtype=CustomDataset), CustomDataset)


def _assert_public_jeol_fid_interop_parity(
    path, monkeypatch, *, origin=None, require_provider=False
):
    from nmrglue import interop

    from spectrochempy_nmr import Experiment
    from spectrochempy_nmr import interop as bridge

    if "legacy_presentation" not in signature(interop.normalize_jeol_fid).parameters:
        if require_provider:
            pytest.fail("provider lacks the JEOL public presentation contract")
        pytest.skip("provider lacks the JEOL public presentation contract")

    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    with monkeypatch.context() as fallback:
        fallback.setattr(reader, "_read_normalized_jeol", lambda path: None)
        historical = scp.nmr.read_jeol(path, origin=origin)
    assert historical is not None

    converted = []
    from_interop = bridge.from_interop

    def convert(result):
        dataset = from_interop(result)
        converted.append(dataset)
        return dataset

    def unexpected_fallback(*args):
        pytest.fail("qualified JDF was sent to the embedded decoder")

    monkeypatch.setattr(bridge, "from_interop", convert)
    monkeypatch.setattr(reader, "_read_jeol_raw", unexpected_fallback)
    actual = scp.nmr.read_jeol(
        path, description="public JEOL qualification", origin=origin
    )
    assert len(converted) == 1
    assert actual is converted[0]
    assert actual.description == "public JEOL qualification"
    assert actual.acquisition_date == historical.acquisition_date
    assert actual.origin == historical.origin
    assert len(actual.meta.applied_operations) == 1
    assert actual.meta.applied_operations[0]["conjugation"] == "conjugate"
    for key in historical.meta:
        assert actual.meta[key] == historical.meta[key], key

    for migrated, legacy in (
        (actual, historical),
        (actual.fft(), historical.fft()),
        (Experiment(actual).process(), Experiment(historical).process()),
    ):
        np.testing.assert_array_equal(migrated.data, legacy.data)
        np.testing.assert_array_equal(migrated.x.data, legacy.x.data)
        assert migrated.x.units == legacy.x.units
        assert migrated.x.title == legacy.x.title
        assert migrated.units == legacy.units


@pytest.mark.skipif(not os.environ.get("NMR_JEOL_COSY"), reason="Set NMR_JEOL_COSY")
def test_public_jeol_cosy_preserves_f2_fft_and_calibrates_axes(monkeypatch):
    from spectrochempy_nmr import Experiment
    from spectrochempy_nmr import interop as bridge

    path = Path(os.environ["NMR_JEOL_COSY"])
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    with monkeypatch.context() as fallback:
        fallback.setattr(reader, "_read_normalized_jeol", lambda path: None)
        historical = scp.nmr.read_jeol(path)

    from_interop = bridge.from_interop
    converted = []

    def convert(result):
        dataset = from_interop(result)
        converted.append(dataset)
        return dataset

    def unexpected_fallback(*args):
        pytest.fail("qualified COSY was sent to the embedded decoder")

    monkeypatch.setattr(bridge, "from_interop", convert)
    monkeypatch.setattr(reader, "_read_jeol_raw", unexpected_fallback)
    actual = scp.nmr.read_jeol(path)
    assert len(converted) == 1
    assert actual is converted[0]
    assert actual.meta.axis_roles == ("F1", "F2")
    assert actual.meta.applied_operations[0]["conjugation"] == "conjugate"
    assert len(actual.meta.applied_operations) == 1
    assert actual.meta.experiment == historical.meta.experiment
    assert actual.meta.ns == historical.meta.ns
    assert actual.meta.sw == historical.meta.sw[::-1]
    assert Experiment(actual).domain == "time"
    np.testing.assert_array_equal(actual.data, historical.data)
    for dim, width in zip(actual.dims, actual.meta.sw, strict=True):
        coord = actual.coordset[dim]
        np.testing.assert_allclose(coord.to("s").data, np.arange(coord.size) / width)

    direct = actual.fft(dim="x")
    np.testing.assert_array_equal(direct.data, historical.fft(dim="x").data)
    np.testing.assert_array_equal(direct.y.data, actual.y.data)
    width = actual.meta.sw[-1]
    expected_hz = np.fft.fftshift(np.fft.fftfreq(actual.x.size, d=1 / width))
    np.testing.assert_allclose(direct.x.data, expected_hz, atol=1.5, rtol=0)
    assert str(direct.x.units) == "Hz"
    decimated = actual[:, ::2]
    decimated_direct = decimated.fft(dim="x")
    expected_decimated_hz = np.fft.fftshift(
        np.fft.fftfreq(decimated.x.size, d=2 / width)
    )
    np.testing.assert_allclose(
        decimated_direct.x.data, expected_decimated_hz, atol=1.5, rtol=0
    )
    assert Experiment(direct).domain == "mixed"
    for dataset in (actual, direct):
        with pytest.raises(NotImplementedError, match="UNVALIDATED:F1"):
            dataset.fft(dim="y")


@pytest.mark.parametrize(
    "experiment, nominal, acquired, initial_indices, dwell",
    [
        ("HSQC", 128, 32, [0, 2, 4, 7], 58.48e-6),
        ("HMBC", 256, 64, [0, 1, 2, 4], 39.76e-6),
    ],
)
def test_public_jeol_listed_preserves_trace_pairs(
    experiment, nominal, acquired, initial_indices, dwell, monkeypatch
):
    from dataclasses import replace

    from spectrochempy_nmr import interop as bridge

    source = os.environ.get(f"NMR_JEOL_{experiment}")
    if not source:
        pytest.skip(f"Set NMR_JEOL_{experiment}")
    path = Path(source)
    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    with monkeypatch.context() as fallback:
        fallback.setattr(reader, "_read_normalized_jeol", lambda path: None)
        historical = scp.nmr.read_jeol(path)

    from_interop = bridge.from_interop
    converted = []

    def convert(result):
        dataset = from_interop(result)
        converted.append((dataset, result))
        return dataset

    def unexpected_fallback(*args):
        pytest.fail("qualified listed JDF was sent to the historical reader")

    monkeypatch.setattr(bridge, "from_interop", convert)
    monkeypatch.setattr(reader, "_read_jeol_raw", unexpected_fallback)
    actual = scp.nmr.read_jeol(path)
    assert len(converted) == 1
    converted_dataset, result = converted[0]
    assert actual is converted_dataset
    assert actual.shape == historical.shape
    assert actual.shape[0] == 2 * acquired
    assert actual.meta.acquisition["nominal_shape"] == (nominal, actual.shape[1])
    assert actual.meta.acquisition["trace_branch_order"] == (
        "section-0/1",
        "section-2/3",
    )
    np.testing.assert_array_equal(actual.data, historical.data)
    times = actual.y.to("s").data
    np.testing.assert_array_equal(times[::2], times[1::2])
    np.testing.assert_allclose(times[::2][:4], np.array(initial_indices) * dwell)
    assert np.all(np.diff(times[::2]) > 0)
    direct = actual.fft(dim="x")
    np.testing.assert_array_equal(direct.data, historical.fft(dim="x").data)
    np.testing.assert_array_equal(direct.y.data, actual.y.data)
    width = actual.meta.sw[-1]
    expected_hz = np.fft.fftshift(np.fft.fftfreq(actual.x.size, d=1 / width))
    assert direct.x.units == scp.ur.Hz
    np.testing.assert_allclose(direct.x.data, expected_hz, atol=width * 4e-4, rtol=0)
    for dataset in (actual, direct):
        with pytest.raises(NotImplementedError, match="UNVALIDATED:F1"):
            dataset.fft(dim="y")

    axis = result.axes[0]
    negative_first = axis.values.copy()
    negative_first[:2] = -5e-8 * dwell
    invalid = replace(
        result, axes=(replace(axis, values=negative_first), result.axes[1])
    )
    with pytest.raises(ValueError, match="nominal sampling grid"):
        from_interop(invalid)
    contradictory = replace(
        result, axes=(replace(axis, direction="decreasing"), result.axes[1])
    )
    with pytest.raises(ValueError, match="time coordinate for each trace pair"):
        from_interop(contradictory)
