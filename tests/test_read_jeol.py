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
# nmrglue vendored layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_jeol_data(), reason="JEOL test data not available")
class TestJeolNMRGlue:
    """Tests for the vendored nmrglue JEOL reading functions."""

    def test_read_jeol_1d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "1H.jdf"))
        assert data.ndim == 1
        assert data.shape == (32768,)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_read_jeol_2d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "COSY.jdf"))
        assert data.ndim == 2
        assert data.shape[0] == 512
        assert data.shape[1] == 1280
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_guess_udic_1d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import guess_udic
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "1H.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 1
        assert udic[0]["label"] == "1H"
        assert udic[0]["sw"] > 0
        assert udic[0]["obs"] > 0

    def test_guess_udic_2d(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import guess_udic
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "COSY.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 2
        assert udic[0]["label"] == "1H"
        assert udic[1]["label"] == "1H"

    def test_guess_udic_hsqc(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import guess_udic
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

        dic, data = read_jeol(str(JEOL_DIR / "HSQC.jdf"))
        udic = guess_udic(dic, data)
        assert udic["ndim"] == 2
        assert udic[0]["label"] == "1H"
        assert udic[1]["label"] == "13C"

    def test_header_fields(self):
        from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol

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
@pytest.mark.parametrize("mode", ["automatic", "explicit"])
def test_public_jeol_reference_stack_retains_legacy_contract(mode, monkeypatch):
    from spectrochempy_nmr import Experiment

    def unexpected_normalization(*args):
        pytest.fail("stacked JDFs require the legacy presentation")

    reader = import_module("spectrochempy_nmr.readers.read_jeol")
    monkeypatch.setattr(reader, "_read_normalized_jeol_fid", unexpected_normalization)
    path = Path(os.environ["NMR_JEOL_REFERENCE"])
    merged = (
        scp.nmr.read_jeol(path, path)
        if mode == "automatic"
        else scp.nmr.read_jeol([path, path], merge=True)
    )
    assert merged.shape[0] == 2
    assert Experiment(merged).domain == "time"


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
        fallback.setattr(reader, "_read_normalized_jeol_fid", lambda path: None)
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
