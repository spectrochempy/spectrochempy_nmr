# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for Agilent/Varian NMR reader."""

import numpy as np
import pytest
from spectrochempy_nmr.nmr_metadata import NMRMetadata
from spectrochempy_nmr.nmr_metadata import extract_agilent_metadata
from spectrochempy_nmr.readers.read_agilent import _reconstruct_agilent_states
from spectrochempy_nmr.readers.read_agilent import _detect_agilent_encoding
from spectrochempy_nmr.readers.read_agilent import _has_unvalidated_states_phase
from spectrochempy_nmr.readers.read_agilent import _validate_agilent_states_profile

import spectrochempy as scp

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
AGILENT_DIR = EXTRA_DATADIR / "testdata" / "nmrdata" / "agilent"


def _has_agilent_data():
    return (AGILENT_DIR / "agilent_1d" / "fid").exists()


def _read_agilent_or_skip(*args, **kwargs):
    try:
        result = scp.nmr.read_agilent(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"Agilent test data incomplete: {exc}")
    if result is None:
        pytest.skip("Agilent test data could not be read in this environment")
    return result


# ---------------------------------------------------------------------------
# nmrglue vendored layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestAgilentNMRGlue:
    """Tests for the vendored nmrglue Agilent reading functions."""

    def test_read_varian_procpar(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_1d" / "procpar"))
        assert isinstance(pdic, dict)
        assert "np" in pdic
        assert "sw" in pdic
        assert "sfrq" in pdic
        assert pdic["np"]["values"] == ["3000"]
        assert pdic["sw"]["values"] == ["50000"]

    def test_find_varian_shape_1d(self):
        from spectrochempy_nmr.extern.nmrglue import find_varian_shape
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_1d" / "procpar"))
        shape = find_varian_shape(pdic)
        assert shape == (1500,)

    def test_find_varian_shape_2d(self):
        from spectrochempy_nmr.extern.nmrglue import find_varian_shape
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_2d" / "procpar"))
        shape = find_varian_shape(pdic)
        assert shape[1] == 1500  # direct dim np/2
        assert shape[0] == 332  # ni * 2 phase values

    def test_read_varian_fid_1d(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian

        dic, data = read_varian(
            str(AGILENT_DIR / "agilent_1d"),
            fid_file="fid",
            procpar_file="procpar",
        )
        assert data.ndim == 1
        assert data.shape == (1500,)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_read_varian_fid_2d(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian

        dic, data = read_varian(
            str(AGILENT_DIR / "agilent_2d"),
            fid_file="fid",
            procpar_file="procpar",
        )
        assert data.ndim == 2
        assert data.shape == (332, 1500)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_find_varian_torder(self):
        from spectrochempy_nmr.extern.nmrglue import find_varian_shape
        from spectrochempy_nmr.extern.nmrglue import find_varian_torder
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_2d" / "procpar"))
        shape = find_varian_shape(pdic)
        torder = find_varian_torder(pdic, shape)
        assert torder in ("f", "r", "o")

    def test_procpar_key_count(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_1d" / "procpar"))
        assert len(pdic) > 100  # typical procpar has 100+ parameters


# ---------------------------------------------------------------------------
# Public reader API
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestReadAgilent:
    """Tests for the public read_agilent() function."""

    def test_read_1d_fid(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.shape == (1500,)
        assert ds.origin == "agilent"
        assert np.issubdtype(ds.data.dtype, np.complexfloating)
        assert ds.history_entries[-1]["operation"] is None
        assert (
            ds.history_entries[-1]["message"]
            == "Imported Agilent/Varian dataset agilent_1d (FID)"
        )

    def test_read_1d_directory(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d"))
        assert ds.shape == (1500,)
        assert ds.origin == "agilent"

    def test_read_2d_fid(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        assert ds.ndim == 3
        assert ds.origin == "agilent"
        assert ds.data.shape == (2, 166, 1500)
        assert ds.meta.td == [2, 166, 1500]
        assert ds.x.size == 1500
        assert ds.y.size == 166

    def test_read_2d_tppi(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d_tppi" / "fid"))
        assert ds.ndim == 2
        assert ds.origin == "agilent"
        enc = ds.meta.encoding
        assert "UNVALIDATED:AGILENT-TPPI" in enc

    def test_1d_metadata_sw(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        sw = ds.meta.sw
        assert sw[0] == 50000.0

    def test_1d_metadata_sfrq(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        sfrq = ds.meta.sfrq
        assert sfrq[0] is not None
        assert 100 < sfrq[0] < 200  # C13 frequency around 125 MHz

    def test_1d_metadata_nucleus(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        nuc = ds.meta.nucleus
        assert nuc[0] is not None

    def test_1d_complex_data(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert np.issubdtype(ds.data.dtype, np.complexfloating)

    def test_1d_normalised(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.meta.ns_norm == 1.0

    def test_1d_has_coords(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.x is not None

    def test_2d_encoding(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        enc = ds.meta.encoding
        assert enc == ["CHANNELS", "UNVALIDATED:AGILENT-STATES", "DIRECT:QSIM"]

    def test_2d_iscomplex(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        ic = ds.meta.iscomplex
        assert len(ic) == 3
        assert ic[-1] is True  # direct dim is complex

    def test_2d_has_coords(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        assert ds.x is not None
        assert ds.y is not None

    def test_2d_origin(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        assert ds.origin == "agilent"

    def test_meta_readonly(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.meta.readonly is True


class TestAgilentStatesReconstruction:
    """Synthetic tests for the documented Agilent States trace layout."""

    def test_pairs_traces_and_preserves_direct_complex_axis(self):
        data = np.empty((4, 3), dtype=np.complex128)
        data[0] = [1 + 2j, 3 + 4j, 5 + 6j]
        data[1] = [10 + 20j, 30 + 40j, 50 + 60j]
        data[2] = [100 + 200j, 300 + 400j, 500 + 600j]
        data[3] = [1000 + 2000j, 3000 + 4000j, 5000 + 6000j]

        result = _reconstruct_agilent_states(data)

        assert result.shape == (2, 2, 3)
        np.testing.assert_array_equal(result[0], data[0::2])
        np.testing.assert_array_equal(result[1], data[1::2])

    def test_odd_trace_count_is_rejected_without_pairing(self):
        with pytest.raises(ValueError, match="even row count"):
            _reconstruct_agilent_states(np.zeros((3, 4), dtype=np.complex128))

    @staticmethod
    def _profile(*, phase=("1", "2"), ni="2", arraydim="4", np_raw="6", ntraces=1):
        phase_values = list(phase) if isinstance(phase, (list, tuple)) else phase
        pdic = {
            "phase": {"values": phase_values},
            "ni": {"values": [ni]},
            "arraydim": {"values": [arraydim]},
            "np": {"values": [np_raw]},
        }
        dic = {} if ntraces is None else {"ntraces": ntraces}
        return pdic, dic

    def test_validates_only_the_established_two_trace_profile(self):
        pdic, dic = self._profile()
        supported, reason = _validate_agilent_states_profile(
            pdic, dic, np.zeros((4, 3), dtype=np.complex128)
        )
        assert supported is True
        assert reason is None

    @pytest.mark.parametrize("np_value", ["inf", "2.5", "not-a-number"])
    def test_malformed_direct_np_does_not_crash_encoding_detection(self, np_value):
        pdic = {"np": {"values": [np_value]}, "phase": {"values": ["1", "2"]}}
        assert _detect_agilent_encoding(pdic, 2) == ("QF", "STATES")

    @pytest.mark.parametrize("phase_values", [None, "1,2"])
    def test_malformed_phase_values_do_not_crash_encoding_detection(self, phase_values):
        pdic = {"np": {"values": ["3000"]}, "phase": {"values": phase_values}}
        assert _detect_agilent_encoding(pdic, 2) == ("QSIM", "QF")

    @pytest.mark.parametrize(
        "phase_entry",
        [
            {"values": None},
            {"values": "1,2"},
            {"values": []},
            {"values": ["1.5"]},
            {"values": ["inf"]},
            {"values": ["1", "2", "3"]},
        ],
    )
    def test_marks_unvalidated_phase_metadata_for_reader_diagnostic(self, phase_entry):
        assert _has_unvalidated_states_phase({"phase": phase_entry}) is True

    def test_does_not_mark_missing_or_single_phase_as_states_candidate(self):
        assert _has_unvalidated_states_phase({}) is False
        assert (
            _has_unvalidated_states_phase(
                {"ni": {"values": ["2"]}, "arraydim": {"values": ["4"]}}
            )
            is True
        )
        assert _has_unvalidated_states_phase({"phase": {"values": ["3"]}}) is False

    def test_rejects_non_scalar_numeric_procpar_values(self):
        pdic, dic = self._profile()
        pdic["ni"] = {"values": np.array([2, 3])}
        supported, reason = _validate_agilent_states_profile(
            pdic, dic, np.zeros((4, 3), dtype=np.complex128)
        )
        assert supported is False
        assert "finite integers" in reason

    @pytest.mark.parametrize(
        "phase_values",
        [None, "1,2", [], ["1.5"], ["inf"], ["1", "2", "3"]],
    )
    def test_public_reader_warns_and_preserves_invalid_states(
        self, tmp_path, monkeypatch, phase_values
    ):
        from importlib import import_module

        reader = import_module("spectrochempy_nmr.readers.read_agilent")
        data_dir = tmp_path / "invalid_states"
        data_dir.mkdir()
        fid_path = data_dir / "fid"
        (data_dir / "procpar").touch()
        fid_path.touch()

        pdic, _ = self._profile(phase=phase_values, ntraces=1)
        fake_dic = {"procpar": pdic, "ntraces": 1, "np": 6}
        raw = np.arange(12, dtype=np.float64).reshape(4, 3).astype(np.complex128)
        monkeypatch.setattr(
            reader, "read_varian", lambda *args, **kwargs: (fake_dic, raw)
        )

        with pytest.raises(NotImplementedError, match="Unvalidated Agilent States"):
            dataset = reader._read_agilent(scp.NDDataset(), fid_path)

    def test_public_reader_reconstructs_only_valid_states_profile(
        self, tmp_path, monkeypatch
    ):
        from importlib import import_module

        reader = import_module("spectrochempy_nmr.readers.read_agilent")
        data_dir = tmp_path / "valid_states"
        data_dir.mkdir()
        fid_path = data_dir / "fid"
        (data_dir / "procpar").touch()
        fid_path.touch()

        pdic, _ = self._profile(ntraces=1)
        fake_dic = {"procpar": pdic, "ntraces": 1, "np": 6}
        raw = np.arange(12, dtype=np.float64).reshape(4, 3).astype(np.complex128)
        monkeypatch.setattr(
            reader, "read_varian", lambda *args, **kwargs: (fake_dic, raw)
        )

        dataset = reader._read_agilent(scp.NDDataset(), fid_path)

        assert dataset.data.shape == (2, 2, 3)

    @pytest.mark.parametrize(
        ("profile", "shape", "message"),
        [
            ({"phase": ("1", "2", "3")}, (4, 3), "phase"),
            ({"arraydim": "6"}, (4, 3), "arraydim"),
            ({}, (3, 3), "traces"),
            ({"ntraces": 2}, (4, 3), "ntraces"),
            ({"ni": "2.5"}, (4, 3), "finite integers"),
            ({"np_raw": "inf"}, (4, 3), "finite integers"),
            ({"phase": ("1.5", "2")}, (4, 3), "phase"),
            ({"phase": ("inf", "2")}, (4, 3), "phase"),
            ({"ntraces": None}, (4, 3), "missing ntraces"),
            ({"ntraces": np.array([1, 2])}, (4, 3), "ntraces"),
            ({"phase": (np.array(1), np.array(2))}, (4, 3), "phase"),
        ],
    )
    def test_rejects_unvalidated_states_profiles(self, profile, shape, message):
        pdic, dic = self._profile(**profile)
        supported, reason = _validate_agilent_states_profile(
            pdic, dic, np.zeros(shape, dtype=np.complex128)
        )
        assert supported is False
        assert message in reason


# ---------------------------------------------------------------------------
# extract_agilent_metadata
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestExtractAgilentMetadata:
    """Tests for extract_agilent_metadata()."""

    def test_1d_metadata(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        nmr_meta = extract_agilent_metadata(ds.meta)
        assert isinstance(nmr_meta, NMRMetadata)
        assert nmr_meta.ndim == 1
        assert nmr_meta.domains == ("time",)
        assert nmr_meta.source_kind == "fid"
        assert nmr_meta.iscomplex == (True,)
        assert nmr_meta.spectral_width_hz is not None
        assert nmr_meta.spectral_width_hz[0] == 50000.0
        assert nmr_meta.spectrometer_freq_mhz is not None
        assert 100 < nmr_meta.spectrometer_freq_mhz[0] < 200

    def test_2d_metadata(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        nmr_meta = extract_agilent_metadata(ds.meta)
        assert nmr_meta.ndim == 2
        assert nmr_meta.domains == ("time", "time")
        assert nmr_meta.source_kind == "ser"
        assert nmr_meta.nuclei is not None
        assert len(nmr_meta.nuclei) == 2

    def test_none_input(self):
        result = extract_agilent_metadata(None)
        assert result.ndim == 0
        assert result.domains == ()

    def test_encoding_strings(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d_tppi" / "fid"))
        nmr_meta = extract_agilent_metadata(ds.meta)
        assert nmr_meta.encoding is not None
        assert "UNVALIDATED:AGILENT-TPPI" in nmr_meta.encoding
