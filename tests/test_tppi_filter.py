"""TPPI filter compensation on the F2 frequency grid, independent matrix oracle."""

import sys

import numpy as np
import pytest

from spectrochempy_nmr import Experiment, read
from test_experiment_interop import _tppi_fid


@pytest.mark.parametrize("delay", [0, 1.5, 2, 5.75])
def test_tppi_phase_filter_matches_matrix_dft_and_persists(
    tmp_path, monkeypatch, delay
):
    source = _tppi_fid()
    source.meta.acquisition["digital_filter"] = {
        "applied": False,
        "group_delay_points": delay,
        "source": "declared acquisition",
    }
    before = source.data.copy()
    ny, nx = before.shape
    direct = np.exp(2j * np.pi * np.outer(np.arange(nx) - nx // 2, np.arange(nx)) / nx)
    f2 = before.conj() @ direct.T
    f2 *= np.exp(-2j * np.pi * delay * np.arange(nx) / nx)
    indirect = np.exp(-2j * np.pi * np.outer(np.arange(ny), np.arange(ny)) / (2 * ny))
    a, b = indirect @ f2.real, indirect @ f2.imag
    expected = np.stack((a.real, b.real, -a.imag, -b.imag))
    result = Experiment(source).process(digital_filter="phase")
    np.testing.assert_allclose(result.data, expected, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(source.data, before)
    uncorrected = Experiment(source).process(digital_filter="none")
    np.testing.assert_array_equal(result.x.data, uncorrected.x.data)
    np.testing.assert_array_equal(result.y.data, uncorrected.y.data)
    assert result.shape == uncorrected.shape
    trace = result.meta.nmr_processing["scp_processing"]
    assert trace["requested"]["digital_filter"] == "phase"
    assert trace["applied"]["digital_filter"]["applied"] is True
    assert trace["applied"]["digital_filter"]["removed_points"] == 0
    assert trace["applied"]["digital_filter"]["group_delay_points"] == delay
    assert result.meta.acquisition["digital_filter"]["applied"] is True
    assert source.meta.acquisition["digital_filter"]["applied"] is False
    spectrum_path = tmp_path / "corrected.scp"
    result.write(spectrum_path, confirm=False)
    spectrum = read(spectrum_path)
    np.testing.assert_array_equal(spectrum.data, result.data)
    assert spectrum.meta.acquisition["digital_filter"]["applied"] is True
    saved = tmp_path / "raw.scp"
    source.write(saved, confirm=False)
    restored = read(saved)
    monkeypatch.setitem(sys.modules, "nmrglue", None)
    restored.origin = "unknown"
    restored.meta.normalization_profile = "unknown"
    again = Experiment(restored).process(digital_filter="phase")
    np.testing.assert_array_equal(again.data, result.data)


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        "unknown",
        {"applied": False},
        {"applied": False, "group_delay_points": True},
        {"applied": False, "group_delay_points": -1},
        {"applied": False, "group_delay_points": float("nan")},
        {"applied": False, "group_delay_points": float("inf")},
        {"applied": False, "group_delay_points": 1000},
        {"applied": True, "group_delay_points": 2},
    ],
)
def test_tppi_phase_filter_requires_known_unapplied_delay(descriptor):
    source = _tppi_fid()
    source.meta.acquisition["digital_filter"] = descriptor
    before = source.data.copy()
    with pytest.raises(ValueError, match="digital-filter delay"):
        Experiment(source).process(digital_filter="phase")
    np.testing.assert_array_equal(source.data, before)


def test_tppi_phase_filter_refuses_prior_correction_in_operation_history():
    source = _tppi_fid()
    source.meta.acquisition["digital_filter"] = {
        "applied": False,
        "group_delay_points": 2,
    }
    source.meta.applied_operations = [{"digital_filter": {"applied": True}}]
    with pytest.raises(ValueError, match="already been corrected"):
        Experiment(source).process(digital_filter="phase")


@pytest.mark.parametrize("bad_option", ["none", "phase"])
def test_tppi_phase_filter_does_not_enable_delay_override(bad_option):
    source = _tppi_fid()
    source.meta.acquisition["digital_filter"] = {
        "applied": False,
        "group_delay_points": 2,
    }
    with pytest.raises(ValueError, match="bruker_group_delay"):
        Experiment(source).process(digital_filter=bad_option, bruker_group_delay=3)
