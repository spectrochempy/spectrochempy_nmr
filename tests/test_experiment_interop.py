"""Normalized Experiment processing is independent of source identity."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop, reference_interop_1d
from spectrochempy_nmr.experiment import Experiment
from test_interop_conversion import _normalized


def _fid(sign=-1):
    size = 32
    source = _normalized(
        np.exp(2j * np.pi * np.arange(size) / 8), axis_values=np.arange(size) / 2000
    )
    source.normalization_profile = "arbitrary-producer-profile"
    source.acquisition.update(
        {
            "observation_frequency_mhz": 125.0,
            "nucleus": "13C",
            "frequency_axis": {
                "schema": "normalized-frequency-axis-v1",
                "bin_sign": sign,
                "reference_frequency_mhz": 125.0,
                "reference_enabled": None,
                "offsets_hz": {"primary": 50.0, "secondary": 250.0},
            },
        }
    )
    return from_interop(source)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("origin", [None, "agilent", "topspin", "other"])
def test_origin_profile_and_raw_fields_do_not_select_processing(sign, origin):
    source = _fid(sign)
    if origin is not None:
        source.origin = origin
    source.meta.origin = origin
    source.meta.normalization_profile = origin
    source.meta.source_metadata = {
        "procpar": {"refpos": {"values": ["n"]}},
        "sw_h": -99,
    }
    experiment = Experiment(source)
    assert experiment.source_kind == "fid"
    assert experiment.domains == ("time",)
    result = experiment.process(reference="primary")
    expected = (50 + sign * (np.arange(32) - 16) * 2000 / 32) / 125
    np.testing.assert_allclose(
        result.data, np.fft.fftshift(np.fft.fft(source.data)), rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(result.x.data, expected, atol=1e-12)
    assert Experiment(result).source_kind == "processed_1d"
    assert result.meta.nmr_processing["scp_processing"]["requested"] == {
        "reference": "primary"
    }
    assert source.meta.data_domain == "time"
    assert source.meta.get("frequency_reference") is None


def test_full_processing_matches_explicit_core_chain_and_persists(tmp_path):
    source = _fid()
    result = Experiment(source).process(
        apodization="em",
        lb=2,
        size=64,
        phase="manual",
        phc0=30,
        phc1=0,
        reference="secondary",
    )
    manual = source.em(lb=2 * scp.ur.Hz).zf_size(size=64).fft()
    manual = Experiment(manual).process(phase="manual", phc0=30, phc1=0)
    manual = reference_interop_1d(manual, reference="secondary")
    np.testing.assert_allclose(result.data, manual.data)
    np.testing.assert_allclose(result.x.data, manual.x.data)
    path = tmp_path / "experiment.scp"
    result.write(path, confirm=False)
    restored = scp.read(path)
    np.testing.assert_array_equal(restored.data, result.data)
    assert Experiment(restored).is_frequency_domain
    np.testing.assert_array_equal(Experiment(restored).process().data, result.data)
    assert restored.meta.frequency_reference == result.meta.frequency_reference


def test_default_stays_in_hz_without_reference_descriptor():
    source = _fid()
    source.meta.acquisition.pop("frequency_axis")
    result = Experiment(source).process()
    assert result.x.units == scp.ur.Hz
    assert result.meta.get("frequency_reference") is None
    assert result.meta.nmr_processing["scp_processing"]["applied"] == {"fft": True}


@pytest.mark.parametrize("invalid", ["index", "irregular", "mask", "width"])
def test_invalid_normalized_time_grid_refuses_without_source_mutation(invalid):
    source = _fid()
    if invalid == "index":
        source.set_coordset(scp.Coord(np.arange(32)))
    elif invalid == "irregular":
        values = source.x.data.copy()
        values[5] += 0.0001
        source.set_coordset(scp.Coord(values, units="s", sigdigits=15))
    elif invalid == "mask":
        source.mask = np.arange(32) == 5
    else:
        source.meta.acquisition["spectral_width_hz"] = 1000
    before = source.data.copy()
    with pytest.raises(ValueError):
        Experiment(source).process()
    np.testing.assert_array_equal(source.data, before)
    assert source.meta.data_domain == "time"


def test_unsupported_interop_structure_never_falls_back_to_vendor():
    source = _fid()
    source.meta.origin = "topspin"
    source.meta.interop_physical_dimensions = ("F1", "F2")
    with pytest.raises(ValueError, match="structure"):
        Experiment(source)


@pytest.mark.parametrize("width", [np.int64(2000), np.float32(2000), np.float64(2000)])
def test_normalized_numpy_width_scalars_remain_usable(width):
    source = _fid()
    source.meta.acquisition["spectral_width_hz"] = width
    result = Experiment(source).process(reference="primary")
    assert result.x.units == scp.ur.ppm


@pytest.mark.parametrize("display", [False, True])
def test_direct_fft_uses_sample_spacing_after_decimation(display):
    source = _fid()
    if display:
        source.meta.acquisition["fft_display_axis"] = {
            "grid": "inclusive-endpoints",
            "direction": "decreasing",
            "center_hz": 0.0,
            "spectral_width_hz": 2100.0,
            "reference_frequency_mhz": 125.0,
        }
        np.testing.assert_allclose(source.fft(ppm=False).x.data[[0, -1]], [1050, -1050])
    source = source[::2]
    spectrum = source.fft()

    expected = np.fft.fftshift(np.fft.fftfreq(source.size, d=0.001))
    assert spectrum.x.units == scp.ur.Hz
    np.testing.assert_allclose(spectrum.x.data, expected, atol=1e-9)
    with pytest.raises(ValueError, match="complete ascending centered FFT grid"):
        reference_interop_1d(spectrum, reference="primary")


def test_fid_preparation_rejects_stale_grid_and_wrong_operation_order():
    source = _fid()
    source.meta.acquisition["fid_preparation"] = {
        "schema": "fid-preparation-v1",
        "complex_convention": "centered-negative-exponent-dft",
        "sample_count": source.size,
        "digital_filter": {"state": "not-required"},
        "scan_gain": {"state": "available", "divisor": 2.0},
    }
    with pytest.raises(ValueError, match="original calibrated time grid"):
        Experiment(source[::2]).prepare_fid(normalize_scan_gain=True)
    prepared = Experiment(source).prepare_fid(
        digital_filter="time", normalize_scan_gain=True
    )
    np.testing.assert_array_equal(prepared.data, source.data / 2)
    assert not prepared.meta.acquisition["digital_filter"]["applied"]
    again = Experiment(prepared).prepare_fid(
        digital_filter="time", normalize_scan_gain=True
    )
    np.testing.assert_array_equal(again.data, prepared.data)
    with pytest.raises(ValueError, match="preparation order"):
        Experiment(prepared).prepare_fid(remove_dc_offset=True)
    assert "scan_gain" not in source.meta.acquisition


def test_qf_display_rejects_repeated_coordinates_after_fft_resize():
    from spectrochempy_nmr.processing.fft_postprocess import (
        _apply_interop_fft_presentation,
    )

    bins = np.arange(4, dtype=float) - 2
    dataset = scp.NDDataset(np.ones((2, 4)), dims=["y", "x"])
    dataset.set_coordset(x=scp.Coord(bins, units="Hz"))
    descriptor = {
        "physical_axis": "F2",
        "spectral_width_hz": 4.0,
        "fft_display_axis": {
            "grid": "inclusive-endpoints",
            "direction": "decreasing",
            "center_hz": -9.99999999999e17,
            "spectral_width_hz": 256.0,
            "reference_frequency_mhz": 1.0,
        },
    }
    with pytest.raises(ValueError, match="unrepresentable interop FFT display grid"):
        _apply_interop_fft_presentation(
            dataset, "x", ppm=False, physical=False, acquisition=descriptor
        )


@pytest.mark.parametrize("delay", [2.25, 3.625])
@pytest.mark.parametrize("size", [32, 64])
def test_qsim_pknl_uses_the_fractional_delay_on_the_final_fft_grid(delay, size):
    source = _fid()
    source.meta.normalization_profile = "bruker-fid-1d-complex-v1"
    source.meta.aq_mod = [1]
    source.meta.nmr_processing = {
        "vendor_profile": {
            "parameters": {"SI": 64, "SW_p": 2000, "SF": 125, "OFFSET": 10}
        }
    }
    acquisition = source.meta.acquisition
    acquisition["storage_points"] = source.size
    acquisition["acquired_points"] = 30
    acquisition["digital_filter"] = {"applied": False}
    acquisition["fid_preparation"] = {
        "schema": "fid-preparation-v1",
        "complex_convention": "centered-negative-exponent-dft",
        "sample_count": source.size,
        "digital_filter": {"state": "available", "group_delay_points": delay},
    }
    original = source.data.copy()
    spectrum = Experiment(source).process(
        digital_filter="pknl", size=size if size == 32 else None
    )

    bins = np.arange(size) / size
    expected = np.fft.fftshift(np.fft.fft(original[:30], n=size))
    expected *= (-1) ** (1 - round(delay)) * np.exp(2j * np.pi * delay * bins)
    np.testing.assert_allclose(spectrum.data, expected, atol=1e-12)
    np.testing.assert_allclose(spectrum.x.data, 10 - 16 * bins, atol=1e-12)
    assert spectrum.x.units == scp.ur.ppm
    trace = spectrum.meta.nmr_processing["scp_processing"]
    assert trace["requested"]["digital_filter"] == "pknl"
    assert trace["applied"]["digital_filter"] == {
        "method": "pknl",
        "requested": True,
        "applied": True,
        "group_delay_points": delay,
        "fft_size": size,
    }
    np.testing.assert_array_equal(source.data, original)
    assert not source.meta.acquisition["digital_filter"]["applied"]
    with pytest.raises(ValueError, match="raw Bruker QSIM"):
        Experiment(spectrum).process(digital_filter="pknl")

    source.meta.acquisition["digital_filter"] = {"applied": True}
    with pytest.raises(ValueError, match="already applied"):
        Experiment(source).process(digital_filter="pknl")
