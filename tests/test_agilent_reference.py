"""Explicit referencing without optional-provider imports or sample changes."""

import json

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import reference_agilent_1d


def _spectrum(size=8):
    signal = np.exp(2j * np.pi * np.arange(size) / size)
    return scp.NDDataset(
        np.fft.fftshift(np.fft.fft(signal)),
        coordset=[scp.Coord((np.arange(size) - size // 2) * 1000 / size, units="Hz")],
        meta={
            "normalization_profile": "agilent-fid-1d-complex-v1",
            "isfreq": [True],
            "td": [size],
            "data_domain": "time",
            "source_metadata": {"procpar": {"refpos": {"values": ["0"]}}},
            "acquisition": {
                "spectral_width_hz": 1000.0,
                "observation_frequency_mhz": 100.00012,
                "original_reference": {
                    "reffrq": 100.0,
                    "rfl": 400.0,
                    "rfp": 20.0,
                    "tof": 37.0,
                },
            },
        },
    )


@pytest.mark.parametrize("reference", ["rfl-rfp", "sfrq-reffrq"])
def test_reference_has_independent_expected_ppm_axis(reference):
    source = _spectrum()
    before = source.data.copy()
    source.mask = np.array([False, False, True, False, False, False, False, False])
    result = reference_agilent_1d(source, reference=reference)
    np.testing.assert_allclose(
        result.x.data, [6.2, 4.95, 3.7, 2.45, 1.2, -0.05, -1.3, -2.55], atol=1e-10
    )
    np.testing.assert_array_equal(result.data, before)
    assert not np.shares_memory(result.data, source.data)
    np.testing.assert_array_equal(result.mask, source.mask)
    assert result.x.units == scp.ur.ppm
    assert result.x.reversed
    assert result.meta.frequency_reference["fft_frequency_sign"] == -1
    assert result.meta.frequency_reference["consistent_within_0_1_hz"]
    assert result.meta.data_domain == "frequency"
    assert source.meta.data_domain == "time"
    assert source.meta.get("frequency_reference") is None
    assert source.x.units == scp.ur.Hz


@pytest.mark.parametrize(
    "reference, missing", [("rfl-rfp", "observation"), ("sfrq-reffrq", "graphical")]
)
def test_explicit_single_source_does_not_invent_other_evidence(reference, missing):
    source = _spectrum()
    if missing == "observation":
        source.meta.acquisition["observation_frequency_mhz"] = None
    else:
        source.meta.acquisition["original_reference"].pop("rfl")
        source.meta.acquisition["original_reference"].pop("rfp")
    result = reference_agilent_1d(source, reference=reference)
    assert result.meta.frequency_reference["disagreement_hz"] is None
    assert result.meta.frequency_reference["consistent_within_0_1_hz"] is None


@pytest.mark.parametrize(
    "kwargs", [{"reference": "auto"}, {"reference": "rfl-rfp", "units": "kHz"}]
)
def test_no_automatic_model_or_unqualified_output_units(kwargs):
    with pytest.raises(ValueError):
        reference_agilent_1d(_spectrum(), **kwargs)


@pytest.mark.parametrize("size", [8, 9, 32])
def test_dft_grid_spacing_and_zero_bin_not_endpoint_linspace(size):
    source = _spectrum(size)
    result = reference_agilent_1d(source, reference="rfl-rfp", units="Hz")
    assert result.x.data[size // 2] == pytest.approx(120)
    np.testing.assert_allclose(np.diff(result.x.data), -1000 / size, atol=1e-10)
    ppm = result.x.to("ppm")
    np.testing.assert_allclose(ppm.data, result.x.data / 100.0, atol=1e-10)
    np.testing.assert_allclose(ppm.to("Hz").data, result.x.data, atol=1e-8)


def test_conflicting_models_require_explicit_choice_and_retain_evidence():
    source = _spectrum()
    source.meta.acquisition["observation_frequency_mhz"] = 100.0005
    with pytest.raises(TypeError, match="reference"):
        reference_agilent_1d(source)
    graphical = reference_agilent_1d(source, reference="rfl-rfp")
    absolute = reference_agilent_1d(source, reference="sfrq-reffrq")
    assert graphical.meta.frequency_reference["center_hz"] == 120
    assert absolute.meta.frequency_reference["center_hz"] == pytest.approx(500)
    assert graphical.meta.frequency_reference["disagreement_hz"] == pytest.approx(380)
    assert not graphical.meta.frequency_reference["consistent_within_0_1_hz"]
    np.testing.assert_allclose(absolute.x.data - graphical.x.data, 3.8, atol=1e-10)
    np.testing.assert_array_equal(absolute.data, graphical.data)


@pytest.mark.parametrize("parameter", ["reffrq", "rfl", "rfp"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "bad", 10**400])
def test_bad_reference_refuses_without_mutating_input(parameter, value):
    source = _spectrum()
    source.meta.acquisition["original_reference"][parameter] = value
    before = source.data.copy()
    coordinates = source.x.data.copy()
    with pytest.raises(ValueError):
        reference_agilent_1d(source, reference="rfl-rfp")
    np.testing.assert_array_equal(source.data, before)
    np.testing.assert_array_equal(source.x.data, coordinates)
    assert source.meta.get("frequency_reference") is None


@pytest.mark.parametrize("value", [0, -1, None])
def test_missing_or_nonpositive_reference_frequency_refuses(value):
    source = _spectrum()
    source.meta.acquisition["original_reference"]["reffrq"] = value
    with pytest.raises(ValueError, match="reffrq"):
        reference_agilent_1d(source, reference="rfl-rfp")


def test_disabled_referencing_and_wrong_domain_refuse():
    source = _spectrum()
    source.meta.source_metadata["procpar"]["refpos"]["values"] = ["n"]
    with pytest.raises(ValueError, match="disabled"):
        reference_agilent_1d(source, reference="rfl-rfp")
    source.meta.isfreq = [False]
    with pytest.raises(ValueError, match="FFT first"):
        reference_agilent_1d(source, reference="rfl-rfp")


def test_cropped_reversed_and_already_referenced_axes_refuse():
    source = _spectrum()
    for invalid in (source[:4], source[::-1]):
        with pytest.raises(ValueError, match="complete ascending"):
            reference_agilent_1d(invalid, reference="rfl-rfp")
    result = reference_agilent_1d(source, reference="rfl-rfp", units="Hz")
    with pytest.raises(ValueError, match="already referenced"):
        reference_agilent_1d(result, reference="rfl-rfp")


def test_ppm_reference_and_context_survive_scp_persistence(tmp_path):
    result = reference_agilent_1d(_spectrum(), reference="rfl-rfp")
    target = tmp_path / "reference.scp"
    result.write(target, confirm=False)
    restored = scp.read(target)
    np.testing.assert_array_equal(restored.data, result.data)
    np.testing.assert_allclose(restored.x.data, result.x.data, atol=1e-12)
    np.testing.assert_allclose(restored.x.to("Hz").data, result.x.data * 100, atol=1e-8)
    assert json.loads(json.dumps(restored.meta.frequency_reference)) == json.loads(
        json.dumps(result.meta.frequency_reference)
    )


def test_inverse_restores_time_axis_without_automatic_rereferencing():
    source = _spectrum()
    referenced = reference_agilent_1d(source, reference="rfl-rfp")
    restored = referenced.ifft()
    np.testing.assert_allclose(restored.x.data, np.arange(8) / 1000, atol=1e-12)
    assert restored.x.units == scp.ur.s
    assert restored.meta.data_domain == "time"
    assert restored.meta.frequency_reference is None
    assert restored.meta.last_frequency_reference["reference"] == "rfl-rfp"
    spectrum = restored.fft()
    np.testing.assert_allclose(spectrum.data, source.data, atol=1e-10)
    assert spectrum.x.units == scp.ur.Hz
    assert spectrum.meta.data_domain == "frequency"
    assert spectrum.meta.frequency_reference is None
