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
