"""Generic real/complex frequency-domain interop, independent of vendor identity."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment
from test_interop_conversion import _normalized


def _spectrum(*, complex_data=True, increasing=False, units="Hz"):
    values = np.arange(1, 9, dtype=float)
    if complex_data:
        values = values + 1j * values[::-1]
    source = _normalized(values, axis_values=np.arange(8))
    source.domains = ("frequency",)
    source.normalization_profile = "unrelated-producer-profile"
    source.quadrature_mode = "complex-1d" if complex_data else "real-1d"
    source.branch_order = ("complex",) if complex_data else ("real",)
    axis = source.axes[0]
    axis.domain = "frequency"
    axis.unit = units
    axis.reference.source = "processing"
    axis.reference.observation_frequency = 400.0
    axis.reference.offset = 10.0
    axis.reference.offset_unit = "ppm"
    grid = 4000 - np.arange(8) * 500
    axis.values = grid[::-1] if increasing else grid
    if units == "ppm":
        axis.values = axis.values / 400
    source.acquisition = {
        "spectral_width_hz": 4000.0,
        "observation_frequency_mhz": 400.0,
        "nucleus": "1H",
    }
    return source


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("units", ["Hz", "ppm"])
@pytest.mark.parametrize("increasing", [False, True])
def test_import_preserves_processed_grid_components_and_reference(
    complex_data, units, increasing
):
    raw = _spectrum(complex_data=complex_data, units=units, increasing=increasing)
    source = from_interop(raw)
    assert source.x.units == scp.ur.Unit(units)
    np.testing.assert_array_equal(source.data, raw.data)
    np.testing.assert_allclose(
        source.x.data, raw.axes[0].values, rtol=1e-12, atol=1e-12
    )
    assert np.iscomplexobj(source.data) is complex_data
    assert source.meta.isfreq == [True]
    assert source.meta.data_domain == "frequency"
    assert source.meta.interop_axes[0]["reference"]["source"] == "processing"
    experiment = Experiment(source)
    assert experiment.source_kind == "processed_1d"
    assert experiment._nmr_meta.iscomplex == (complex_data,)
    expected = raw.axes[0].values / 400 if units == "Hz" else raw.axes[0].values
    np.testing.assert_allclose(source.x.to("ppm").data, expected, atol=1e-12)


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("origin", [None, "agilent", "topspin", "unknown"])
def test_frequency_process_does_not_fft_or_reinterpret_origin(
    monkeypatch, complex_data, origin
):
    raw = _spectrum(complex_data=complex_data)
    source = from_interop(raw)
    if origin is not None:
        source.origin = origin
    source.meta.origin = origin
    source.meta.normalization_profile = origin
    source.meta.source_metadata = {
        "procs": {"SF": 999, "OFFSET": -30},
        "procpar": {"rfl": -1},
    }

    def forbidden_fft(*args, **kwargs):
        raise AssertionError("frequency input must not be transformed again")

    monkeypatch.setattr(scp.NDDataset, "fft", forbidden_fft, raising=False)
    result = Experiment(source).process()
    np.testing.assert_array_equal(result.data, source.data)
    np.testing.assert_array_equal(result.x.data, source.x.data)
    assert result.meta.nmr_processing["scp_processing"]["applied"] == {}
    assert Experiment(result).is_frequency_domain
    assert not np.shares_memory(result.data, source.data)


@pytest.mark.parametrize(
    "arguments",
    [
        {"apodization": "em"},
        {"size": 16},
        {"reference": "acquisition"},
        {"phase": "manual", "phc0": 30},
        {"phase": "metadata"},
    ],
)
def test_real_processed_input_refuses_invalid_operations_without_fabricating_imaginary(
    arguments,
):
    source = from_interop(_spectrum(complex_data=False))
    before, grid = source.data.copy(), source.x.data.copy()
    exception = RuntimeError if "apodization" in arguments else ValueError
    with pytest.raises(exception):
        Experiment(source).process(**arguments)
    np.testing.assert_array_equal(source.data, before)
    np.testing.assert_array_equal(source.x.data, grid)
    assert not np.iscomplexobj(source.data)
    assert source.meta.get("nmr_processing") is None


def test_complex_processed_phase_is_explicit_and_persists_without_axis_recalibration(
    tmp_path,
):
    source = from_interop(_spectrum())
    result = Experiment(source).process(phase="manual", phc0=30)
    assert not np.array_equal(result.data, source.data)
    np.testing.assert_array_equal(result.x.data, source.x.data)
    trace = result.meta.nmr_processing["scp_processing"]
    assert "fft" not in trace["applied"]
    assert "phase" in trace["applied"]
    path = tmp_path / "processed.scp"
    result.write(path, confirm=False)
    restored = scp.read(path)
    np.testing.assert_array_equal(restored.data, result.data)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    np.testing.assert_allclose(
        restored.x.to("ppm").data, source.x.data / 400, atol=1e-12
    )
    assert Experiment(restored).is_frequency_domain


@pytest.mark.parametrize("invalid", ["duplicate", "mask", "units", "reference", "mode"])
def test_malformed_processed_axes_or_components_are_rejected(invalid):
    raw = _spectrum()
    if invalid == "duplicate":
        raw.axes[0].values = np.ones(8)
    elif invalid == "mask":
        raw.axes[0].values = np.ma.array(raw.axes[0].values, mask=np.arange(8) == 1)
    elif invalid == "units":
        raw.axes[0].unit = "s"
    elif invalid == "reference":
        raw.axes[0].reference.observation_frequency = True
    else:
        raw.quadrature_mode = "real-1d"
    with pytest.raises(ValueError):
        from_interop(raw)


def test_unsigned_frequency_axis_is_preserved_in_float_coordinates():
    raw = _spectrum()
    raw.data = raw.data[:3]
    raw.axes[0].values = np.array([4000, 3000, 2000], dtype=np.uint64)
    result = from_interop(raw)
    np.testing.assert_array_equal(result.x.data, [4000, 3000, 2000])
    np.testing.assert_allclose(result.x.to("ppm").data, [10, 7.5, 5], atol=1e-12)


@pytest.mark.parametrize("values", [[1, 0, 2], [2**53 + 1, 2**53 + 4, 2**53 + 6]])
def test_unsigned_nonmonotonic_or_unrepresentable_coordinates_refuse(values):
    raw = _spectrum()
    raw.data = raw.data[:3]
    raw.axes[0].values = np.array(values, dtype=np.uint64)
    with pytest.raises(ValueError):
        from_interop(raw)
