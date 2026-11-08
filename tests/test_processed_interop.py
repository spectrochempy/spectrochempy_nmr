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


def _real_plane():
    source = _spectrum(complex_data=False)
    source.data = np.arange(24, dtype=np.float32).reshape(3, 8)
    source.dimensions = source.physical_dimensions = ("F1", "F2")
    source.domains = ("frequency", "frequency")
    source.encoding = ("real", "real")
    source.quadrature_mode = "real-2d"
    indirect = _spectrum(complex_data=False, increasing=True, units="ppm").axes[0]
    indirect.values = np.array([1.0, 2.0, 3.0005])
    indirect.reference.observation_frequency = 100.0
    source.axes = (indirect, *source.axes)
    return source


@pytest.mark.parametrize("transposed", [False, True])
def test_import_real_plane_preserves_order_precision_and_independent_storage(
    transposed,
):
    source = _real_plane()
    if transposed:
        source.data = source.data.T
        source.axes = source.axes[::-1]
        source.dimensions = source.physical_dimensions = source.dimensions[::-1]
    dataset = from_interop(source)
    assert dataset.dims == ["y", "x"]
    assert dataset.dtype == source.data.dtype
    np.testing.assert_array_equal(dataset.data, source.data)
    assert not np.shares_memory(dataset.data, source.data)
    assert dataset.meta.axis_roles == source.dimensions
    assert dataset.meta.isfreq == [True, True]
    assert dataset.meta.td == list(source.data.shape)
    for coord, axis in zip((dataset.y, dataset.x), source.axes, strict=True):
        np.testing.assert_allclose(coord.data, axis.values, rtol=1e-12, atol=1e-12)
        assert coord.units == scp.ur.Unit(axis.unit)
        assert (
            coord.meta.acquisition_frequency.magnitude
            == axis.reference.observation_frequency
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("data", np.ones((3, 8), dtype=complex)),
        ("data", np.full((3, 8), np.nan)),
        ("domains", ("time", "frequency")),
        ("encoding", ("complex", "real")),
        ("dimensions", ("F1", "F1")),
        ("physical_dimensions", ("F2", "F1")),
        ("branch_order", ("rr",)),
        ("channel_axis", 0),
    ],
)
def test_import_real_plane_rejects_unsupported_contract(field, value):
    source = _real_plane()
    setattr(source, field, value)
    with pytest.raises(ValueError, match="from_interop 2D requires"):
        from_interop(source)


def _complex_plane():
    source = _real_plane()
    source.data = np.stack((source.data, -source.data - 1))
    source.dimensions = ("channels", *source.physical_dimensions)
    source.encoding = ("real", "complex")
    source.quadrature_mode = "stored-components-2d"
    source.branch_order = ("rr", "ri")
    source.channel_axis = 0
    return source


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_import_singly_complex_plane_preserves_components(dtype):
    source = _complex_plane()
    source.data = source.data.astype(dtype)
    source.data[1, 0, 0] = -0.0
    dataset = from_interop(source)
    assert dataset.dtype == np.result_type(dtype, np.complex64)
    assert dataset.meta.interop_complex_axis == "F2"
    assert dataset.meta.axis_roles == ("F1", "F2")
    assert dataset.meta.td == [3, 8]
    np.testing.assert_array_equal(dataset.data.real, source.data[0])
    np.testing.assert_array_equal(dataset.data.imag, source.data[1])
    assert np.signbit(dataset.data.imag[0, 0])
    assert not np.shares_memory(dataset.data, source.data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoding", ("complex", "complex")),
        ("branch_order", ("ri", "rr")),
        ("branch_order", ("rr", "ir")),
        ("channel_axis", 1),
        ("data", np.zeros((4, 3, 8), dtype=np.float32)),
        ("data", np.zeros((2, 3, 8), dtype=np.int32)),
        ("domains", ("time", "frequency")),
        ("dimensions", ("channels", "F2", "F1")),
    ],
)
def test_import_singly_complex_plane_rejects_unsupported_contract(field, value):
    source = _complex_plane()
    setattr(source, field, value)
    with pytest.raises(ValueError, match="from_interop 2D requires"):
        from_interop(source)


def _four_component_plane():
    source = _complex_plane()
    source.data = np.concatenate((source.data, source.data * -3 + 5))
    source.encoding = ("complex", "complex")
    source.branch_order = ("rr", "ri", "ir", "ii")
    return source


def test_import_four_component_plane_preserves_labelled_real_planes():
    source = _four_component_plane()
    source.data[3, 0, 0] = -0.0
    dataset = from_interop(source)
    assert dataset.dims == ["c", "y", "x"]
    assert dataset.dtype == source.data.dtype
    np.testing.assert_array_equal(dataset.data, source.data)
    assert np.signbit(dataset.data[3, 0, 0])
    assert not np.shares_memory(dataset.data, source.data)
    assert tuple(dataset.c.labels) == source.branch_order
    assert dataset.meta.component_axis == "c"
    assert dataset.meta.component_labels == source.branch_order
    assert dataset.meta.axis_roles == ("component", "F1", "F2")
    assert dataset.meta.isfreq == [False, True, True]
    assert dataset.meta.td == [4, 3, 8]
    assert dataset.meta.get("interop_complex_axis") is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("branch_order", ("rr", "ir", "ri", "ii")),
        ("encoding", ("real", "complex")),
        ("domains", ("frequency", "time")),
        ("channel_axis", 2),
        ("data", np.zeros((4, 3, 8), dtype=np.complex64)),
        ("data", np.full((4, 3, 8), np.nan, dtype=np.float32)),
    ],
)
def test_import_four_component_plane_rejects_unsupported_contract(field, value):
    source = _four_component_plane()
    setattr(source, field, value)
    with pytest.raises(ValueError, match="from_interop 2D requires"):
        from_interop(source)


@pytest.mark.parametrize("dtype", [np.int32, np.float64])
def test_import_bruker_named_planes_without_reinterpreting_labels(dtype):
    source = _four_component_plane()
    source.encoding = ("stored", "stored")
    source.branch_order = ("2rr", "2ri", "2ir", "2ii")
    source.data = source.data.astype(dtype)
    dataset = from_interop(source)
    assert tuple(dataset.c.labels) == source.branch_order
    assert dataset.dtype == source.data.dtype
    np.testing.assert_array_equal(dataset.data, source.data)
    source.branch_order = ("2rr", "2ir", "2ri", "2ii")
    with pytest.raises(ValueError, match="from_interop 2D requires"):
        from_interop(source)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("rows", [1, 3])
def test_import_normalized_qf_fid_preserves_axes_and_samples(transposed, rows):
    source = _complex_plane()
    source.data = source.data[:, :rows]
    source.domains = ("time", "time")
    source.acquisition["indirect_sampling"] = "qf"
    for axis, size, width in zip(source.axes, (rows, 8), (1000, 4000), strict=True):
        axis.values = np.arange(size) / width
        axis.unit = "s"
        axis.domain = "time"
        axis.direction = "increasing"
    if transposed:
        source.data = source.data.transpose(0, 2, 1)
        source.axes = source.axes[::-1]
        source.physical_dimensions = source.physical_dimensions[::-1]
        source.dimensions = ("channels", *source.physical_dimensions)
        source.encoding = ("complex", "real")
        source.branch_order = ("rr", "ir")
    dataset = from_interop(source)
    assert dataset.meta.axis_roles == source.physical_dimensions
    assert dataset.meta.isfreq == [False, False]
    assert dataset.meta.data_domain == "time"
    assert dataset.meta.interop_complex_axis == "F2"
    assert dataset.meta.acquisition["indirect_sampling"] == "qf"
    assert dataset.dtype == np.complex64
    np.testing.assert_array_equal(dataset.data.real, source.data[0])
    np.testing.assert_array_equal(dataset.data.imag, source.data[1])
    for coordinate, axis in zip((dataset.y, dataset.x), source.axes, strict=True):
        assert coordinate.units == scp.ur.s
        np.testing.assert_array_equal(coordinate.data, axis.values)


def test_import_time_plane_requires_qf_marker():
    source = _complex_plane()
    source.domains = ("time", "time")
    with pytest.raises(ValueError, match="requires a normalized"):
        from_interop(source)
