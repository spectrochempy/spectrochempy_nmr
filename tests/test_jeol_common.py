"""Explicit quadrature at reading, common processing independent of provenance."""

import sys
from types import SimpleNamespace

from nmrglue import interop, jeol
import numpy as np
import pytest
import spectrochempy as scp


@pytest.fixture
def qf_jdf(tmp_path, monkeypatch):
    ny, nx = 8, 16
    y, x = np.indices((ny, nx))
    data = np.exp(-y / 7 - x / 15) * np.exp(1j * (0.71 * x - 1.03 * y + 0.4))
    data += 0.37 * np.exp(1j * (-0.43 * x + 0.79 * y - 0.9))
    header = {
        "data_format": "two_d",
        "data_type": "float64",
        "data_points": [nx, ny],
        "data_offset_start": [0, 0],
        "data_offset_stop": [nx - 1, ny - 1],
        "data_axis_start": [0, 0],
        "data_axis_stop": [(nx - 1) / 16, (ny - 1) / 8],
        "base_freq": [400, 400],
        "units": [("none", 1, "second")] * 2,
        "data_axis_type": ["real_complex", "real_complex"],
        "data_axis_ranged": ["ranged", "ranged"],
        "reversed": [False, True],
    }
    params = {
        "x_points": nx,
        "y_points": ny,
        "x_sweep": 16,
        "y_sweep": 8,
        "x_freq": 400e6,
        "y_freq": 400e6,
        "x_offset": 5.0,
        "y_offset": 4.0,
        "x_domain": "1H",
        "y_domain": "1H",
        "experiment": "unrelated_name",
    }
    stored = SimpleNamespace(
        format="jeol-2d-storage", metadata={"header": header, "parameters": params}
    )
    normalized = interop.NormalizedAcquisition(
        data=data,
        axes=tuple(
            interop.Axis(
                values=np.arange(n) / width,
                unit="s",
                label="1H",
                domain="time",
                direction="increasing",
            )
            for n, width in ((ny, 8), (nx, 16))
        ),
        dimensions=("F1", "F2"),
        physical_dimensions=("F1", "F2"),
        domains=("time", "time"),
        encoding=("real_complex", "real_complex"),
        quadrature_mode="jeol-stored-section-pairs",
        branch_order=("section-0/1",),
        channel_axis=None,
        normalization_state="storage-normalized",
        normalization_profile="jeol-two-sections",
        acquisition={
            "spectral_width_hz": (8.0, 16.0),
            "sampling": ("regular", "regular"),
            "nominal_shape": (ny, nx),
        },
        applied_operations=(),
        provenance={},
        metadata={},
    )
    monkeypatch.setattr(jeol, "read", lambda *a, **kw: (stored.metadata, data.copy()))
    path = tmp_path / "input.jdf"
    path.touch()
    return path, stored, normalized, data


@pytest.mark.parametrize("shape", [(8, 16), (12, 20), (9, 17)])
def test_signed_complex_dft_and_axes(qf_jdf, shape):
    path, stored, _, values = qf_jdf
    raw = scp.nmr.read(path, quadrature="qf")
    before = raw.copy()
    actual = scp.nmr.Experiment(raw).process(size=shape)
    # Independent explicit DFT matrices: positive F2, negative F1 then conjugate.
    ny, nx = shape
    f2 = np.exp(2j * np.pi * np.outer(np.arange(nx), np.arange(values.shape[1])) / nx)
    f1 = np.exp(-2j * np.pi * np.outer(np.arange(ny), np.arange(values.shape[0])) / ny)
    expected = np.fft.fftshift((f1 @ values @ f2.T).conj())
    if ny % 2 == 0:
        expected = np.roll(expected, -1, axis=0)
    np.testing.assert_allclose(actual.data, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual.y.data, 1600 + (ny // 2 - int(ny % 2 == 0) - np.arange(ny)) * 8 / ny
    )
    np.testing.assert_allclose(
        actual.x.data, 2000 - 0.5 + (nx // 2 - np.arange(nx)) * 16 / nx
    )
    np.testing.assert_array_equal(raw.data, before.data)
    assert raw.meta == before.meta
    assert actual.meta.stage == "spectrum"
    assert scp.nmr.Experiment(actual).is_frequency_domain
    assert stored.metadata["parameters"]["experiment"] == "unrelated_name"


def test_read_save_process_without_provider_or_source(qf_jdf, tmp_path, monkeypatch):
    path, _, _, _ = qf_jdf
    raw = scp.nmr.read(path, quadrature="qf")
    expected = scp.nmr.Experiment(raw).process(size=(12, 20))
    raw.meta.source_metadata = {}
    raw.meta.provenance = {"vendor": "other"}
    raw.origin = "other"
    raw.write(tmp_path / "raw.scp", confirm=False)
    monkeypatch.setitem(sys.modules, "nmrglue", None)
    saved = scp.nmr.read(tmp_path / "raw.scp")
    actual = scp.nmr.Experiment(saved).process(size=(12, 20))
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)
    np.testing.assert_array_equal(actual.x.data, expected.x.data)


@pytest.mark.parametrize(
    "damage", ["listed", "four-sections", "reverse", "calibration"]
)
def test_explicit_mode_does_not_override_unsupported_storage(qf_jdf, damage):
    from spectrochempy.utils.exceptions import SpectroChemPyError

    path, stored, _, _ = qf_jdf
    if damage == "listed":
        stored.metadata["header"]["data_axis_ranged"][1] = "listed"
    elif damage == "four-sections":
        stored.metadata["header"]["data_axis_type"] = ["complex", "complex"]
    elif damage == "reverse":
        stored.metadata["header"]["reversed"][0] = True
    else:
        stored.metadata["parameters"]["x_freq"] = np.nan
    with pytest.raises(SpectroChemPyError):
        scp.nmr.read(path, quadrature="qf")


@pytest.mark.parametrize("size", [True, (7, 16), (8, 15), (8, 16.0), (8,), -2])
def test_invalid_fft_size_rejected_without_mutation(qf_jdf, size):
    raw = scp.nmr.read(qf_jdf[0], quadrature="qf")
    before = raw.copy()
    with pytest.raises(ValueError, match="QF"):
        scp.nmr.Experiment(raw).process(size=size)
    np.testing.assert_array_equal(raw.data, before.data)
    assert raw.meta == before.meta


def test_symmetry_uses_physical_frequencies():
    ds = scp.NDDataset(
        [[0.0, 1.0, 4.0], [3.0, 2.0, 5.0]],
        coordset=[
            scp.Coord([0.0, 2.0], units="Hz"),
            scp.Coord([0.0, 1.0, 2.0], units="Hz"),
        ],
    )
    ds.meta.update({"axis_roles": ("F1", "F2"), "data_domain": "frequency"})
    # Use the processing function for a neutral real frequency plane.
    from spectrochempy_nmr.processing.symmetrize import symmetrize_minimum

    result = symmetrize_minimum(ds, frequency_grid=([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]))
    np.testing.assert_allclose(result.data, result.data.T)
    assert result.data[0, 2] == 3
    assert result.data[0, 1] == 1
    outside = symmetrize_minimum(ds, frequency_grid=([-1.0, 0.0, 2.0], [0.0, 1.0, 3.0]))
    assert outside.data[0, 0] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("frequency_offset_hz", (0, float("nan"))),
        ("acquisition_frequency_mhz", (400, 0)),
        ("nyquist_edge", ("negative", "unknown")),
    ],
)
def test_invalid_physical_reference_rejected(qf_jdf, field, value):
    raw = scp.nmr.read(qf_jdf[0], quadrature="qf")
    raw.meta.acquisition[field] = value
    with pytest.raises(ValueError, match="QF"):
        scp.nmr.Experiment(raw).process()


def test_magnitude_spectrum_remains_supported_after_save(qf_jdf, tmp_path):
    raw = scp.nmr.read(qf_jdf[0], quadrature="qf")
    spectrum = scp.nmr.Experiment(raw).process()
    result = scp.nmr.Experiment(abs(spectrum)).symmetrize()
    result.write(tmp_path / "magnitude.scp", confirm=False)
    saved = scp.nmr.read(tmp_path / "magnitude.scp")
    assert scp.nmr.Experiment(saved).is_frequency_domain
    np.testing.assert_array_equal(saved.data, result.data)


@pytest.mark.parametrize("damage", ["complex", "negative", "mask", "grid"])
def test_invalid_magnitude_input_rejected(qf_jdf, damage):
    raw = scp.nmr.read(qf_jdf[0], quadrature="qf")
    spectrum = scp.nmr.Experiment(raw).process()
    magnitude = abs(spectrum)
    if damage == "complex":
        magnitude = spectrum
    elif damage == "negative":
        magnitude.data[0, 0] = -1
    elif damage == "mask":
        mask = np.zeros(magnitude.shape, dtype=bool)
        mask[0, 0] = True
        magnitude.mask = mask
    else:
        magnitude.y.data = np.zeros(magnitude.shape[0])
    with pytest.raises(ValueError):
        scp.nmr.Experiment(magnitude).symmetrize()
