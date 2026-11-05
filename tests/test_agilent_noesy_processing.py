"""Synthetic tests for the opt-in Agilent NOESY prototype."""

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy_hypercomplex import as_float_array

from spectrochempy_nmr import (
    extract_noesy_components,
    process_agilent_noesy_states_tppi,
    process_agilent_noesy_states_tppi_channels,
)
from spectrochempy_nmr.processing.complex_channels import process_complex_channels


def _dataset(mask=None):
    rows, points = 6, 8
    row = np.arange(rows, dtype=float)[:, None]
    col = np.arange(points, dtype=float)[None, :]
    data = (
        np.exp(1j * (0.31 * row + 0.17 * col)) * (1.0 + 0.07 * row + 0.013 * col)
    ).astype(np.complex64)
    if mask is not None:
        data = np.ma.array(data, mask=mask)
    ds = scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(rows), scp.Coord.arange(points)],
    )
    ds.meta.update(
        {
            "seqfil": "NOESY",
            "phase": (1, 2),
            "ni": 3,
            "np": 16,
            "arraydim": 6,
            "ntraces": 1,
        }
    )
    return ds


def _reference(raw):
    f2 = raw.shape[1] * np.fft.fftshift(np.fft.ifft(raw, axis=1), axes=1)
    e1, e2 = f2[0::2], f2[1::2]
    sr = np.fft.fftshift(np.fft.fft(e1.real - 1j * e2.real, axis=0), axes=0)
    si = np.fft.fftshift(np.fft.fft(e1.imag - 1j * e2.imag, axis=0), axes=0)
    return np.stack([sr.real, si.real, sr.imag, si.imag], axis=-1)


def _var2pipe_reference(raw):
    return (-1) ** np.arange(raw.shape[0])[:, None] * np.conjugate(raw)


def test_noesy_pipeline_matches_independent_complex_reference():
    source = _dataset()
    expected = _reference(source.data)

    result = process_agilent_noesy_states_tppi(source)

    assert result.data.shape == (3, 8)
    np.testing.assert_allclose(
        as_float_array(result.data), expected, rtol=2e-6, atol=2e-6
    )
    assert result.meta.mathematical_component_order == ("RR", "RI", "IR", "II")
    assert result.meta.processing_profile == "agilent-noesy-states-tppi-experimental"
    assert result.meta.raw_shape == (6, 8)
    assert result.meta.raw_ni == 3
    assert result.meta.raw_np == 16
    assert result.meta.raw_ntraces == 1
    assert result.meta.returned_shape == (3, 8)
    assert result.meta.td == [3, 8]
    assert result.meta.isfreq == [True, True]
    assert result.meta.coordinates_status == "provisional_index"
    assert result.meta.raw_acquisition["arraydim"] == 6
    assert result.coordset[0].size == 8
    assert result.coordset[1].size == 3
    assert source.data.shape == (6, 8)


def test_noesy_pipeline_accepts_explicit_var2pipe_transport():
    source = _dataset()
    result = process_agilent_noesy_states_tppi(source, transport="var2pipe")
    expected = _reference(_var2pipe_reference(source.data))

    assert result.meta.transport == "var2pipe"
    assert result.data.shape == (3, 8)
    np.testing.assert_allclose(
        as_float_array(result.data), expected, rtol=2e-6, atol=2e-6
    )


def test_noesy_pipeline_rejects_unknown_transport():
    with pytest.raises(ValueError, match="transport"):
        process_agilent_noesy_states_tppi(_dataset(), transport="guess")


@pytest.mark.parametrize(
    "changes",
    [
        {"seqfil": "gHSQCAD"},
        {"phase": (1,)},
        {"phase": (True, 2)},
        {"phase": (1.0, 2.0)},
        {"arraydim": 4},
        {"np": 15},
        {"ni": True},
        {"ntraces": 2},
        {"ntraces": None},
    ],
)
def test_noesy_pipeline_rejects_non_noesy_profiles(changes):
    source = _dataset()
    source.meta.update(changes)

    with pytest.raises(ValueError):
        process_agilent_noesy_states_tppi(source)


def test_noesy_pipeline_inplace_returns_none():
    source = _dataset()

    result = process_agilent_noesy_states_tppi(source, inplace=True)

    assert result is None
    assert source.data.shape == (3, 8)


def test_noesy_pipeline_out_of_place_preserves_input_metadata():
    source = _dataset()
    source.meta["marker"] = "input"

    result = process_agilent_noesy_states_tppi(source)

    assert source.data.shape == (6, 8)
    assert source.meta.marker == "input"
    assert result.meta.data_domain == "frequency"


def test_noesy_channels_match_independent_reference_and_have_explicit_contract():
    source = _dataset()
    source.meta.provenance = "Synthetic"
    source.meta.acquisition = {"mode": "normalized"}
    source.meta.normalization = "validated-raw"
    expected = _reference(source.data)

    result = process_agilent_noesy_states_tppi_channels(source)

    assert result.shape == (2, 3, 8)
    assert result.dims == ["c", "y", "x"]
    assert result.dtype.kind == "c"
    assert result.coordset["c"].labels.tolist() == ["S_R", "S_I"]
    assert result.meta.stage == "spectrum"
    assert result.meta.component_axis == "c"
    assert result.meta.axis_roles == ("component", "F1", "F2")
    assert result.meta.component_labels == ("S_R", "S_I")
    assert result.meta.quadrature_mode == "states-tppi"
    assert result.meta.branch_order == ("E1", "E2")
    assert result.meta.f1coef == (1, 0, 0, 0, 0, 0, -1, 0)
    assert result.meta.sequence == "NOESY"
    assert result.meta.vendor == "Agilent"
    assert result.meta.provenance == "Synthetic"
    assert result.meta.acquisition == {"mode": "normalized"}
    assert result.meta.normalization == "validated-raw"
    np.testing.assert_allclose(result.data[0].real, expected[:, :, 0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.data[1].real, expected[:, :, 1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.data[0].imag, expected[:, :, 2], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.data[1].imag, expected[:, :, 3], rtol=2e-6, atol=2e-6)
    assert source.data.shape == (6, 8)


def test_noesy_components_keep_the_explicit_rr_ri_ir_ii_mapping():
    result = process_agilent_noesy_states_tppi_channels(_dataset())
    components = extract_noesy_components(result)

    assert list(components) == ["RR", "RI", "IR", "II"]
    assert all(component.dims == ["y", "x"] for component in components.values())
    np.testing.assert_allclose(components["RR"].data, result.data[0].real)
    np.testing.assert_allclose(components["IR"].data, result.data[0].imag)
    np.testing.assert_allclose(components["RI"].data, result.data[1].real)
    np.testing.assert_allclose(components["II"].data, result.data[1].imag)

    original = result.data[0, 0, 0]
    components["RR"].data[0, 0] = 999
    assert result.data[0, 0, 0] == original


@pytest.mark.parametrize("provenance", ["Bruker", "Agilent", "NMRPipe", "Synthetic"])
def test_common_channel_engine_ignores_vendor_provenance(provenance):
    source = _dataset()
    channels = scp.NDDataset(
        np.stack([source.data[0::2], source.data[1::2]], axis=0),
        coordset=[
            scp.Coord.arange(2, dims="c", labels=["branch_0", "branch_1"]),
            scp.Coord.arange(3, dims="y"),
            scp.Coord.arange(8, dims="x"),
        ],
        dims=["c", "y", "x"],
    )
    channels.meta.update(
        {
            "stage": "acquisition",
            "component_labels": ("branch_0", "branch_1"),
            "component_axis": "c",
            "axis_roles": ("component", "F1", "F2"),
            "quadrature_mode": "states-tppi",
            "branch_order": ("E1", "E2"),
            "f1coef": (1, 0, 0, 0, 0, 0, -1, 0),
            "tppi_alternation_state": "encoded",
            "normalization_state": "normalized",
            "f2_convention": "centered-inverse-unscaled",
            "provenance": provenance,
            "acquisition": {"mode": "normalized"},
            "normalization": "validated-branches",
            "isfreq": [False, False, False],
            "td": [2, 3, 8],
        }
    )

    result = process_complex_channels(
        channels,
        reconstruction="states-tppi",
    )

    assert result.dims == ["c", "y", "x"]
    assert result.meta.stage == "spectrum"
    assert result.meta.component_labels == ("S_R", "S_I")
    assert result.meta.provenance == provenance
    assert result.meta.acquisition == {"mode": "normalized"}
    assert result.meta.normalization == "validated-branches"
    expected = _reference(source.data)
    np.testing.assert_allclose(result.data[0].real, expected[:, :, 0], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.data[1].real, expected[:, :, 1], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.data[0].imag, expected[:, :, 2], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(result.data[1].imag, expected[:, :, 3], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "spectrum"},
        {"component_labels": ("branch_1", "branch_0")},
        {"component_axis": "y"},
        {"axis_roles": ("F1", "component", "F2")},
        {"isfreq": [False, True, False]},
        {"td": [2, 3, 7]},
        {"encoding": "states-tppi"},
        {"quadrature_mode": "unknown"},
    ],
)
def test_common_channel_engine_rejects_non_acquisition_contract(changes):
    source = _dataset()
    channels = scp.NDDataset(
        np.stack([source.data[0::2], source.data[1::2]], axis=0),
        coordset=[
            scp.Coord.arange(2, dims="c", labels=["branch_0", "branch_1"]),
            scp.Coord.arange(3, dims="y"),
            scp.Coord.arange(8, dims="x"),
        ],
        dims=["c", "y", "x"],
    )
    channels.meta.update(
        {
            "stage": "acquisition",
            "component_labels": ("branch_0", "branch_1"),
            "component_axis": "c",
            "axis_roles": ("component", "F1", "F2"),
            "quadrature_mode": "states-tppi",
            "branch_order": ("E1", "E2"),
            "f1coef": (1, 0, 0, 0, 0, 0, -1, 0),
            "tppi_alternation_state": "encoded",
            "normalization_state": "normalized",
            "f2_convention": "centered-inverse-unscaled",
            "isfreq": [False, False, False],
            "td": [2, 3, 8],
        }
    )
    channels.meta.update(changes)

    with pytest.raises(ValueError):
        process_complex_channels(channels, reconstruction="states-tppi")


def test_common_channel_engine_runs_both_ffts_on_the_3d_dataset(monkeypatch):
    source = _dataset()
    channels = scp.NDDataset(
        np.stack([source.data[0::2], source.data[1::2]], axis=0),
        coordset=[
            scp.Coord.arange(2, dims="c", labels=["branch_0", "branch_1"]),
            scp.Coord.arange(3, dims="y"),
            scp.Coord.arange(8, dims="x"),
        ],
        dims=["c", "y", "x"],
    )
    channels.meta.update(
        {
            "stage": "acquisition",
            "component_labels": ("branch_0", "branch_1"),
            "component_axis": "c",
            "axis_roles": ("component", "F1", "F2"),
            "quadrature_mode": "states-tppi",
            "branch_order": ("E1", "E2"),
            "f1coef": (1, 0, 0, 0, 0, 0, -1, 0),
            "tppi_alternation_state": "encoded",
            "normalization_state": "normalized",
            "f2_convention": "centered-inverse-unscaled",
            "isfreq": [False, False, False],
            "td": [2, 3, 8],
        }
    )
    calls = []
    original_fft = scp.fft

    def recording_fft(dataset, *, dim):
        calls.append((dataset.shape, list(dataset.dims), dim))
        return original_fft(dataset, dim=dim)

    monkeypatch.setattr(scp, "fft", recording_fft)
    process_complex_channels(
        channels,
        reconstruction="states-tppi",
    )

    assert calls == [
        ((2, 3, 8), ["c", "y", "x"], "x"),
        ((2, 3, 8), ["c", "y", "x"], "y"),
    ]


def test_noesy_channels_reject_masked_time_domain_input():
    mask = np.zeros((6, 8), dtype=bool)
    mask[0, 0] = True
    source = _dataset(mask)

    with pytest.raises(ValueError, match="unmasked input"):
        process_agilent_noesy_states_tppi_channels(source)


def test_legacy_noesy_pipeline_keeps_existing_masked_input_behavior():
    mask = np.zeros((6, 8), dtype=bool)
    mask[0, 0] = True
    source = _dataset(mask)

    result = process_agilent_noesy_states_tppi(source)

    np.testing.assert_allclose(
        as_float_array(result.data), _reference(source.data), rtol=2e-6, atol=2e-6
    )


def test_noesy_spectral_masks_survive_components_and_persistence(tmp_path):
    result = process_agilent_noesy_states_tppi_channels(_dataset())
    mask = np.zeros(result.shape, dtype=bool)
    mask[0, 1, 2] = True
    mask[1, 2, 3] = True
    result.mask = mask

    components = extract_noesy_components(result)
    assert components["RR"].mask[1, 2]
    assert components["IR"].mask[1, 2]
    assert components["RI"].mask[2, 3]
    assert components["II"].mask[2, 3]

    path = result.save_as(tmp_path / "noesy_channels", confirm=False)
    loaded = scp.NDDataset.load(path)
    np.testing.assert_allclose(loaded.data, result.data)
    np.testing.assert_array_equal(loaded.mask, result.mask)
    assert loaded.dims == ["c", "y", "x"]
    assert list(loaded.meta.component_labels) == ["S_R", "S_I"]
