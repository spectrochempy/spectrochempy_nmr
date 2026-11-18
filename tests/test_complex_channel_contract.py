"""Acquisition-contract checks for the common complex-channel engine."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.processing.complex_channels import process_complex_channels
from test_complex_channels import _normalized_channels


def _dataset():
    source = _normalized_channels()
    raw = np.empty((6, 8), dtype=source.dtype)
    raw[::2], raw[1::2] = source.data
    return scp.NDDataset(raw)


def _reference(raw):
    f2 = raw.shape[1] * np.fft.fftshift(np.fft.ifft(raw, axis=1), axes=1)
    e1, e2 = f2[0::2], f2[1::2]
    sr = np.fft.fftshift(np.fft.fft(e1.real - 1j * e2.real, axis=0), axes=0)
    si = np.fft.fftshift(np.fft.fft(e1.imag - 1j * e2.imag, axis=0), axes=0)
    return np.stack([sr.real, si.real, sr.imag, si.imag], axis=-1)


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
    np.testing.assert_allclose(
        result.data[0].real, expected[:, :, 0], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        result.data[1].real, expected[:, :, 1], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        result.data[0].imag, expected[:, :, 2], rtol=2e-6, atol=2e-6
    )
    np.testing.assert_allclose(
        result.data[1].imag, expected[:, :, 3], rtol=2e-6, atol=2e-6
    )


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
