"""Tests for the vendor-neutral complex-channel engine."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.processing.complex_channels import process_complex_channels


def _normalized_channels(shape=(3, 8), dtype=np.complex128):
    row = np.arange(shape[0], dtype=float)[:, None]
    col = np.arange(shape[1], dtype=float)[None, :]
    raw = np.stack(
        [
            np.exp(1j * (0.2 * row + 0.1 * col)),
            np.exp(1j * (0.3 * row - 0.15 * col)),
        ]
    ).astype(dtype)
    dataset = scp.NDDataset(
        raw,
        coordset=[
            scp.Coord.arange(2, dims="c", labels=["branch_0", "branch_1"]),
            scp.Coord.arange(shape[0], dims="y"),
            scp.Coord.arange(shape[1], dims="x"),
        ],
        dims=["c", "y", "x"],
    )
    dataset.meta.update(
        {
            "stage": "acquisition",
            "component_axis": "c",
            "axis_roles": ("component", "F1", "F2"),
            "component_labels": ("branch_0", "branch_1"),
            "quadrature_mode": "states-tppi",
            "branch_order": ("E1", "E2"),
            "f1coef": (1, 0, 0, 0, 0, 0, -1, 0),
            "tppi_alternation_state": "encoded",
            "normalization_state": "normalized",
            "f2_convention": "centered-inverse-unscaled",
            "isfreq": [False, False, False],
            "td": list(raw.shape),
            "provenance": "Synthetic",
        }
    )
    return dataset


def test_engine_preserves_normalized_metadata_and_input():
    source = _normalized_channels()
    before = source.data.copy()

    result = process_complex_channels(source, reconstruction="states-tppi")

    assert result.shape == (2, 3, 8)
    assert result.dims == ["c", "y", "x"]
    assert result.meta.provenance == "Synthetic"
    assert result.meta.stage == "spectrum"
    assert result.meta.component_labels == ("S_R", "S_I")
    np.testing.assert_array_equal(source.data, before)


@pytest.mark.parametrize("sequence", ["NOESY", "COSY", "TOCSY", "synthetic"])
def test_engine_ignores_sequence_identity_after_normalization(sequence):
    source = _normalized_channels()
    source.meta.sequence = sequence
    candidate = process_complex_channels(source, reconstruction="states-tppi")

    reference = _normalized_channels()
    reference.meta.sequence = "reference"
    expected = process_complex_channels(reference, reconstruction="states-tppi")

    np.testing.assert_array_equal(candidate.data, expected.data)


@pytest.mark.parametrize("shape", [(3, 7), (3, 8), (4, 7), (4, 8)])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_engine_matches_independent_fft_reference(shape, dtype):
    source = _normalized_channels(shape, dtype)
    before = source.copy()
    branches = shape[1] * np.fft.fftshift(np.fft.ifft(source.data, axis=2), axes=2)
    interferograms = np.stack(
        [
            branches[0].real - 1j * branches[1].real,
            branches[0].imag - 1j * branches[1].imag,
        ]
    )
    expected = np.fft.fftshift(np.fft.fft(interferograms, axis=1), axes=1)

    result = process_complex_channels(source, reconstruction="states-tppi")

    tolerance = 2e-6 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(result.data, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(source.data, before.data)
    assert source.meta == before.meta
    assert result.shape == (2, *shape)
    assert result.dtype.kind == "c"
    assert result.meta.isfreq == [False, True, True]
    assert result.coordset["c"].labels.tolist() == ["S_R", "S_I"]
