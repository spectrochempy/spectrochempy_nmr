"""Tests for the explicitly supported quadrature reconstruction."""

import numpy as np
import pytest

from spectrochempy_nmr.processing.quadrature import reconstruct_f1_interferograms


def test_noesy_states_tppi_reconstruction_keeps_branch_associations():
    branches = np.array(
        [
            [[1 + 2j, 3 + 4j]],
            [[5 + 6j, 7 + 8j]],
        ]
    )

    result = reconstruct_f1_interferograms(
        branches,
        profile="states-tppi",
    )

    expected = np.stack(
        [branches[0].real - 1j * branches[1].real,
         branches[0].imag - 1j * branches[1].imag]
    )
    np.testing.assert_array_equal(result, expected)


def test_quadrature_rejects_unvalidated_profiles():
    with pytest.raises(ValueError, match="unsupported"):
        reconstruct_f1_interferograms(
            np.zeros((2, 1, 1), dtype=complex),
            profile="echo-antiecho",
        )
