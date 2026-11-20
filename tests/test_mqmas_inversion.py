"""Regression checks for constrained population inversion through the plugin API."""

import numpy as np
import pytest
import spectrochempy as scp


@pytest.mark.parametrize("column_scale", [0.1, 1, 10, 1000])
@pytest.mark.parametrize("normalize_columns", [False, True])
def test_local_population_constraint_preserves_physical_units(
    column_scale, normalize_columns
):
    inversion = scp.nmr.mqmas_inversion
    axis = np.arange(3.0)
    spectrum = scp.NDDataset(np.arange(1.0, 10.0).reshape(3, 3))
    spectrum.set_coordset(
        y=scp.Coord(axis, units="ppm"), x=scp.Coord(axis, units="ppm")
    )
    spectrum.mask = np.eye(3, dtype=bool)[::-1]
    spectrum.mask[0, 0] = False
    kernel = inversion.Kernel((axis, axis), axis, axis, column_scale * np.eye(9))
    result = inversion.fit(
        spectrum,
        kernel,
        regularization=0,
        normalize_columns=normalize_columns,
        local_region=((0, 0), (0, 0)),
        local_weight=0.1,
    )
    expected = spectrum.data / (9 * column_scale)
    expected[spectrum.mask] = 0
    expected[0, 0] = 0.1
    np.testing.assert_allclose(result.population.data, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        result.amplitude.data, expected * column_scale, rtol=1e-8, atol=1e-10
    )
    np.testing.assert_array_equal(result.residual.mask, spectrum.mask)
    selected = ~spectrum.mask
    np.testing.assert_allclose(
        result.reconstructed.data[selected],
        (expected * column_scale * 9)[selected],
        rtol=1e-8,
        atol=1e-10,
    )
