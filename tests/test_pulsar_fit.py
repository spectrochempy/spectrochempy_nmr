"""Fitting and coordinate conventions for PULSAR."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy_nmr import fit_central_transition
from spectrochempy_nmr.pulsar import _central_transition_profile
from spectrochempy_nmr.pulsar_fit import CentralParameters


@pytest.mark.parametrize("units", ["kHz", "ppm"])
def test_pulsar_fit_preserves_axis_and_recovers_parameters(units):
    larmor, width, points, accuracy = 100e6, 20e3, 4096, 8
    truth = CentralParameters(2.6e6, 0.55, 725, 115)
    offset, mass = _central_transition_profile(
        cq_hz=truth.cq_hz,
        eta=truth.eta,
        shift_hz=truth.shift_hz,
        larmor_hz=larmor,
        spectral_width_hz=width,
        points=points,
        accuracy=accuracy,
        spinning_hz=8000,
    )
    step = offset[1] - offset[0]
    line = gaussian_filter1d(
        mass, truth.gaussian_fwhm_hz / (np.sqrt(8 * np.log(2)) * step)
    )
    target = np.linspace(4000, -5000, 1201)
    origin = 2000 if units == "kHz" else 0
    coordinate = (
        (target + origin) / 1000 if units == "kHz" else target / (larmor * 1e-6)
    )
    data = NDDataset(np.zeros(target.size))
    data.set_coordset(x=Coord(coordinate, units=units))
    # Use the coordinates as stored by Coord, which can round decimal values.
    offsets = data.x.data * (1000 if units == "kHz" else larmor * 1e-6) - origin
    measured = (
        2500 * np.interp(offsets, offset + step / 2, line) + 0.05 + offsets * 1e-5
    )
    data.data = measured
    data.mask = np.arange(target.size) == 600

    result = fit_central_transition(
        data,
        initial=CentralParameters(2.5e6, 0.5, 680, 100),
        bounds=(
            CentralParameters(2e6, 0.05, 0, 30),
            CentralParameters(3e6, 0.95, 1500, 220),
        ),
        larmor_hz=larmor,
        spectral_width_hz=width,
        origin_hz=origin,
        points=points,
        accuracy=accuracy,
        spinning_hz=8000,
        window_hz=(-4000, 3500),
    )

    assert result.parameters.cq_hz == pytest.approx(truth.cq_hz, abs=150)
    assert result.parameters.eta == pytest.approx(truth.eta, abs=0.0005)
    assert result.parameters.shift_hz == pytest.approx(truth.shift_hz, abs=0.2)
    assert result.parameters.gaussian_fwhm_hz == pytest.approx(
        truth.gaussian_fwhm_hz, abs=0.3
    )
    assert result.evaluations > 1
    assert result.rms < 1e-3
    for dataset in (result.fitted, result.residuals):
        assert dataset.x.units == data.x.units
        np.testing.assert_array_equal(dataset.x.data, data.x.data)
        np.testing.assert_array_equal(dataset.mask, data.mask)
    np.testing.assert_allclose(result.fitted.data + result.residuals.data, measured)


@pytest.mark.parametrize(
    ("initial", "lower", "error"),
    [
        (
            CentralParameters(0, 0.5, 0, 100),
            CentralParameters(0, 0, -500, 20),
            "initial Cq",
        ),
        (
            CentralParameters(2e6, 0.5, 0, 0),
            CentralParameters(0, 0, -500, 0),
            "minimum Gaussian FWHM",
        ),
    ],
)
def test_fit_rejects_flat_initial_directions(initial, lower, error):
    axis = np.linspace(-2000, 2000, 101)
    dataset = NDDataset(np.exp(-((axis / 100) ** 2)))
    dataset.set_coordset(x=Coord(axis, units="Hz"))
    with pytest.raises(ValueError, match=error):
        fit_central_transition(
            dataset,
            initial=initial,
            bounds=(lower, CentralParameters(4e6, 1, 500, 200)),
            larmor_hz=100e6,
            spectral_width_hz=8000,
        )
