"""Integration tests for the bounded, two-stage Czjzek fit."""

import numpy as np
import pytest

from spectrochempy_nmr import CzjzekDistribution
from spectrochempy_nmr import CzjzekRFLibrary
from spectrochempy_nmr import fit_czjzek
from spectrochempy_nmr import simulate_czjzek
from spectrochempy_nmr.pulsar import Pulse


@pytest.mark.parametrize("bandwidth", [None, 8e3])
def test_two_stage_retains_direct_result_and_tracks_library_fallbacks(bandwidth):
    events = (Pulse(0.5e-6, 50e3),)
    settings = dict(
        spin=2.5,
        larmor_hz=195e6,
        spectral_width_hz=16e3,
        points=129,
        accuracy=1,
        probe_bandwidth_hz=bandwidth,
    )
    library = CzjzekRFLibrary.build(
        events,
        cq_grid_hz=[2e6, 2.2e6],
        eta_grid=[0.1, 0.5],
        shift_hz=0,
        shift_bins=3,
        **settings,
    )

    def distribution(amplitude):
        return CzjzekDistribution(0, 2e6, 0.5, shift_hz=amplitude)

    observed = simulate_czjzek(distribution(500), events=events, **settings).data

    def objective(parameters, simulate):
        (shift,) = parameters
        difference = simulate(distribution(shift)).data - observed
        return np.vdot(difference, difference).real + (shift - 500) ** 2

    result = fit_czjzek(
        objective, [300], [(0, 750)], events=events, library=library, **settings
    )
    assert result.interpolated.optimization.success
    assert 0 < result.interpolated.propagations < result.interpolated.optimization.nfev
    assert result.direct.optimization.success
    assert result.direct.propagations == result.direct.optimization.nfev
    np.testing.assert_allclose(result.direct.optimization.x, [500], atol=1e-4)

    direct = fit_czjzek(objective, [300], [(0, 750)], events=events, **settings)
    assert direct.interpolated is None
    np.testing.assert_allclose(
        result.direct.optimization.x, direct.direct.optimization.x, atol=1e-4
    )

    with pytest.raises(ValueError, match="does not match"):
        fit_czjzek(
            objective,
            [300],
            [(0, 750)],
            events=(Pulse(0.5e-6, 30e3),),
            library=library,
            **settings,
        )
