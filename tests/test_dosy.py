"""Scientific checks for the public experimental DOSY map workflow."""

import numpy as np
import pytest

import spectrochempy as scp


def mixture(*, noise=0, reverse=False):
    b = np.linspace(0, 8e10, 32)
    ppm = np.linspace(4, 0, 101)
    lines = np.array(
        [
            np.exp(-(((ppm - 2.6) / 0.23) ** 2)),
            0.8 * np.exp(-(((ppm - 2.3) / 0.23) ** 2)),
        ]
    )
    values = np.exp(-b[:, None] * np.array([2e-11, 1e-10])) @ lines
    values += np.random.default_rng(42).normal(0, noise, values.shape)
    if reverse:
        b, ppm, values = b[::-1], ppm[::-1], values[::-1, ::-1]
    data = scp.NDDataset(
        values,
        coordset=[
            scp.Coord(b, units="s/m^2", rounding=False),
            scp.Coord(ppm, units="ppm"),
        ],
        units="count",
    )
    return data


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("noise", [0, 0.003])
def test_mixture_map_recovers_known_motion_and_reconstructs(noise, reverse):
    spectra = mixture(noise=noise, reverse=reverse)
    before = spectra.data.copy()
    result = scp.nmr.Diffusion(spectra, b=spectra.y).dosy(
        region=(3.4, 1.4), D=(3e-12, 3e-10)
    )
    assert result.map.y.units == scp.ur("m^2/s").units
    assert result.map.x.units == scp.ur.ppm
    assert result.map.units == spectra.units
    assert np.all(result.map.data >= 0)
    # The peaks overlap; at the outer shoulders each species dominates.
    # Assert relative accuracy explicitly: default pytest abs=1e-12 is too lax.
    for ppm, expected in [(2.8, 2e-11), (2.1, 1e-10)]:
        column = np.argmin(abs(result.map.x.data - ppm))
        mode = result.map.y.data[np.argmax(result.map.data[:, column])]
        assert mode == pytest.approx(expected, rel=0.15, abs=0)
    np.testing.assert_allclose(
        result.measured.data, result.fitted.data + result.residual.data, atol=1e-15
    )
    np.testing.assert_array_equal(spectra.data, before)
    # Older Core exposes coordinates rounded to its stored significant digits.
    np.testing.assert_allclose(
        result.measured.y.data, spectra.y.data, rtol=1e-12, atol=0
    )
    assert result.diagnostics["rmse"] < 0.006
    assert result.provenance["regularization"] == 1e-3


def test_bounds_units_and_zero_columns():
    spectra = mixture()
    spectra.data[:, :10] = 0
    exp = scp.nmr.Diffusion(spectra, b=spectra.y)
    limits = np.array([0.003, 0.3]) * scp.ur("um^2/s")
    result = exp.dosy(region=(4, 0), D=limits, points=40)
    np.testing.assert_allclose(
        result.map.y.data[[0, -1]], [3e-15, 3e-13], rtol=1e-12, atol=0
    )
    np.testing.assert_array_equal(result.map.data[:, :10], 0)
    assert result.diagnostics["boundary_fraction"].max() > 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"D": (0, 1e-10)},
        {"D": (1e-10, 1e-12)},
        {"D": (1e-12, np.inf)},
        {"points": 7},
        {"points": 20.5},
        {"points": True},
        {"regularization": 0},
        {"regularization": -1},
        {"regularization": np.nan},
        {"region": (5, 2)},
        {"region": (2, 2)},
    ],
)
def test_unsupported_grid_and_region_are_refused(kwargs):
    spectra = mixture()
    options = {"region": (3.4, 1.4), "D": (3e-12, 3e-10), **kwargs}
    with pytest.raises(ValueError):
        scp.nmr.Diffusion(spectra, b=spectra.y).dosy(**options)


def test_plot_uses_physical_log_diffusion_axis():
    import matplotlib.pyplot as plt

    spectra = mixture()
    result = scp.nmr.Diffusion(spectra, b=spectra.y).dosy(
        region=(3.4, 1.4), D=(3e-12, 3e-10)
    )
    axes = result.plot()
    assert axes.get_yscale() == "log"
    assert axes.get_ylim()[0] > 0
    plt.close("all")


def test_real_peg_map_and_processing_metadata():
    path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/102/ser"
    if not path.is_file():
        pytest.skip("Fetch PEG 600 with docs/gallery-fixtures.toml")
    experiment = scp.nmr.Diffusion(path)
    spectra = experiment.process()
    result = experiment.dosy(region=(73.0, 72.5), D=(1e-12, 5e-11))
    # Check the strongest map feature, not the sum of noise/baseline columns.
    peak_row = np.unravel_index(result.map.data.argmax(), result.map.shape)[0]
    mode = result.map.y.data[peak_row]
    assert mode == pytest.approx(7.6e-12, rel=0.2, abs=0)
    assert result.provenance["b_source"].endswith("diff.xml")
    assert result.measured.meta["diffusion"]["processing"]
    np.testing.assert_array_equal(experiment.process().data, spectra.data)
    assert result.diagnostics["rmse"] < 0.1 * np.max(abs(result.measured.data))
