import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment


def _map(reverse=False):
    x = np.arange(-12.0, 13)
    y = np.arange(-64.0, 64)
    ratio = 19 / 12
    data = np.exp(-0.5 * ((y[:, None] - ratio * x) / 3) ** 2)
    if reverse:
        data, x, y = data[::-1, ::-1], x[::-1], y[::-1]
    return scp.NDDataset(
        data,
        coordset=[
            scp.Coord(y, units="Hz", rounding=False),
            scp.Coord(x, units="Hz", rounding=False, sigdigits=15),
        ],
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_mqmas_shears_an_analytical_ridge_without_mutating_source(reverse):
    source = _map(reverse)
    before = source.copy()
    result = Experiment(source).shear_mqmas(spin=2.5)
    expected = np.exp(-0.5 * (source.y.data / 3) ** 2)[:, None]
    np.testing.assert_allclose(
        result.data, np.broadcast_to(expected, source.shape), atol=1e-10
    )
    np.testing.assert_allclose(result.y.data, source.y.data / (17 / 12))
    np.testing.assert_array_equal(source.data, before.data)
    np.testing.assert_array_equal(source.y.data, before.y.data)
    assert not source.meta.get("mqmas_shearing")
    assert result.meta.mqmas_shearing["ratio"] == pytest.approx(19 / 12)
    with pytest.raises(ValueError, match="already"):
        Experiment(result).shear_mqmas(spin=2.5)


def test_mqmas_preserves_dimension_meaning_after_transpose():
    source = _map()
    expected = Experiment(source).shear_mqmas(spin=2.5)
    actual = Experiment(source.T).shear_mqmas(spin=2.5)
    np.testing.assert_allclose(actual.T.data, expected.data, atol=1e-12)


@pytest.mark.parametrize(
    ("spin", "order", "ratio"),
    [(1.5, 3, -7 / 9), (2.5, 5, -25 / 12), (3.5, 3, 101 / 45), (4.5, 9, -31 / 6)],
)
def test_mqmas_signed_ratios(spin, order, ratio):
    result = Experiment(_map()).shear_mqmas(spin=spin, coherence_order=order)
    assert result.meta.mqmas_shearing["ratio"] == pytest.approx(ratio)


@pytest.mark.parametrize(
    "spin,order",
    [
        (1, 3),
        (0.5, 3),
        (2.1, 3),
        (True, 3),
        (np.nan, 3),
        (2.5, 2),
        (2.5, 7),
        (2.5, 3.0),
    ],
)
def test_mqmas_rejects_invalid_transition(spin, order):
    with pytest.raises(ValueError):
        Experiment(_map()).shear_mqmas(spin=spin, coherence_order=order)


def test_mqmas_rejects_time_domain_masked_and_irregular_maps():
    source = _map()
    source.coordset["y"] = scp.Coord(source.y.data, units="s")
    with pytest.raises(ValueError, match="axes"):
        Experiment(source).shear_mqmas(spin=2.5)
    source = _map()
    source[0, 0] = scp.MASKED
    with pytest.raises(ValueError, match="unmasked"):
        Experiment(source).shear_mqmas(spin=2.5)


def test_mqmas_preserves_each_complex_component():
    base = _map()
    data = np.stack([base.data, 2 * base.data, 3 * base.data, 4 * base.data])
    source = scp.NDDataset(data, dims=["c", "y", "x"])
    source.set_coordset(
        c=scp.Coord(labels=["rr", "ri", "ir", "ii"]), y=base.y, x=base.x
    )
    source.meta.component_axis = "c"
    result = Experiment(source).shear_mqmas(spin=2.5)
    for i in range(1, 4):
        np.testing.assert_allclose(result.data[i], (i + 1) * result.data[0], atol=1e-12)


def test_mqmas_ppm_and_hz_are_equivalent():
    hz = _map()
    for dim in ("x", "y"):
        hz.coordset[dim].meta["acquisition_frequency"] = 100 * scp.ur.MHz
    ppm = hz.copy()
    ppm.x.ito("ppm")
    ppm.y.ito("ppm")
    expected = Experiment(hz).shear_mqmas(spin=2.5)
    actual = Experiment(ppm).shear_mqmas(spin=2.5)
    np.testing.assert_allclose(actual.data, expected.data, atol=1e-10)
    np.testing.assert_allclose(actual.y.to("Hz").data, expected.y.data, atol=1e-10)


@pytest.mark.parametrize("pivot", [True, np.nan, np.inf, "oops", 1 * scp.ur.s])
def test_mqmas_rejects_invalid_pivot(pivot):
    with pytest.raises((ValueError, TypeError)):
        Experiment(_map()).shear_mqmas(spin=2.5, pivot=pivot)


def test_mqmas_rejects_irregular_and_nonfinite_data():
    source = _map()
    values = source.y.data.copy()
    values[5] += 0.3
    source.coordset["y"] = scp.Coord(values, units="Hz")
    with pytest.raises(ValueError, match="uniform"):
        Experiment(source).shear_mqmas(spin=2.5)
    source = _map()
    source.data[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Experiment(source).shear_mqmas(spin=2.5)


def test_mqmas_handles_public_reader_complex_pairs():
    base = _map()
    source = scp.NDDataset(
        np.stack([base.data * (1 + 2j), base.data * (3 + 4j)]), dims=["c", "y", "x"]
    )
    source.set_coordset(c=scp.Coord(labels=["S_R", "S_I"]), y=base.y, x=base.x)
    source.meta.component_axis = "c"
    result = Experiment(source).shear_mqmas(spin=2.5)
    expected = Experiment(base).shear_mqmas(spin=2.5)
    np.testing.assert_allclose(result.data[0], expected.data * (1 + 2j), atol=1e-12)
    np.testing.assert_allclose(result.data[1], expected.data * (3 + 4j), atol=1e-12)


def test_mqmas_integer_shift_matches_periodic_sample_permutation():
    source = scp.NDDataset(
        np.arange(32.0).reshape(8, 4),
        coordset=[
            scp.Coord(np.arange(8.0), units="Hz"),
            scp.Coord(
                np.arange(4.0) * 12 / 19, units="Hz", rounding=False, sigdigits=15
            ),
        ],
    )
    result = Experiment(source).shear_mqmas(spin=2.5)
    expected = np.column_stack([np.roll(source.data[:, j], -j) for j in range(4)])
    np.testing.assert_allclose(result.data, expected, atol=1e-12)


def test_mqmas_pivot_changes_shift_and_uses_single_quantum_reference():
    source = _map()
    source.x.meta.acquisition_frequency = 100 * scp.ur.MHz
    source.y.meta.acquisition_frequency = 300 * scp.ur.MHz
    before = source.copy()
    # One F2 step corresponds to an exact one-bin F1 translation here.
    pivot = 12 / 19 * scp.ur.Hz
    zero = Experiment(source).shear_mqmas(spin=2.5)
    moved = Experiment(source).shear_mqmas(spin=2.5, pivot=pivot)
    np.testing.assert_allclose(moved.data, np.roll(zero.data, 1, axis=0), atol=1e-12)
    assert moved.y.meta.acquisition_frequency == 100 * scp.ur.MHz
    assert source.y.meta.acquisition_frequency == before.y.meta.acquisition_frequency
    assert source.history == before.history


def test_mqmas_rejects_unreferenced_ppm():
    source = _map()
    source.coordset["y"] = scp.Coord(source.y.data, units="ppm")
    with pytest.raises(ValueError, match="reference"):
        Experiment(source).shear_mqmas(spin=2.5)


def test_mqmas_fractional_shift_does_not_mix_packed_nyquist_components():
    x = scp.Coord(np.arange(2.0) * 12 / 19, units="Hz", rounding=False, sigdigits=15)
    y = scp.Coord(np.arange(8.0), units="Hz")
    nyquist = np.broadcast_to((-1.0) ** np.arange(8)[:, None], (8, 2))
    source = scp.NDDataset(
        np.stack((nyquist.astype(complex), np.zeros_like(nyquist))),
        dims=["c", "y", "x"],
    )
    source.set_coordset(c=scp.Coord(labels=["S_R", "S_I"]), y=y, x=x)
    source.meta.component_axis = "c"
    result = Experiment(source).shear_mqmas(spin=2.5, pivot=-6 / 19)
    np.testing.assert_allclose(result.data, 0, atol=1e-12)


def test_mqmas_rejects_dataset_without_coordinates():
    source = scp.NDDataset(np.zeros((8, 2)))
    with pytest.raises(ValueError, match="calibrated spectral grids"):
        Experiment(source).shear_mqmas(spin=2.5)
