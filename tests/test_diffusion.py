"""Public PFG workflow: scientific recovery, units and refusal boundaries."""

import numpy as np
import pytest

import spectrochempy as scp


def spectra_and_b(*, reverse=False):
    b = np.linspace(0, 3e11, 24)
    ppm = np.linspace(3, 1, 201)
    line = np.exp(-(((ppm - 2) / 0.08) ** 2))
    values = 5 * np.exp(-b[:, None] * 8e-12) * line[None, :]
    if reverse:
        b, values = b[::-1], values[::-1]
    spectra = scp.NDDataset(
        values,
        coordset=[scp.Coord.arange(24), scp.Coord(ppm, units="ppm")],
        units="count",
    )
    return spectra, scp.Coord(b, units="s/m^2", rounding=False)


@pytest.mark.parametrize("reverse", [False, True])
def test_known_diffusion_is_recovered_with_units_and_residual(reverse):
    spectra, b = spectra_and_b(reverse=reverse)
    original = spectra.copy()
    experiment = scp.nmr.Diffusion(spectra, b=b)
    result = experiment.fit(region=(1.7, 2.3))
    assert result.D.to("m^2/s").magnitude == pytest.approx(8e-12, rel=1e-4, abs=0)
    assert result.diagnostics["success"]
    assert result.I0.units == scp.ur.count * scp.ur.ppm
    np.testing.assert_allclose(result.areas.x.data, b.data)
    np.testing.assert_allclose(
        result.residual.data, result.areas.data - result.fitted.data
    )
    assert result.rmse.magnitude < 1e-5
    np.testing.assert_array_equal(spectra.data, original.data)
    np.testing.assert_array_equal(spectra.y.data, original.y.data)
    assert "Diffusion" in dir(scp.nmr)


def test_b_units_are_converted_and_ambiguous_numbers_refused():
    spectra, b = spectra_and_b()
    experiment = scp.nmr.Diffusion(spectra, b=b.to("s/cm^2"))
    assert experiment.fit(region=(1.7, 2.3)).D.magnitude == pytest.approx(
        8e-12, rel=1e-4, abs=0
    )
    for invalid in [
        None,
        b.data,
        scp.Coord(b.data, units="s"),
        scp.Coord([-1] * 24, units="s/m^2"),
        scp.Coord([1] * 24, units="s/m^2"),
    ]:
        with pytest.raises(ValueError):
            scp.nmr.Diffusion(spectra, b=invalid)


@pytest.mark.parametrize("kind", ["complex", "nan", "masked", "nonuniform", "time"])
def test_unprepared_or_invalid_spectra_are_refused(kind):
    spectra, b = spectra_and_b()
    if kind == "complex":
        spectra.data = spectra.data.astype(complex) + 1j
    elif kind == "nan":
        spectra.data[0, 0] = np.nan
    elif kind == "masked":
        mask = np.zeros(spectra.shape, dtype=bool)
        mask[0, 0] = True
        spectra.mask = mask
    elif kind == "nonuniform":
        x = spectra.x.data.copy()
        x[5] += 0.001
        spectra.x = scp.Coord(x, units="ppm")
    else:
        spectra.x = scp.Coord(spectra.x.data, units="s")
    with pytest.raises(ValueError):
        scp.nmr.Diffusion(spectra, b=b)


@pytest.mark.parametrize(
    "region", [(0, 2), (2, 2), (2, np.nan), (1, 2, 3), (2.00001, 2.00002)]
)
def test_invalid_regions_are_refused(region):
    spectra, b = spectra_and_b()
    with pytest.raises(ValueError):
        scp.nmr.Diffusion(spectra, b=b).fit(region=region)


def test_negative_and_empty_bands_are_refused():
    spectra, b = spectra_and_b()
    for values in [-spectra.data, np.zeros(spectra.shape)]:
        data = spectra.copy()
        data.data = values
        with pytest.raises(ValueError, match="positive area"):
            scp.nmr.Diffusion(data, b=b).fit(region=(1.7, 2.3))


def test_result_plot_uses_two_panels(tmp_path):
    spectra, b = spectra_and_b()
    result = scp.nmr.Diffusion(spectra, b=b).fit(region=(1.7, 2.3))
    result.plot(output=tmp_path / "fit.png")
    assert (tmp_path / "fit.png").is_file()


def test_exported_b_rejects_inconsistent_rows(tmp_path):
    from spectrochempy_nmr.diffusion import _exported_b

    table = tmp_path / "diff.xml"
    table.write_text(
        '<diff><pulprog>diffSte</pulprog><nXValues>4</nXValues><XValues><X i="1" B="1"/></XValues></diff>'
    )
    with pytest.raises(ValueError, match="row indices"):
        _exported_b(tmp_path)
    table.write_text("<diff><pulprog>other_sequence</pulprog></diff>")
    with pytest.raises(ValueError, match="diffSte only"):
        _exported_b(tmp_path)


def test_peg600_reference_and_reprocessing():
    path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/relax/102/ser"
    if not path.is_file():
        pytest.skip(
            "PEG 600 acquisition not available; fetch docs/gallery-fixtures.toml"
        )
    experiment = scp.nmr.Diffusion(path)
    first = experiment.process()
    second = experiment.process()
    np.testing.assert_array_equal(first.data, second.data)
    assert first.shape == (12, 65536)
    assert first.y.units == scp.ur("s/m^2").units
    result = experiment.fit(region=(72.649, 72.916))
    assert result.D.magnitude == pytest.approx(7.559092e-12, rel=0.01, abs=0)
    assert result.provenance["b_source"].endswith("diff.xml")
    assert result.provenance["processing"]["group_delay_points"] == pytest.approx(
        67.9839477539062
    )
    # Editing a returned view must not change the stored spectra or a later fit.
    first.data[:] = 0
    assert experiment.fit(region=(72.916, 72.649)).D.magnitude == pytest.approx(
        result.D.magnitude, rel=1e-10, abs=0
    )
    for options in [dict(size=1024), dict(size=True), dict(lb=-1), dict(phc0=np.nan)]:
        with pytest.raises(ValueError):
            experiment.process(**options)
