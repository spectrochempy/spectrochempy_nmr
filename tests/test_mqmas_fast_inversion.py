"""Physical RF kernels connected to population inversion through the public API."""

import numpy as np
import pytest
import spectrochempy as scp


P = scp.nmr.pulsar
INV = scp.nmr.mqmas_inversion
SETTINGS = dict(
    spin=1.5,
    larmor_hz=130.9e6,
    spinning_hz=20000,
    spectral_width_hz=24000,
    points=256,
    accuracy=1,
    sidebands=0,
    max_rf_step_s=1e-6,
)
PREP = (P.Pulse(5e-6, 60e3, 15),)
CONVERT = (P.Pulse(2.5e-6, 70e3, -20), P.Coherence(1))
F1 = np.linspace(-20, 20, 64)
F2 = np.linspace(-18, 18, 81)


def response(**settings):
    return INV.mqmas_response(
        F1,
        F2,
        PREP,
        CONVERT,
        f1_gaussian_fwhm_hz=600,
        f2_gaussian_fwhm_hz=100,
        **dict(SETTINGS, **settings),
    )


def spectrum(data, *axes):
    result = scp.NDDataset(data)
    if len(axes) == 2:
        result.set_coordset(
            y=scp.Coord(axes[0], units="ppm"), x=scp.Coord(axes[1], units="ppm")
        )
    else:
        result.set_coordset(x=scp.Coord(axes[0], units="ppm"))
    return result


@pytest.mark.parametrize("reverse", [False, True])
def test_response_preserves_model_signed_density_and_reference(reverse):
    f1, f2 = (F1[::-1], F2[::-1]) if reverse else (F1, F2)
    carrier, offset, shift = 11, 1.25, 12.5
    f1 = f1 + carrier + offset
    f2 = f2 + carrier
    r = INV.mqmas_response(
        f1,
        f2,
        PREP,
        CONVERT,
        carrier_ppm=carrier,
        f1_offset_ppm=offset,
        receiver_phase_deg=31,
        f1_gaussian_fwhm_hz=600,
        f2_gaussian_fwhm_hz=100,
        **SETTINGS,
    )
    actual = r(shift, 1.8e6, 0.42)
    native = P.MQMASModel(
        PREP,
        CONVERT,
        shift_hz=(shift - carrier) * 130.9,
        cq_hz=1.8e6,
        eta=0.42,
        receiver_phase_deg=31,
        **SETTINGS,
    ).spectrum(
        f1_points=F1.size,
        f1_spectral_width_hz=np.diff(F1).mean() * F1.size * 130.9,
        f1_offset_hz=F1[F1.size // 2] * 130.9,
        f1_gaussian_fwhm_hz=600,
        f2_gaussian_fwhm_hz=100,
    )
    dx = np.diff(native.x.data).mean()
    dy = np.diff(native.y.data).mean()
    expected = np.array(
        [
            np.interp(
                F2 * 130.9,
                native.x.data + dx / 2,
                row.real / (dx * dy),
                left=0,
                right=0,
            )
            for row in native.data
        ]
    )
    np.testing.assert_allclose(
        actual, expected[::-1, ::-1] if reverse else expected, rtol=1e-11, atol=1e-18
    )


def test_grid_and_width_views_reuse_rf_and_cache_can_be_released(monkeypatch):
    r = response()
    first = r(0, 1.8e6, 0.42)

    def fail(*args, **kwargs):
        raise AssertionError("RF propagation repeated")

    monkeypatch.setattr(P, "MQMASModel", fail)
    np.testing.assert_array_equal(r(0, 1.8e6, 0.42), first)
    changed = r.with_broadening(f1_gaussian_fwhm_hz=800, f2_gaussian_fwhm_hz=200)
    assert not np.allclose(changed(0, 1.8e6, 0.42), first, atol=1e-12)
    assert r.cache_info().misses == 1
    assert changed.cache_info().currsize == 1
    r.cache_clear()
    assert changed.cache_info().currsize == 0


def test_phase_cycle_generators_can_be_reused_for_distinct_grid_cells():
    scans = ((pulse,) for pulse in (P.Pulse(5e-6, 60e3), P.Pulse(6e-6, 55e3)))
    r = INV.mqmas_response(
        F1,
        F2,
        P.PhaseCycle(scans, (c for c in (0.4, 0.6j))),
        iter(CONVERT),
        f1_gaussian_fwhm_hz=600,
        **SETTINGS,
    )
    assert np.linalg.norm(r(0, 1.6e6, 0.42)) > 0
    assert np.linalg.norm(r(1, 1.8e6, 0.42)) > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(f1_ppm=[0, 1, 3]),
        dict(f2_ppm=[0, 0]),
        dict(larmor_hz=0),
        dict(f1_gaussian_fwhm_hz=0),
        dict(f2_gaussian_fwhm_hz=-1),
        dict(cq_hz=1e6),
    ],
)
def test_invalid_response_configuration(kwargs):
    settings = dict(SETTINGS, f1_ppm=F1, f2_ppm=F2, f1_gaussian_fwhm_hz=600)
    settings.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        INV.mqmas_response(preparation=PREP, conversion=CONVERT, **settings)


def test_rf_library_recovers_two_populations_in_mqmas_and_joint_mas():
    shifts, couplings = np.array([-3.0, 0, 3]), np.array([1.2, 1.5, 1.8])
    r = response()
    blank = spectrum(np.zeros((F1.size, F2.size)), F1, F2)
    kernel = INV.build_kernel(blank, shifts, couplings, r, asymmetry=0.42)
    assert r.cache_info().currsize == 9
    truth = np.zeros(9)
    truth[0], truth[-1] = 0.65, 0.35
    observation = spectrum((kernel.matrix @ truth).reshape(blank.shape), F1, F2)
    r.cache_clear()
    mas_r = INV.mas_response(
        F2,
        (P.IdealPulse(90),),
        gaussian_fwhm_hz=100,
        transitions="central",
        **SETTINGS,
    )
    mas_kernel = INV.build_kernel(
        spectrum(np.zeros(F2.size), F2), shifts, couplings, mas_r, asymmetry=0.42
    )
    mas = spectrum(1.7 * (mas_kernel.matrix @ truth) + 0.02, F2)
    for options in ({}, dict(mas=mas, mas_kernel=mas_kernel)):
        fitted = INV.fit(observation, kernel, regularization=0, **options)
        recovered = fitted.population.data.ravel()
        np.testing.assert_allclose(
            recovered / recovered.sum(), truth, rtol=1e-3, atol=5e-4
        )
        assert fitted.rms < 5e-4
    assert r.cache_info().currsize == 0
