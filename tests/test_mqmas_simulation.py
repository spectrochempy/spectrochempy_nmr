"""Direct MQMAS positions, RF transfer and cached-map contracts."""

import numpy as np
import pytest

from spectrochempy.nmr import pulsar
from spectrochempy.core.units import ur
from spectrochempy_nmr.processing.mqmas import _mqmas_geometry


PREPARATION = (pulsar.Pulse(12e-6, 110e3, 17),)
CONVERSION = (pulsar.Pulse(4e-6, 85e3, -23), pulsar.Coherence(1))
SETTINGS = dict(
    spin=2.5,
    larmor_hz=104.46851646e6,
    cq_hz=2.923e6,
    eta=0.136,
    shift_hz=750,
    spinning_hz=12500,
    spectral_width_hz=24000,
    points=512,
    accuracy=1,
    sidebands=0,
    max_rf_step_s=1e-6,
)


@pytest.mark.parametrize("spin,order", [(1.5, 3), (2.5, 3), (2.5, 5), (3.5, 3)])
def test_analytical_mqmas_position_matches_each_sheared_crystallite(spin, order):
    settings = dict(SETTINGS, spin=spin, coherence_order=order)
    model = pulsar.MQMASModel(PREPARATION, CONVERSION, **settings)
    directions = pulsar._powder_mesh(2)[0]
    angle = np.arccos(1 / np.sqrt(3))
    terms = list(
        pulsar._transition_harmonics(
            directions,
            settings["cq_hz"],
            settings["eta"],
            angle,
            spin,
            settings["larmor_hz"],
            0,
        )
    )
    upper, lower = int(spin - order / 2), int(spin + order / 2)
    multiple = sum(term[2][0].real for term in terms[upper:lower])
    central = terms[int(spin - 0.5)][2][0].real
    _, ratio, scale = _mqmas_geometry(spin, order)
    isotropic = settings["shift_hz"] + (multiple - ratio * central) / scale
    np.testing.assert_allclose(isotropic, model.f1_position_hz, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("sign", [1, -1])
def test_symmetric_mq_selection_preserves_only_the_requested_pair(sign):
    full = pulsar.simulate_pulse_sequence(
        (*PREPARATION, pulsar.Coherence(sign * 3)), **SETTINGS, output="density"
    )
    symmetric = pulsar.simulate_pulse_sequence(
        (*PREPARATION, pulsar.Coherence(sign * 3, symmetric=True)),
        **SETTINGS,
        output="density",
    )
    expected = np.zeros_like(full)
    row, col = (1, 4) if sign == 1 else (4, 1)
    expected[row, col] = full[row, col]
    np.testing.assert_allclose(symmetric, expected, atol=1e-14)
    assert abs(expected[row, col]) > 1e-5


def test_pq_and_cq_give_identical_rf_profiles_and_positions():
    cq, eta = SETTINGS["cq_hz"], SETTINGS["eta"]
    first = pulsar.MQMASModel(PREPARATION, CONVERSION, **SETTINGS)
    second = pulsar.MQMASModel(
        PREPARATION,
        CONVERSION,
        **dict(SETTINGS, cq_hz=0, pq_hz=cq * np.sqrt(1 + eta**2 / 3)),
    )
    kwargs = dict(f1_spectral_width_hz=12000, f1_points=128, f1_gaussian_fwhm_hz=500)
    assert first.f1_position_hz == pytest.approx(second.f1_position_hz, rel=1e-14)
    np.testing.assert_allclose(
        first.spectrum(**kwargs).data, second.spectrum(**kwargs).data, atol=1e-13
    )


def test_f1_map_preserves_the_complex_rf_weighted_f2_projection():
    model = pulsar.MQMASModel(PREPARATION, CONVERSION, **SETTINGS)
    result = model.spectrum(
        f1_spectral_width_hz=12000, f1_points=128, f1_gaussian_fwhm_hz=500
    )
    paths = tuple(
        (*PREPARATION, pulsar.Coherence(sign * 3, symmetric=True), *CONVERSION)
        for sign in (1, -1)
    )
    reference = pulsar.simulate_pulse_sequence(
        pulsar.PhaseCycle(paths),
        **SETTINGS,
        transitions="central",
        rotor_angle_deg=np.rad2deg(np.arccos(1 / np.sqrt(3))),
    )
    np.testing.assert_allclose(
        result.data.sum(axis=0), reference.data, rtol=1e-12, atol=1e-14
    )
    assert abs(reference.data.sum()) > 1e-5
    assert result.shape == (128, 512)
    assert result.y.units == result.x.units == ur.Hz
    np.testing.assert_array_equal(result.x.data, reference.x.data)


def test_cached_model_renders_without_repeating_rf_propagation(monkeypatch):
    model = pulsar.MQMASModel(PREPARATION, CONVERSION, **SETTINGS)

    def fail(*args, **kwargs):
        pytest.fail("Rendering an existing model must reuse its RF profiles")

    monkeypatch.setattr(pulsar, "simulate_pulse_sequence", fail)
    monkeypatch.setattr(pulsar, "_mqmas_rf_profiles", fail)
    monkeypatch.setattr(pulsar, "_conversion_readout", fail)
    first = model.spectrum(f1_spectral_width_hz=12000, f1_points=128)
    second = model.spectrum(
        f1_spectral_width_hz=16000,
        f1_points=192,
        f1_gaussian_fwhm_hz=300,
        f2_gaussian_fwhm_hz=100,
    )
    assert first.shape == (128, 512)
    assert second.shape == (192, 512)
    np.testing.assert_allclose(
        first.data.sum(), second.data.sum(), rtol=1e-11, atol=1e-13
    )


def test_rf_excitation_and_conversion_losses_are_retained():
    normal = pulsar.simulate_mqmas(
        PREPARATION, CONVERSION, **SETTINGS, f1_spectral_width_hz=12000
    )
    no_excitation = pulsar.simulate_mqmas(
        (pulsar.Pulse(12e-6, 0),), CONVERSION, **SETTINGS, f1_spectral_width_hz=12000
    )
    no_conversion = pulsar.simulate_mqmas(
        PREPARATION, (pulsar.Delay(4e-6),), **SETTINGS, f1_spectral_width_hz=12000
    )
    assert np.linalg.norm(normal.data) > 1e-5
    assert np.linalg.norm(no_excitation.data) < 1e-13
    assert np.linalg.norm(no_conversion.data) < 1e-13


@pytest.mark.parametrize(
    "settings",
    [
        {"spin": 2},
        {"coherence_order": 2},
        {"coherence_order": 7},
        {"spinning_hz": 0},
        {"larmor_hz": 0},
        {"eta": 1.1},
        {"pq_hz": -1, "cq_hz": 0},
        {"pq_hz": 1e6},
        {"cq_hz": np.nan},
    ],
)
def test_invalid_mqmas_systems_are_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        pulsar.MQMASModel(PREPARATION, CONVERSION, **dict(SETTINGS, **settings))


@pytest.mark.parametrize(
    "options",
    [
        {"f1_points": 1},
        {"f1_gaussian_fwhm_hz": 0},
        {"f1_spectral_width_hz": -1},
        {"f1_offset_hz": 1e9},
        {"f2_gaussian_fwhm_hz": -1},
    ],
)
def test_invalid_mqmas_output_grids_are_rejected(options):
    model = pulsar.MQMASModel(PREPARATION, CONVERSION, **SETTINGS)
    with pytest.raises((ValueError, TypeError)):
        model.spectrum(**{"f1_spectral_width_hz": 12000, **options})


def test_phase_cycle_generators_are_materialized_before_reusing_the_paths():
    ordinary = pulsar.MQMASModel(
        pulsar.PhaseCycle((PREPARATION, PREPARATION), (0.2, 0.7)),
        CONVERSION,
        **SETTINGS,
    )
    generated = pulsar.MQMASModel(
        pulsar.PhaseCycle((iter(PREPARATION), iter(PREPARATION)), (0.2, 0.7)),
        iter(CONVERSION),
        **SETTINGS,
    )
    np.testing.assert_allclose(
        ordinary.spectrum(f1_spectral_width_hz=12000).data,
        generated.spectrum(f1_spectral_width_hz=12000).data,
        rtol=1e-13,
        atol=1e-14,
    )


def test_saved_rf_library_distinguishes_symmetric_coherence_selection(tmp_path):
    from spectrochempy_nmr import CzjzekRFLibrary

    settings = {
        key: value
        for key, value in SETTINGS.items()
        if key not in ("cq_hz", "eta", "shift_hz")
    }
    settings.update(accuracy=1, points=256, transitions="central")
    grid = dict(
        cq_grid_hz=[2.8e6, 3e6], eta_grid=[0.1, 0.2], shift_hz=750, shift_bins=1
    )
    events = (*PREPARATION, pulsar.Coherence(1, symmetric=True))
    library = CzjzekRFLibrary.build(events, **grid, **settings)
    filename = tmp_path / "symmetric.npz"
    library.save(filename)
    CzjzekRFLibrary.load(filename, events, **grid, **settings)
    with pytest.raises(ValueError, match="does not match"):
        CzjzekRFLibrary.load(
            filename, (*PREPARATION, pulsar.Coherence(1)), **grid, **settings
        )


@pytest.mark.parametrize("sidebands", [0, 2])
def test_shared_rf_quadratures_match_explicit_complex_phase_cycles(sidebands):
    preparation = pulsar.PhaseCycle(
        ((pulsar.Pulse(7e-6, 90e3, 27, 250),), (pulsar.Pulse(8e-6, 75e3, -13, -300),)),
        (0.3 + 0.1j, -0.25j),
    )
    conversion = (
        pulsar.Pulse(3e-6, 80e3, 35, 400),
        pulsar.Coherence(0),
        pulsar.Delay(2e-6),
        pulsar.IdealPulse(65, -20),
    )
    settings = dict(
        SETTINGS,
        t2_s=65e-6,
        receiver_phase_deg=17,
        workers=2,
        sidebands=sidebands,
        spectral_width_hz=80000,
    )
    model = pulsar.MQMASModel(preparation, conversion, **settings)
    paths = tuple(
        (*scan, pulsar.Coherence(sign * 3, symmetric=True), *conversion)
        for scan in preparation.scans
        for sign in (1, -1)
    )
    for profile, factors in zip(
        model._profiles, ((0.5, 0.5), (-0.5j, 0.5j)), strict=True
    ):
        coefficients = tuple(
            weight * factor for weight in preparation.coefficients for factor in factors
        )
        reference = pulsar.simulate_pulse_sequence(
            pulsar.PhaseCycle(paths, coefficients),
            **settings,
            transitions="central",
            rotor_angle_deg=np.rad2deg(np.arccos(1 / np.sqrt(3))),
        )
        np.testing.assert_allclose(profile.data, reference.data, rtol=1e-11, atol=1e-13)


def test_rotor_synchronized_rubidium_map_matches_explicit_states_series():
    settings = dict(
        spin=1.5,
        larmor_hz=130.9e6,
        cq_hz=1.8e6,
        eta=0.42,
        shift_hz=300,
        spinning_hz=20000,
        spectral_width_hz=24000,
        points=2048,
        accuracy=4,
        sidebands=0,
        max_rf_step_s=1e-6,
    )
    preparation = (pulsar.Pulse(5e-6, 60e3),)
    conversion = (pulsar.Pulse(2.5e-6, 70e3), pulsar.Coherence(1))
    _, ratio, scale = _mqmas_geometry(1.5, 3)
    time = np.arange(48) / 20000
    paths = tuple(
        (*preparation, pulsar.Coherence(sign * 3, symmetric=True)) for sign in (1, -1)
    )
    branches = [
        pulsar.simulate_t1_series(
            pulsar.PhaseCycle(paths, weights),
            time,
            conversion,
            **settings,
            transitions="central",
            rotor_angle_deg=np.rad2deg(np.arccos(1 / np.sqrt(3))),
        )
        for weights in ((0.5, 0.5), (-0.5j, 0.5j))
    ]
    cosine, sine = [branch.data for branch in branches]
    window = np.exp(-2 * np.pi**2 * (600 / np.sqrt(8 * np.log(2)) * scale * time) ** 2)
    window[0] *= 0.5
    shear = np.exp(-2j * np.pi * ratio * time[:, None] * branches[0].x.data)
    # Independently transform F1 for the real and imaginary F2 channels.
    channels = (cosine.real + 1j * sine.real, cosine.imag + 1j * sine.imag)
    spectra = [
        np.fft.fftshift(
            np.fft.fft(channel * shear * window[:, None], n=512, axis=0), axes=0
        )
        for channel in channels
    ]
    reference = 2 / 512 * (spectra[0].real + 1j * spectra[1].real)
    direct = pulsar.simulate_mqmas(
        preparation,
        conversion,
        **settings,
        f1_spectral_width_hz=20000 / scale,
        f1_points=512,
        f1_gaussian_fwhm_hz=600,
    )
    assert np.linalg.norm(direct.data - reference) / np.linalg.norm(reference) < 0.01
    np.testing.assert_allclose(
        direct.data.sum(), reference.sum(), rtol=1e-11, atol=1e-13
    )
