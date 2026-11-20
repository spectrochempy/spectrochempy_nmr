"""Physical shear coefficients use one quadrature/reconstruction workflow."""

import numpy as np
import pytest
import spectrochempy as scp

from test_mqmas_common import states_path as mqmas_path
from test_stmas import states_path as _states_path


@pytest.fixture
def states_path(tmp_path):
    return _states_path.__wrapped__(tmp_path)


def _phase(raw, *, points=16):
    mixed = scp.nmr.Experiment(raw).xf2(
        size=32, direct_points=points, correct_digital_filter=False
    )
    return scp.nmr.Experiment(mixed).xf2p(phc0=-31, phc1=24, pivot=2)


@pytest.mark.parametrize("shift", [0, -0.1, 0.1])
def test_stmas_shortcut_uses_full_xfshear(states_path, shift):
    path, _, _ = states_path
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    phased = _phase(raw)
    before = phased.copy()
    expected = scp.nmr.Experiment(phased).xfshear(
        size=32, ratio=-8 / 9, f1_scale=17 / 9, f1_shift=-shift
    )
    actual = scp.nmr.Experiment(phased).stmas_shear(size=32, f1_shift=shift)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.x.data, expected.x.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)
    assert actual.y.meta == expected.y.meta
    for component in ("RR", "RI", "IR", "II"):
        got = getattr(scp.nmr.Experiment(actual), component)
        ref = getattr(scp.nmr.Experiment(expected), component)
        np.testing.assert_array_equal(got.data, ref.data)
    np.testing.assert_array_equal(phased.data, before.data)
    assert phased.meta == before.meta


@pytest.mark.parametrize("shift", [0, -1.5, 1.5])
def test_explicit_mqmas_coefficients_preserve_all_components(tmp_path, shift):
    raw = scp.nmr.read(mqmas_path.__wrapped__(tmp_path))
    phased = _phase(raw)
    auto = scp.nmr.Experiment(phased).xfshear(size=16, f1_shift=shift)
    manual = scp.nmr.Experiment(phased).xfshear(
        size=16, ratio=19 / 12, f1_scale=17 / 12, f1_shift=shift
    )
    np.testing.assert_array_equal(manual.data, auto.data)
    np.testing.assert_array_equal(manual.y.data, auto.y.data)
    assert manual.y.meta == auto.y.meta


@pytest.mark.parametrize("points", [2, 16])
@pytest.mark.parametrize("shift", [0, -0.1, 0.1])
def test_states_and_states_tppi_share_shear_after_decoding(tmp_path, points, shift):
    path = mqmas_path.__wrapped__(tmp_path)
    reference = scp.nmr.read(path)
    # Encode exactly the same States samples with an alternating F1 sign.
    words = np.fromfile(path / "ser", dtype=">i4").reshape(8, 256)
    words[2:4] *= -1
    words[6:8] *= -1
    words.tofile(path / "ser")
    file = path / "acqu2s"
    file.write_text(file.read_text().replace("FnMODE= 4", "FnMODE= 5"))
    encoded = scp.nmr.read(path)
    assert encoded.meta.quadrature_mode == "states-tppi"
    before = encoded.copy()
    expected_phase = _phase(reference, points=points)
    actual_phase = _phase(encoded, points=points)
    np.testing.assert_array_equal(actual_phase.data, expected_phase.data)
    assert actual_phase.meta.tppi_alternation_state == "applied"
    for ratio, scale in ((19 / 12, 17 / 12), (-8 / 9, 17 / 9)):
        options = dict(size=16, ratio=ratio, f1_scale=scale, f1_shift=shift)
        expected = scp.nmr.Experiment(expected_phase).xfshear(**options)
        actual = scp.nmr.Experiment(actual_phase).xfshear(**options)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.y.data, expected.y.data)
    np.testing.assert_array_equal(encoded.data, before.data)
    assert encoded.meta == before.meta


@pytest.mark.parametrize(
    "options",
    [
        {"ratio": 1},
        {"f1_scale": 1},
        {"ratio": True, "f1_scale": 1},
        {"ratio": np.nan, "f1_scale": 1},
        {"ratio": np.inf, "f1_scale": 1},
        {"ratio": 0, "f1_scale": 0},
        {"ratio": 0, "f1_scale": -1},
        {"ratio": 0, "f1_scale": np.nan},
        {"ratio": 0, "f1_scale": True},
        {"ratio": 0, "f1_scale": 1, "spin": 1.5},
        {"ratio": 0, "f1_scale": 1, "coherence_order": 5},
    ],
)
def test_invalid_physical_coefficients_do_not_mutate(states_path, options):
    path, _, _ = states_path
    phased = _phase(scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]"))
    before = phased.copy()
    with pytest.raises(ValueError, match="coefficients|ratio"):
        scp.nmr.Experiment(phased).xfshear(size=32, **options)
    np.testing.assert_array_equal(phased.data, before.data)
    assert phased.meta == before.meta


@pytest.mark.parametrize("shift", [0, -0.1, 0.1])
def test_four_stmas_planes_against_analytic_line(states_path, shift):
    """A separable line has a geometric F1 sum and a known Hilbert kernel."""
    path, _, (nx, ny, sw2, dwell, carrier, sf, kx, ky) = states_path
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    initial_phase = 0.37
    cosine, sine = raw.data
    raw.data = np.stack(
        (
            np.cos(initial_phase) * cosine - np.sin(initial_phase) * sine,
            np.sin(initial_phase) * cosine + np.cos(initial_phase) * sine,
        )
    )
    mixed = scp.nmr.Experiment(raw).xf2(size=nx, correct_digital_filter=False)
    phased = scp.nmr.Experiment(mixed).xf2p()
    result = scp.nmr.Experiment(phased).xfshear(
        size=ny, ratio=-8 / 9, f1_scale=17 / 9, f1_shift=shift
    )
    # The shear cancels the acquired F2-dependent evolution at the line bin.
    # Half weighting of t1=0 subtracts 1/2 from the finite geometric sum.
    bins = np.arange(ny) - ny // 2
    frequency = (bins - ky) / ny + shift * carrier * (17 / 9) * dwell
    geometric = ny * np.sinc(ny * frequency) / np.sinc(frequency)
    geometric = geometric * np.exp(1j * np.pi * (ny - 1) * frequency)
    f1 = nx * np.exp(1j * initial_phase) * (geometric - 0.5)
    line = np.zeros(nx)
    ix = nx // 2 + kx
    line[ix] = 1
    # Imaginary part of the analytic delta, zero-extended to 2*nx, in F2.
    distance = np.arange(nx) - ix
    harmonics = np.arange(1, nx)
    hilbert_kernel = np.sin(np.pi * distance[:, None] * harmonics / nx).sum(axis=1) / nx
    expected = (
        f1.real[:, None] * line,
        -f1.imag[:, None] * line,
        -f1.real[:, None] * hilbert_kernel,
        f1.imag[:, None] * hilbert_kernel,
    )
    view = scp.nmr.Experiment(result)
    for component, plane in zip(("RR", "RI", "IR", "II"), expected, strict=True):
        np.testing.assert_allclose(getattr(view, component).data, plane, atol=1e-10)
    reference = sf + shift * carrier * 1e-6
    grid = (carrier - reference) * 1e6 / reference - bins / (
        ny * dwell * (17 / 9) * reference
    )
    np.testing.assert_allclose(result.y.data, grid, atol=1e-10)
    assert result.y.meta["reference_frequency"].to("MHz").magnitude == pytest.approx(
        reference
    )


def test_states_tppi_scp_resume_does_not_decode_twice(tmp_path, monkeypatch):
    from nmrglue import interop

    path = mqmas_path.__wrapped__(tmp_path)
    file = path / "acqu2s"
    file.write_text(file.read_text().replace("FnMODE= 4", "FnMODE= 5"))
    raw = scp.nmr.read(path)
    expected = scp.nmr.Experiment(_phase(raw)).xfshear(size=16)
    raw.write(tmp_path / "encoded.scp", confirm=False)

    def unavailable(*args, **kwargs):
        pytest.fail("Processing must use the saved physical contract")

    monkeypatch.setattr(interop, "read_bruker_ser_storage", unavailable)
    restored = scp.nmr.read(tmp_path / "encoded.scp")
    mixed = scp.nmr.Experiment(restored).xf2(
        size=32, direct_points=16, correct_digital_filter=False
    )
    mixed.write(tmp_path / "mixed.scp", confirm=False)
    resumed = scp.nmr.read(tmp_path / "mixed.scp")
    phased = scp.nmr.Experiment(resumed).xf2p(phc0=-31, phc1=24, pivot=2)
    actual = scp.nmr.Experiment(phased).xfshear(size=16)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)
    assert raw.meta.tppi_alternation_state == "encoded"
    assert resumed.meta.tppi_alternation_state == "applied"


def test_explicit_physical_coefficients_do_not_require_isotope_lookup(states_path):
    path, _, _ = states_path
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    raw.meta.readonly = False
    raw.meta.acquisition["nuclei"] = ("unknown", "unknown")
    phased = _phase(raw)
    with pytest.raises(ValueError, match="known observed isotope"):
        scp.nmr.Experiment(phased).xfshear(size=32)
    result = scp.nmr.Experiment(phased).xfshear(size=32, ratio=0, f1_scale=1)
    assert result.shape == (2, 32, 32)
    assert result.meta.mqmas_shearing["spin"] is None
