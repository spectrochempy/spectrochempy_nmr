"""Geometry selection and acquisition quadrature are independent."""

from copy import deepcopy

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from test_quadrature_contract import _acquisition
from test_shared_xfshear import _phase
from test_stmas import states_path as _states_path


@pytest.fixture
def raw(tmp_path):
    path, _, _ = _states_path.__wrapped__(tmp_path)
    dataset = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    dataset.meta.readonly = False
    dataset.meta.acquisition["nuclei"] = ("27Al", "27Al")
    return dataset


def _encoded(raw, mode):
    source, profile = _acquisition(mode)
    cosine, sine = raw.data
    if mode == "echo-antiecho":
        source.data = np.stack(((cosine - 1j * sine) / 2, (-cosine - 1j * sine) / 2))
    else:
        sign = (
            (1 - 2 * (np.arange(raw.shape[-2]) % 2))[:, None]
            if mode == "states-tppi"
            else 1
        )
        cosine, sine = cosine * sign, sine * sign
        source.data = np.stack((cosine.real, cosine.imag, sine.real, sine.imag))
    source.axes[0].values = raw.y.to("s").data.copy()
    source.axes[1].values = raw.x.to("s").data.copy()
    source.acquisition = deepcopy(dict(raw.meta.acquisition))
    source.acquisition["indirect_sampling"] = mode
    source.normalization_profile = "no-experiment-or-vendor-selection"
    source.metadata = {}
    return from_interop(source, profile=profile)


@pytest.mark.parametrize(
    "experiment,ratio,scale",
    [
        ("3QMAS", 19 / 12, 17 / 12),
        ("5QMAS", -25 / 12, 85 / 12),
        ("STMAS", 7 / 24, 17 / 24),
        ("ZQ-TROSY", -1, 1),
    ],
)
@pytest.mark.parametrize("shift", [0, -0.1, 0.1])
@pytest.mark.parametrize("points", [2, 16])
def test_all_experiments_use_the_same_quadrature_decoding(
    raw, experiment, ratio, scale, shift, points
):
    if experiment == "ZQ-TROSY":
        raw.meta.acquisition["nuclei"] = ("15N", "1H")
        raw.meta.acquisition["acquisition_frequency_mhz"] = (50.7, 500)
        raw.meta.acquisition["reference_frequency_mhz"] = (50.699, 499.999)
    expected = None
    for mode in ("states", "states-tppi", "echo-antiecho"):
        encoded = _encoded(raw, mode)
        before = encoded.copy()
        phased = _phase(encoded, points=points)
        result = scp.nmr.Experiment(phased).xfshear(
            size=32, experiment=experiment, f1_shift=shift
        )
        geometry = result.meta.mqmas_shearing
        assert geometry["ratio"] == pytest.approx(ratio)
        assert geometry["scale_divisor"] == pytest.approx(scale)
        assert phased.meta.mqmas_xf2["indirect_quadrature"] == mode.upper()
        if expected is None:
            expected = result
        else:
            np.testing.assert_allclose(
                result.data, expected.data, rtol=1e-12, atol=1e-9
            )
            np.testing.assert_array_equal(result.y.data, expected.y.data)
            np.testing.assert_array_equal(result.x.data, expected.x.data)
        np.testing.assert_array_equal(encoded.data, before.data)
        assert encoded.meta == before.meta


@pytest.mark.parametrize(
    "nucleus,ratio,scale",
    [
        ("87Rb", -8 / 9, 17 / 9),
        ("27Al", 7 / 24, 17 / 24),
        ("51V", 28 / 45, 17 / 45),
        ("93Nb", 55 / 72, 17 / 72),
    ],
)
def test_stmas_spin_table_is_resolved_from_isotope(raw, nucleus, ratio, scale):
    raw.meta.acquisition["nuclei"] = (nucleus, nucleus)
    result = scp.nmr.Experiment(_phase(raw)).xfshear(size=32, experiment="STMAS")
    assert result.meta.mqmas_shearing["ratio"] == pytest.approx(ratio)
    assert result.meta.mqmas_shearing["scale_divisor"] == pytest.approx(scale)


@pytest.mark.parametrize("mode", ["states", "states-tppi", "echo-antiecho"])
@pytest.mark.parametrize("indirect_nucleus", ["1H", "15N", "off", None])
def test_stmas_ignores_unused_indirect_isotope(raw, mode, indirect_nucleus):
    expected = scp.nmr.Experiment(_phase(_encoded(raw, mode))).xfshear(
        size=32, experiment="STMAS", f1_shift=-0.1
    )
    raw.meta.acquisition["nuclei"] = (indirect_nucleus, "27Al")
    raw.meta.acquisition["nuclear_spin"] = (0.5, 2.5)
    raw.meta.acquisition["acquisition_frequency_mhz"] = (500.0, 130.0)
    raw.meta.acquisition["reference_frequency_mhz"] = (499.99, 129.999)
    encoded = _encoded(raw, mode)
    before = deepcopy(dict(encoded.meta.acquisition))
    actual = scp.nmr.Experiment(_phase(encoded)).xfshear(
        size=32, experiment="STMAS", f1_shift=-0.1
    )
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.x.data, expected.x.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)
    assert actual.meta.mqmas_shearing["spin"] == 2.5
    assert dict(encoded.meta.acquisition) == before


def test_stmas_still_rejects_an_unsuitable_observed_spin(raw):
    raw.meta.acquisition["nuclei"] = ("27Al", "1H")
    with pytest.raises(ValueError, match="STMAS requires a half-integer spin"):
        scp.nmr.Experiment(_phase(raw)).xfshear(size=32, experiment="STMAS")


def test_missing_isotope_uses_metadata_or_explicit_spin(raw):
    raw.meta.acquisition["nuclei"] = ("unknown", "unknown")
    phased = _phase(raw)
    with pytest.raises(ValueError, match="pass spin="):
        scp.nmr.Experiment(phased).xfshear(size=32, experiment="5QMAS")
    explicit = scp.nmr.Experiment(phased).xfshear(size=32, experiment="5QMAS", spin=2.5)
    raw.meta.acquisition["nuclear_spin"] = (2.5, 2.5)
    metadata = scp.nmr.Experiment(_phase(raw)).xfshear(size=32, experiment="5QMAS")
    np.testing.assert_array_equal(metadata.data, explicit.data)
    with pytest.raises(ValueError, match="does not match"):
        scp.nmr.Experiment(_phase(raw)).xfshear(size=32, spin=1.5)


@pytest.mark.parametrize("shift", [0, -0.1, 0.1])
def test_zq_trosy_preserves_independent_f1_reference(raw, shift):
    raw.meta.acquisition["nuclei"] = ("15N", "1H")
    raw.meta.acquisition["acquisition_frequency_mhz"] = (50.7, 500)
    raw.meta.acquisition["reference_frequency_mhz"] = (50.699, 499.999)
    phased = _phase(raw)
    result = scp.nmr.Experiment(phased).xfshear(
        size=32, experiment="ZQ-TROSY", f1_shift=shift
    )
    bins = np.arange(32) - 16
    expected = (50.7 - 50.699) * 1e6 / 50.699 - bins * 10000 / (32 * 50.699)
    np.testing.assert_allclose(result.y.data, expected, atol=1e-10)
    assert result.y.meta["reference_frequency"].to("MHz").magnitude == pytest.approx(
        50.699
    )
    phased.meta.readonly = False
    phased.meta.acquisition["reference_frequency_mhz"] = (None, 499.999)
    with pytest.raises(ValueError, match="normalized F1"):
        scp.nmr.Experiment(phased).xfshear(size=32, experiment="ZQ-TROSY")


@pytest.mark.parametrize(
    "options",
    [
        {"experiment": "COSY"},
        {"experiment": "STMAS", "coherence_order": 3},
        {"experiment": "STMAS", "coherence_order": True},
        {"experiment": "5QMAS", "coherence_order": 3},
        {"experiment": "3QMAS", "spin": 1.5},
    ],
)
def test_conflicting_geometry_is_rejected_without_mutation(raw, options):
    phased = _phase(raw)
    before = phased.copy()
    with pytest.raises(ValueError):
        scp.nmr.Experiment(phased).xfshear(size=32, **options)
    np.testing.assert_array_equal(phased.data, before.data)
    assert phased.meta == before.meta


@pytest.mark.parametrize("mode", ["states", "states-tppi", "echo-antiecho"])
def test_raw_and_phased_scp_resume_without_reconstructing_twice(raw, mode, tmp_path):
    encoded = _encoded(raw, mode)
    expected = scp.nmr.Experiment(_phase(encoded)).xfshear(size=32, experiment="5QMAS")
    encoded.write(tmp_path / "raw.scp", confirm=False)
    restored = scp.nmr.read(tmp_path / "raw.scp")
    phased = _phase(restored)
    phased.write(tmp_path / "phased.scp", confirm=False)
    resumed = scp.nmr.read(tmp_path / "phased.scp")
    actual = scp.nmr.Experiment(resumed).xfshear(size=32, experiment="5QMAS")
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)
    assert restored.meta.mqmas_xf2 is None
    if mode == "echo-antiecho":
        assert resumed.meta.echo_antiecho_reconstruction == "rance-kay"


@pytest.mark.parametrize("value", [True, np.bool_(False), -1, 0, np.nan, np.inf, "5/2"])
def test_invalid_normalized_spin_is_not_silently_ignored(raw, value):
    raw.meta.acquisition["nuclear_spin"] = value
    phased = _phase(raw)
    with pytest.raises(ValueError, match="spin"):
        scp.nmr.Experiment(phased).xfshear(size=32, experiment="STMAS")


def test_normalized_spin_conflicting_with_isotope_is_rejected(raw):
    raw.meta.acquisition["nuclear_spin"] = 1.5  # 27Al has spin 5/2.
    with pytest.raises(ValueError, match="does not match"):
        scp.nmr.Experiment(_phase(raw)).xfshear(size=32, experiment="3QMAS")


def test_explicit_spin_conflict_retains_the_isotope_diagnostic(raw):
    with pytest.raises(ValueError, match="spin 1.5 does not match 27Al"):
        scp.nmr.Experiment(_phase(raw)).xfshear(size=32, spin=1.5)


def test_zq_trosy_does_not_require_an_unneeded_spin(raw):
    raw.meta.acquisition["nuclei"] = (None, None)
    result = scp.nmr.Experiment(_phase(raw)).xfshear(size=32, experiment="ZQ-TROSY")
    assert result.meta.mqmas_shearing["spin"] is None
    assert result.meta.mqmas_shearing["ratio"] == -1


def test_advanced_coefficients_override_geometry_and_keep_zq_reference(raw):
    raw.meta.acquisition["nuclei"] = (None, None)
    raw.meta.acquisition["reference_frequency_mhz"] = (50.699, 129.999)
    raw.meta.acquisition["acquisition_frequency_mhz"] = (50.7, 130)
    result = scp.nmr.Experiment(_phase(raw)).xfshear(
        size=32, experiment="ZQ-TROSY", ratio=0, f1_scale=1
    )
    assert result.meta.mqmas_shearing["ratio"] == 0
    assert result.meta.mqmas_shearing["coefficient_source"] == "explicit"
    assert result.y.meta["reference_frequency"].to("MHz").magnitude == pytest.approx(
        50.699
    )


def test_public_bruker_echo_antiecho_read_supplies_shear_context(tmp_path):
    from types import SimpleNamespace
    from qualification.conftest import paired_ser

    path = paired_ser.__wrapped__(tmp_path, SimpleNamespace(param=6))
    for name in ("acqus", "acqu2s"):
        file = path / name
        file.write_text(file.read_text().replace("13C", "27Al"))
    source = scp.nmr.read(path)
    assert source.meta.quadrature_mode == "echo-antiecho-stored-2d"
    result = scp.nmr.Experiment(_phase(source)).xfshear(size=16, experiment="5QMAS")
    assert result.shape == (2, 16, 32)
    assert np.isfinite(result.data).all()


@pytest.mark.parametrize(
    "canonical,alias",
    [
        (None, 2.5),
        ((None, None), 2.5),
        (None, (1.5, 2.5)),
        ((None, None), (None, 2.5)),
        ((None, 2.5), (None, None)),
        ((None, 2.5), (None, 2.5)),
    ],
)
def test_available_spin_alias_is_used_even_when_canonical_is_empty(
    raw, canonical, alias
):
    raw.meta.acquisition["nuclei"] = ("unknown", "unknown")
    reference = scp.nmr.Experiment(_phase(raw)).xfshear(
        size=32, experiment="5QMAS", spin=2.5
    )
    raw.meta.acquisition["nuclear_spin"] = canonical
    raw.meta.acquisition["spin"] = alias
    actual = scp.nmr.Experiment(_phase(raw)).xfshear(size=32, experiment="5QMAS")
    np.testing.assert_array_equal(actual.data, reference.data)
    assert actual.meta.mqmas_shearing["spin"] == 2.5


@pytest.mark.parametrize(
    "canonical,alias",
    [(2.5, 1.5), ((None, 2.5), 1.5), (2.5, (None, 1.5)), ((2.5, 2.5), (None, 1.5))],
)
def test_conflicting_spin_aliases_are_both_validated(raw, canonical, alias):
    raw.meta.acquisition["nuclei"] = ("unknown", "unknown")
    raw.meta.acquisition["nuclear_spin"] = canonical
    raw.meta.acquisition["spin"] = alias
    phased = _phase(raw)
    before = phased.copy()
    with pytest.raises(ValueError, match="does not match"):
        scp.nmr.Experiment(phased).xfshear(size=32, experiment="5QMAS")
    np.testing.assert_array_equal(phased.data, before.data)
    assert phased.meta == before.meta
