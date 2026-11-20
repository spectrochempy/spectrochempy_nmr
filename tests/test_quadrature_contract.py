"""Common raw 2D processing depends on quadrature and component meaning."""

from copy import deepcopy

from nmrglue import interop
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import from_interop
from spectrochempy_nmr.experiment import Experiment
from test_interop_conversion import _normalized


MODES = ("qf", "states", "states-tppi", "tppi", "echo-antiecho")


def _acquisition(mode):
    y, x = np.indices((8, 16))
    first = np.exp(-y / 9 - x / 17) * np.exp(1j * (0.71 * x + 0.32 * y + 0.4))
    second = 0.63 * np.exp(-y / 11 - x / 23) * np.exp(1j * (0.43 * x - 0.51 * y))
    source = _normalized(first, axis_values=np.arange(16) / 4000)
    f1 = deepcopy(source.axes[0])
    f1.values = np.arange(8) / 2000
    source.axes = (f1, *source.axes)
    source.physical_dimensions = ("F1", "F2")
    source.domains = ("time", "time")
    source.acquisition = {"spectral_width_hz": (2000, 4000)}
    source.provenance = {}
    source.metadata = {}
    source.normalization_profile = "arbitrary-file-format"
    profile = None
    if mode == "tppi":
        source.dimensions = ("F1", "F2")
        source.encoding = ("tppi", "complex")
        source.quadrature_mode = "tppi-stored-2d"
        source.branch_order = ("complex",)
        source.acquisition = {"spectral_widths_hz": (2000, 4000)}
    elif mode == "echo-antiecho":
        source.data = np.stack((first, second))
        source.dimensions = ("channels", "F1", "F2")
        source.channel_axis = 0
        source.encoding = ("echo-antiecho", "complex")
        source.quadrature_mode = "echo-antiecho-stored-2d"
        source.branch_order = ("first", "second")
    else:
        source.dimensions = ("channels", "F1", "F2")
        source.channel_axis = 0
        source.quadrature_mode = "stored-components-2d"
        if mode == "qf":
            source.data = np.stack((first.real, first.imag))
            source.encoding = ("real", "complex")
            source.branch_order = ("rr", "ri")
            source.acquisition["indirect_sampling"] = "qf"
        else:
            source.data = np.stack((first.real, first.imag, second.real, second.imag))
            source.encoding = ("complex", "complex")
            source.branch_order = ("rr", "ri", "ir", "ii")
            profile = mode
    return source, profile


@pytest.mark.parametrize("mode", MODES)
def test_read_process_persists_without_provider_or_experiment_identity(
    mode, monkeypatch, tmp_path
):
    normalized, profile = _acquisition(mode)
    monkeypatch.setattr(interop, "read", lambda *a, **kw: object())
    monkeypatch.setattr(interop, "normalize", lambda *a, **kw: normalized)
    acquisition = scp.nmr.read("input", protocol="interop", conversion_profile=profile)
    expected = Experiment(acquisition).process()
    before = acquisition.data.copy()
    filename = tmp_path / "raw.scp"
    acquisition.write(filename, confirm=False)
    restored = scp.read(filename)

    def unavailable(*args, **kwargs):
        pytest.fail("Processing must not consult the binary reader or normalizer")

    monkeypatch.setattr(interop, "read", unavailable)
    monkeypatch.setattr(interop, "normalize", unavailable)
    for vendor, experiment in (("jeol", "COSY"), ("topspin", "HSQC"), ("other", None)):
        restored.meta.readonly = False
        restored.origin = vendor
        restored.meta.origin = vendor
        restored.meta.normalization_profile = "unrelated-profile"
        restored.meta.provenance = {"vendor": vendor, "experiment": experiment}
        restored.meta.source_metadata = {
            "experiment": experiment,
            "acqus": {"GRPDLY": 99},
        }
        restored.meta.pulprog = experiment
        actual = Experiment(restored).process()
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.y.data, expected.y.data)
        np.testing.assert_array_equal(actual.x.data, expected.x.data)
    np.testing.assert_array_equal(acquisition.data, before)


@pytest.mark.parametrize("mode", ("qf", "states", "states-tppi", "echo-antiecho"))
def test_declared_component_permutations_produce_one_arrangement(mode):
    normalized, profile = _acquisition(mode)
    expected = from_interop(normalized, profile=profile)
    permutation = tuple(range(normalized.data.shape[0] - 1, -1, -1))
    normalized.data = normalized.data[list(permutation)]
    normalized.branch_order = tuple(normalized.branch_order[i] for i in permutation)
    before = normalized.data.copy()
    actual = from_interop(normalized, profile=profile)
    np.testing.assert_array_equal(actual.data, expected.data)
    if actual.ndim == 3:
        assert tuple(actual.c.labels) == tuple(expected.c.labels)
    np.testing.assert_array_equal(
        Experiment(actual).process().data, Experiment(expected).process().data
    )
    np.testing.assert_array_equal(normalized.data, before)


@pytest.mark.parametrize("mode", ("states", "states-tppi"))
def test_states_stored_complex_pairs_match_normalized_real_planes(mode):
    normalized, profile = _acquisition(mode)
    expected = from_interop(normalized, profile=profile)
    normalized.data = expected.data.conj()
    normalized.data[1] *= -1
    normalized.encoding = ("states", "complex")
    normalized.quadrature_mode = "states-stored-2d"
    normalized.branch_order = ("branch_0", "branch_1")
    normalized.acquisition.update(
        quadrature_selection={"mode": mode},
        states_fft_input_transform="conjugate-and-negate-second-branch",
    )
    before = normalized.data.copy()
    actual = from_interop(normalized)
    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.meta.acquisition["states_fft_input_transform"] is None
    assert actual.meta.applied_operations[-1]["stage"] == "quadrature-normalization"
    np.testing.assert_array_equal(normalized.data, before)
    np.testing.assert_array_equal(
        Experiment(actual).process().data, Experiment(expected).process().data
    )


def test_filter_uses_imported_delay_and_ignores_later_vendor_parameters(monkeypatch):
    normalized, profile = _acquisition("echo-antiecho")
    normalized.metadata = {"acqus": {"GRPDLY": 2.5}}
    source = from_interop(normalized)
    expected = Experiment(source).process(digital_filter="phase")
    source.meta.source_metadata = {"acqus": {"GRPDLY": 7.25}}
    source.meta.normalization_profile = "different-format"
    actual = Experiment(source).process(digital_filter="phase")
    np.testing.assert_array_equal(actual.data, expected.data)
    assert actual.meta.digital_filter["group_delay_points"] == 2.5
    source.meta.acquisition["digital_filter"] = {}
    with pytest.raises(ValueError, match="normalized acquisition"):
        Experiment(source).process(digital_filter="phase")


@pytest.mark.parametrize("mode", ("qf", "states", "states-tppi", "echo-antiecho"))
def test_duplicate_components_are_rejected_without_reinterpreting_them(mode):
    normalized, profile = _acquisition(mode)
    normalized.branch_order = (normalized.branch_order[0],) * len(
        normalized.branch_order
    )
    with pytest.raises(ValueError):
        from_interop(normalized, profile=profile)
