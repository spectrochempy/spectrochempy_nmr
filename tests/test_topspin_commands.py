# ruff: noqa: S101
"""In-memory TopSpin command equivalents."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy_nmr import Experiment


def _dataset(data, *, frequency=(True, True)):
    paired = data.ndim == 3
    dataset = scp.NDDataset(data, dims=["c", "y", "x"] if paired else ["y", "x"])
    dataset.set_coordset(
        **({"c": scp.Coord(np.arange(data.shape[0]))} if paired else {}),
        y=scp.Coord(np.arange(data.shape[-2], dtype=float), units="ppm"),
        x=scp.Coord(np.arange(data.shape[-1], dtype=float)[::-1], units="ppm"),
    )
    dataset.meta.isfreq = ([False] if paired else []) + list(frequency)
    if paired:
        dataset.meta.component_labels = ("S_R", "S_I")
    return dataset


def test_numbered_slices_keep_the_correct_complex_components():
    data = np.arange(24).reshape(2, 3, 4) * (1 + 3j)
    dataset = _dataset(data)
    exp = Experiment(dataset)
    row, col = exp.rsr(1), exp.rsc(2)
    np.testing.assert_array_equal(row.data, data[0, 0])
    np.testing.assert_array_equal(
        col.data, data[0, :, 1].real + 1j * data[1, :, 1].real
    )
    np.testing.assert_array_equal(row.coord(0).data, dataset.x.data)
    np.testing.assert_array_equal(col.coord(0).data, dataset.y.data)
    assert row.coord(0).units == col.coord(0).units == scp.ur.ppm
    with pytest.raises(IndexError, match="between 1 and 3"):
        exp.rsr(0)


def test_rser_counts_both_branches_in_ser_order():
    data = np.arange(24).reshape(2, 3, 4) * (1 + 3j)
    exp = Experiment(_dataset(data, frequency=(False, False)))
    for number, branch, row in [(1, 0, 0), (2, 1, 0), (3, 0, 1), (6, 1, 2)]:
        np.testing.assert_array_equal(exp.rser(number).data, data[branch, row])
    with pytest.raises(ValueError, match="frequency-domain F2"):
        exp.rsr(1)


@pytest.mark.parametrize("command", ["abs1", "abs2"])
def test_baseline_corrects_only_rr_and_preserves_source(command):
    rr = (np.arange(8)[:, None] + 2) * (np.arange(10)[None, :] + 3)
    data = np.stack([rr + 3j, rr * (2 + 4j)])
    dataset = _dataset(data)
    corrected = getattr(Experiment(dataset), command)(order=1)
    np.testing.assert_allclose(corrected.data[0].real, 0, atol=1e-12)
    np.testing.assert_array_equal(corrected.data[0].imag, data[0].imag)
    np.testing.assert_array_equal(corrected.data[1], data[1])
    np.testing.assert_array_equal(dataset.data, data)


def test_baseline_limits_and_fit_ranges_with_a_descending_axis():
    values = np.arange(10)[::-1]
    data = np.tile(2.0 * values + 3, (4, 1))
    data[:, 4] += 7
    dataset = _dataset(data)
    corrected = Experiment(dataset).abs2(
        order=1, limits=(2, 8), ranges=[(2, 4), (7, 8)]
    )
    expected = data.copy()
    expected[:, 1:8] = 0
    expected[:, 4] = 7
    np.testing.assert_allclose(corrected.data, expected, atol=1e-12)
    np.testing.assert_array_equal(dataset.data, data)


@pytest.mark.parametrize("stored_names", [False, True])
def test_four_plane_spectrum(stored_names):
    data = np.arange(4 * 8 * 10, dtype=float).reshape(4, 8, 10)
    dataset = _dataset(data)
    labels = ("rr", "ri", "ir", "ii")
    dataset.meta.component_labels = tuple(
        "2" + label if stored_names else label for label in labels
    )
    exp = Experiment(dataset)
    f2, f1 = (2, 1) if stored_names else (1, 2)
    np.testing.assert_array_equal(exp.rsr(1).data, data[0, 0] + 1j * data[f2, 0])
    np.testing.assert_array_equal(exp.rsc(1).data, data[0, :, 0] + 1j * data[f1, :, 0])
    corrected = exp.abs1(order=1)
    np.testing.assert_allclose(corrected.data[0], 0, atol=1e-12)
    np.testing.assert_array_equal(corrected.data[1:], data[1:])


def test_complex_axis_is_respected_for_a_paired_spectrum():
    data = np.arange(24).reshape(2, 3, 4) * (1 + 3j)
    dataset = _dataset(data)
    dataset.meta.interop_complex_axis = "F1"
    exp = Experiment(dataset)
    np.testing.assert_array_equal(
        exp.rsr(1).data, data[0, 0].real + 1j * data[1, 0].real
    )
    np.testing.assert_array_equal(exp.rsc(1).data, data[0, :, 0])


@pytest.mark.parametrize("direction", ["abs1", "abs2"])
def test_default_baseline_excludes_signed_peaks(direction):
    axis = np.linspace(-1, 1, 100)
    values = np.tile(3 + 2 * axis, (100, 1))
    values[:, 30:33] += 60
    values[:, 62:65] -= 50
    expected = values - (3 + 2 * axis)
    if direction == "abs1":
        values, expected = values.T, expected.T
    result = getattr(Experiment(_dataset(values)), direction)(order=1)
    np.testing.assert_allclose(result.data, expected, atol=1e-12)


def test_apk_is_the_public_autophase_alias():
    assert scp.nmr.apk is scp.nmr.autophase


def test_explicit_baseline_ranges_exclude_masked_values():
    axis = np.arange(10)
    data = np.tile(2.0 * axis[::-1] + 3, (4, 1))
    data[0, 2] = 1000
    dataset = _dataset(data)
    mask = np.zeros(data.shape, dtype=bool)
    mask[0, 2] = True
    dataset.mask = mask
    result = Experiment(dataset).abs2(order=1, ranges=[(0, 3), (6, 9)])
    np.testing.assert_allclose(result.data[~mask], 0, atol=1e-12)
    np.testing.assert_array_equal(result.mask, mask)


def test_baseline_limits_reject_nonmonotonic_axes():
    dataset = _dataset(np.tile([3.0, 9.0, 19.0, 11.0, 21.0], (4, 1)))
    dataset.x = scp.Coord([0.0, 3.0, 8.0, 4.0, 9.0], units="ppm")
    with pytest.raises(ValueError, match="strictly monotonic"):
        Experiment(dataset).abs2(order=1, limits=(3, 4))


def test_column_preserves_f1_phase_metadata():
    data = np.arange(24).reshape(2, 3, 4) * (1 + 3j)
    dataset = _dataset(data)
    dataset.meta.interop_complex_axis = "F1"
    dataset.meta.exptc = [0, 1, 2]
    dataset.meta.physical_pivots = [False, True, False]
    exp = Experiment(dataset)
    col = exp.rsc(1)
    np.testing.assert_array_equal(col.data, data[0, :, 0])
    assert col.meta.exptc == [1]
    assert col.meta.physical_pivots == [True]


def test_bruker_public_profile_keeps_native_topspin_slice_components():
    data = np.arange(24).reshape(2, 3, 4) * (1 + 3j)
    dataset = _dataset(data)
    dataset.meta.normalization_profile = "bruker-pdata-2d-public-v1"
    dataset.meta.interop_complex_axis = "F1"
    exp = Experiment(dataset)
    np.testing.assert_array_equal(exp.rsr(1).data, data[0, 0])
    expected = data[0, :, 0].real + 1j * data[1, :, 0].real
    np.testing.assert_array_equal(exp.rsc(1).data, expected)


@pytest.mark.parametrize(
    ("reader", "writer", "imaginary", "location"),
    [("rsr", "wsr", 2, np.s_[0, :]), ("rsc", "wsc", 1, np.s_[:, 0])],
)
def test_replace_modified_topspin_trace(reader, writer, imaginary, location):
    data = np.arange(4 * 8 * 10, dtype=float).reshape(4, 8, 10)
    dataset = _dataset(data)
    dataset.meta.component_labels = ("2rr", "2ri", "2ir", "2ii")
    exp = Experiment(dataset)
    trace = getattr(exp, reader)(1) * 1j
    replace = getattr(exp, writer)
    result = replace(1, trace)
    expected = data.copy()
    expected[0][location] = -data[imaginary][location]
    expected[imaginary][location] = data[0][location]
    np.testing.assert_array_equal(result.data, expected)
    np.testing.assert_array_equal(dataset.data, data)
    with pytest.raises(IndexError):
        replace(0, trace)
    with pytest.raises(ValueError, match="coordinate grid"):
        shifted = trace.copy()
        shifted.set_coordset(scp.Coord(trace.coord(0).data + 1, units="ppm"))
        replace(1, shifted)


def test_replace_column_preserves_unrelated_packed_imaginary_channels():
    data = np.arange(24).reshape(2, 3, 4) * (1 + 3j)
    dataset = _dataset(data)
    exp = Experiment(dataset)
    trace = exp.rsc(2) * 1j
    result = exp.wsc(2, trace)
    np.testing.assert_array_equal(result.data.imag, data.imag)
    np.testing.assert_array_equal(Experiment(result).rsc(2).data, trace.data)


def test_replace_row_replaces_mask_only_in_target_trace():
    dataset = _dataset(np.arange(24, dtype=float).reshape(4, 6))
    mask = np.zeros(dataset.shape, dtype=bool)
    mask[0, 1] = mask[1, 2] = True
    dataset.mask = mask
    exp = Experiment(dataset)
    trace = exp.rsr(1)
    trace.remove_masks()
    result = exp.wsr(1, trace)
    expected = mask.copy()
    expected[0] = False
    np.testing.assert_array_equal(result.mask, expected)
    np.testing.assert_array_equal(dataset.mask, mask)


def test_replace_packed_column_preserves_individual_component_masks():
    dataset = _dataset(np.arange(24).reshape(2, 3, 4) * (1 + 3j))
    mask = np.zeros(dataset.shape, dtype=bool)
    mask[0, 0, 1] = True
    dataset.mask = mask
    exp = Experiment(dataset)
    result = exp.wsc(2, exp.rsc(2))
    np.testing.assert_array_equal(result.mask, mask)
    np.testing.assert_array_equal(result.data[~mask], dataset.data[~mask])


@pytest.mark.parametrize("per_row", [False, True])
def test_apk2d_applies_reference_or_individual_phase(per_row):
    x = np.arange(512)
    line = 1 / (1 + 1j * (x - 180) / 8)
    injected = np.array([29.0, -43.0, 75.0])
    raw = line * np.exp(1j * np.deg2rad(injected[:, None]))
    dataset = _dataset(raw, frequency=(False, True))
    result = Experiment(dataset).apk2d(
        reference_row=2,
        per_row=per_row,
        method="dispa",
        references=iter([(150, 211, 1)]),
    )
    expected = (
        line if per_row else line * np.exp(1j * np.deg2rad(injected[:, None] + 43))
    )
    np.testing.assert_allclose(
        result.data, np.broadcast_to(expected, raw.shape), atol=0.01
    )
    np.testing.assert_array_equal(dataset.data, raw)
    np.testing.assert_array_equal(result.x.data, dataset.x.data)


def test_apk2d_identifies_the_failing_row():
    x = np.arange(512)
    line = 1 / (1 + 1j * (x - 180) / 8)
    dataset = _dataset(np.stack([line, np.zeros_like(line)]), frequency=(False, True))

    with pytest.raises(ValueError, match="row 2: spectrum has no signal"):
        Experiment(dataset).apk2d(
            per_row=True, method="dispa", references=[(150, 211, 1)]
        )
