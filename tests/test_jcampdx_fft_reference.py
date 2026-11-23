"""Explicit point calibration after the common 2D FFT, via public APIs."""

from copy import deepcopy
from inspect import signature

import sys

import numpy as np
import pytest
from nmrglue import interop

from spectrochempy_nmr import Experiment, read
from test_jcampdx_qf_interop import _write_qf
from test_jcampdx_states import _write


pytestmark = pytest.mark.skipif(
    not all(
        hasattr(interop, name)
        for name in ("read_jcampdx_fid_2d", "normalize_jcampdx_fid_2d")
    ),
    reason="requires nmrglue-ng JCAMP 2D interop profiles",
)


REFERENCE = {
    "F1": {
        "reference_frequency_mhz": 100.0,
        "point_index": 1,
        "ppm": 8.0,
        "source": "processing",
    },
    "F2": {
        "reference_frequency_mhz": 400.0,
        "point_index": 1,
        "ppm": -1.0,
        "source": "processing",
    },
}


def _raw(tmp_path, mode):
    path = tmp_path / "fid.dx"
    if mode == "qf":
        _write_qf(path)
        options = {"quadrature": "qf", "conjugation": "none"}
        return read(
            path,
            protocol="interop",
            profile="jcampdx-fid-2d",
            normalize_options=options,
        )
    normalizer = getattr(interop, "normalize_jcampdx_fid_2d", lambda: None)
    if "confirm_adjacent_pairs" not in signature(normalizer).parameters:
        pytest.skip("requires development adjacent-pair normalization")
    if mode.startswith("states") and '"states-tppi"' not in (normalizer.__doc__ or ""):
        pytest.skip("requires development States-family normalization")
    traces = np.arange(24).reshape(4, 6) + 1j * np.arange(24, 48).reshape(4, 6)
    _write(path, traces, mode)
    return read(
        path,
        protocol="interop",
        profile="jcampdx-fid-2d",
        conversion_profile=mode if mode != "echo-antiecho" else None,
        normalize_options={
            "quadrature": mode,
            "conjugation": "none",
            "confirm_adjacent_pairs": True,
            "indirect_dwell_s": 0.002,
        },
    )


@pytest.mark.parametrize("mode", ["qf", "states", "states-tppi", "echo-antiecho"])
def test_public_fft_reference_preserves_signed_components_and_scp(tmp_path, mode):
    raw = _raw(tmp_path, mode)
    before = raw.data.copy()
    plain = Experiment(raw).process()
    refs = deepcopy(REFERENCE)
    result = Experiment(raw).process(reference=refs)
    np.testing.assert_array_equal(result.data, plain.data)
    np.testing.assert_array_equal(raw.data, before)
    for role, dim in [("F1", "y"), ("F2", "x")]:
        spec = refs[role]
        hz = plain.coordset[dim].data
        expected = spec["ppm"] + (hz - hz[1]) / spec["reference_frequency_mhz"]
        np.testing.assert_allclose(
            result.coordset[dim].data, expected, rtol=0, atol=1e-13
        )
        assert str(result.coordset[dim].units) == "ppm"
        assert (
            result.coordset[dim].meta.reference_frequency.to("MHz").magnitude
            == spec["reference_frequency_mhz"]
        )
    saved = tmp_path / "spectrum.scp"
    result.write(saved, confirm=False)
    restored = read(saved)
    np.testing.assert_array_equal(restored.data, result.data)
    np.testing.assert_array_equal(restored.x.data, result.x.data)
    assert restored.meta.frequency_reference["axes"]["F1"]["point_index"] == 1
    refs["F1"]["ppm"] = 1000
    assert result.meta.frequency_reference["axes"]["F1"]["ppm"] == 8


@pytest.mark.parametrize("reference", [{}, {"F1": REFERENCE["F1"]}, "primary", False])
def test_rejects_incomplete_2d_reference(tmp_path, reference):
    with pytest.raises(ValueError):
        Experiment(_raw(tmp_path, "qf")).process(reference=reference)


@pytest.mark.parametrize("axis", ["F1", "F2"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_frequency_mhz", True),
        ("reference_frequency_mhz", 0),
        ("reference_frequency_mhz", -1),
        ("reference_frequency_mhz", np.inf),
        ("reference_frequency_mhz", "400"),
        ("reference_frequency_mhz", 1e-320),
        ("reference_frequency_mhz", 1e300),
        ("point_index", True),
        ("point_index", 1.0),
        ("point_index", -1),
        ("point_index", 100),
        ("ppm", np.nan),
        ("ppm", True),
        ("ppm", "8"),
        ("ppm", 1e300),
        ("source", "inferred"),
        ("extra", 1),
    ],
)
def test_invalid_axis_reference_is_rejected(tmp_path, axis, field, value):
    refs = deepcopy(REFERENCE)
    refs[axis][field] = value
    raw = _raw(tmp_path, "qf")
    before = raw.data.copy()
    with pytest.raises(ValueError):
        Experiment(raw).process(reference=refs)
    np.testing.assert_array_equal(raw.data, before)


def test_preserves_true_acquisition_frequency_separately(tmp_path):
    raw = _raw(tmp_path, "qf")
    raw.meta.acquisition["acquisition_frequency_mhz"] = (99.0, 399.0)
    result = Experiment(raw).process(reference=REFERENCE)
    assert result.x.meta.acquisition_frequency.to("MHz").magnitude == 399
    assert result.x.meta.reference_frequency.to("MHz").magnitude == 400
    assert result.meta.acquisition["acquisition_frequency_mhz"] == (99.0, 399.0)


def test_zero_fill_anchors_use_output_indices(tmp_path):
    raw = _raw(tmp_path, "qf")
    plain = Experiment(raw).process(size=(4, 8))
    refs = deepcopy(REFERENCE)
    refs["F2"]["point_index"] = np.int64(7)
    refs["F2"]["reference_frequency_mhz"] = np.float64(400)
    result = Experiment(raw).process(size=(4, 8), reference=refs)
    np.testing.assert_array_equal(result.data, plain.data)
    np.testing.assert_allclose(
        result.x.data, -1 + (plain.x.data - plain.x.data[7]) / 400, rtol=0, atol=1e-14
    )


def test_reference_guard_rejects_raw_and_double_calibration(tmp_path):
    from spectrochempy_nmr.processing.interop_reference import _reference_interop_2d

    raw = _raw(tmp_path, "qf")
    with pytest.raises(ValueError, match="frequency axes"):
        _reference_interop_2d(raw, REFERENCE)
    result = Experiment(raw).process(reference=REFERENCE)
    with pytest.raises(ValueError, match="already referenced"):
        _reference_interop_2d(result, REFERENCE)


def _plan(raw, mode="qf"):
    from spectrochempy_nmr.processing.topspin_xfb import _build_xfb_plan

    ny, nx = raw.shape[-2:]
    widths = raw.meta.acquisition["spectral_width_hz"]
    status = {
        "ME_mod": 0,
        "SI": 8,
        "FTSIZE": 8,
        "STSI": 8,
        "STSR": 0,
        "TDeff": ny if mode == "qf" else 2 * ny,
        "TDoff": 0,
        "FT_mod": 4 if mode == "states-tppi" else 6,
        "PH_mod": 2 if mode == "qf" else 1,
        "REVERSE": False,
        "BC_mod": 0,
        "WDW": 1,
        "LB": 2.0,
        "FCOR": 1.0,
        "PHC0": 23.0,
        "PHC1": -11.0,
        "OFFSET": 8.0,
        "SW_p": widths[0],
        "SF": 100.0,
        "PKNL": False,
    }
    direct = {
        **status,
        "TDeff": 2 * nx,
        "FT_mod": 6,
        "PH_mod": 0 if mode == "qf" else 1,
        "SW_p": widths[1],
        "SF": 400.0,
        "OFFSET": 10.0,
        "PHC0": -7.0,
    }
    return _build_xfb_plan(
        {"proc2s": status, "procs": direct, "acqus": {"GRPDLY": 0.0}},
        mode,
        (ny, nx),
        widths,
    )


@pytest.mark.parametrize("mode", ["qf", "states", "states-tppi", "echo-antiecho"])
def test_explicit_recipe_uses_public_jcamp_source_and_survives_scp(
    tmp_path, monkeypatch, mode
):
    from spectrochempy_nmr.processing.topspin_xfb import _execute_xfb
    from spectrochempy_nmr.interop import _normalize_states_components

    raw = _raw(tmp_path, mode)
    plan = _plan(raw, mode)
    expected, _ = _execute_xfb(_normalize_states_components(raw), raw.data, plan)
    actual = Experiment(raw).process(replay=plan)
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.x.data, expected.x.data)
    assert str(actual.x.units) == "ppm"
    path = tmp_path / "raw.scp"
    raw.write(path, confirm=False)
    restored = read(path)
    restored.meta.source_metadata = {}
    restored.origin = "unrelated provenance label"
    restored.meta.normalization_profile = "unrelated profile label"
    monkeypatch.setitem(sys.modules, "nmrglue", None)
    again = Experiment(restored).process(replay=plan)
    np.testing.assert_array_equal(again.data, actual.data)


@pytest.mark.parametrize(
    "mutation", ["mode", "shape", "width", "schema", "nan", "double_filter"]
)
def test_recipe_rejects_incompatible_plan_or_corrected_source(tmp_path, mutation):
    raw = _raw(tmp_path, "qf")
    plan = _plan(raw)
    if mutation == "mode":
        plan["mode"] = "echo-antiecho"
    if mutation == "shape":
        plan["source_shape"] = [100, 100]
    if mutation == "width":
        plan["widths_hz"] = [1.0, 2.0]
    if mutation == "schema":
        plan["schema"] = "unknown"
    if mutation == "nan":
        plan["f2"]["phase_zero_deg"] = np.nan
    if mutation == "double_filter":
        raw.meta.acquisition["digital_filter"] = {"applied": True}
    with pytest.raises(ValueError):
        Experiment(raw).process(replay=plan)


def test_recipe_rejects_overrides_and_mismatched_quadrature(tmp_path):
    raw = _raw(tmp_path, "qf")
    with pytest.raises(ValueError):
        Experiment(raw).process(replay=_plan(raw), reference=REFERENCE)
    with pytest.raises(ValueError):
        Experiment(raw).process(replay=_plan(raw), size=8)
    states = _raw(tmp_path, "states")
    with pytest.raises(ValueError):
        Experiment(states).process(replay=_plan(raw))


def test_public_qf_recipe_against_explicit_matrix_dfts(tmp_path):
    raw = _raw(tmp_path, "qf")
    plan = _plan(raw)
    sy, sx = plan["f1"]["fft_size"], plan["f2"]["fft_size"]
    wy, wx = plan["widths_hz"]
    y, x = plan["f1"], plan["f2"]
    values = raw.data * np.exp(
        -np.pi * x["line_broadening_hz"] * np.arange(raw.shape[1]) / wx
    )
    f2 = (
        values
        @ np.exp(
            2j * np.pi * np.outer(np.arange(sx) - sx // 2, np.arange(raw.shape[1])) / sx
        ).T
    )
    f2 *= np.exp(
        -1j
        * np.deg2rad(x["phase_zero_deg"] + x["phase_first_deg"] * np.arange(sx) / sx)
    )
    f2 *= np.exp(-np.pi * y["line_broadening_hz"] * np.arange(raw.shape[0]) / wy)[
        :, None
    ]
    f1 = (
        np.exp(
            -2j
            * np.pi
            * np.outer(np.arange(sy) - sy // 2, np.arange(raw.shape[0]))
            / sy
        )
        @ f2
    )
    expected = np.abs(f1)
    expected[sy // 2, sx // 2] = (
        expected[sy // 2 - 1, sx // 2] + expected[sy // 2 + 1, sx // 2]
    ) / 2
    result = Experiment(raw).process(replay=plan)
    np.testing.assert_allclose(result.data, expected, rtol=1e-13, atol=1e-12)


def test_qf_physical_dwell_and_recipe_roundoff_are_explicit(tmp_path):
    raw = _raw(tmp_path, "qf")
    plan = _plan(raw)
    close = deepcopy(plan)
    close["widths_hz"] = [w * (1 + 1e-15) for w in plan["widths_hz"]]
    result = Experiment(raw).process(replay=close)
    trace = result.meta.nmr_processing["scp_processing"]["applied"]
    assert trace["width_match_rtol"] == 1e-12
    changed = deepcopy(plan)
    changed["widths_hz"] = [w * (1 + 1e-8) for w in plan["widths_hz"]]
    with pytest.raises(ValueError):
        Experiment(raw).process(replay=changed)


def test_readonly_captured_recipe_can_be_transported(tmp_path):
    raw = _raw(tmp_path, "qf")
    raw.meta.captured_recipe = _plan(raw)
    raw.meta.readonly = True
    result = Experiment(raw).process(replay=raw.meta.captured_recipe)
    assert raw.meta.readonly
    assert (
        result.meta.nmr_processing["scp_processing"]["requested"]["plan"]["mode"]
        == "qf"
    )


@pytest.mark.parametrize("mode", ["states", "states-tppi"])
def test_states_recipe_signed_planes_follow_common_dft_contract(tmp_path, mode):
    normalizer = interop.normalize_jcampdx_fid_2d
    if '"states-tppi"' not in (normalizer.__doc__ or ""):
        pytest.skip("requires development States normalization")
    path = tmp_path / "paired.dx"
    traces = np.arange(64).reshape(8, 8) + 1j * np.arange(64, 128).reshape(8, 8)
    _write(path, traces, mode)
    raw = read(
        path,
        protocol="interop",
        profile="jcampdx-fid-2d",
        conversion_profile=mode,
        normalize_options={
            "quadrature": mode,
            "conjugation": "none",
            "confirm_adjacent_pairs": True,
            "indirect_dwell_s": 0.002,
        },
    )
    plain = Experiment(raw).process()
    plan = _plan(raw, mode)
    plan["f1"]["fft_size"] = 4
    for axis in ("f1", "f2"):
        plan[axis].update(window_mode=0, phase_zero_deg=0.0, phase_first_deg=0.0)
    expected = plain.data[[0, 2, 1, 3]] * np.array([1, -1, -1, 1])[:, None, None]
    expected[0, 2, 4] = (expected[0, 1, 4] + expected[0, 3, 4]) / 2
    result = Experiment(raw).process(replay=plan)
    actual = np.stack(
        [getattr(Experiment(result), name)().data for name in ("rr", "ri", "ir", "ii")]
    )
    np.testing.assert_array_equal(actual, expected)
    assert result.meta.get("echo_antiecho_reconstruction") is None
    assert (result.meta.get("tppi_alternation_state") == "applied") == (
        mode == "states-tppi"
    )


@pytest.mark.parametrize("axis", ["f1", "f2"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_frequency_mhz", 1e-320),
        ("reference_frequency_mhz", 1e300),
        ("first_point_ppm", 1e300),
    ],
)
def test_explicit_recipe_rejects_unrepresentable_ppm_grid(tmp_path, axis, field, value):
    raw = _raw(tmp_path, "qf")
    plan = _plan(raw)
    plan[axis][field] = value
    before = raw.data.copy()
    with pytest.raises(ValueError, match="ppm"):
        Experiment(raw).process(replay=plan)
    np.testing.assert_array_equal(raw.data, before)


@pytest.mark.parametrize(
    "override",
    [
        {"apodization": None},
        {"lb": None},
        {"gb": None},
        {"ssb": None},
        {"pow": None},
        {"size": None},
        {"phase": None},
        {"phc0": 0.0},
        {"phc1": 0.0},
        {"reference": REFERENCE},
        {"echo_antiecho": "rance-kay"},
        {"digital_filter": "phase"},
        {"bruker_group_delay": 0.0},
        {"tppi_reference": ((1.0, 2.0), (3.0, 4.0))},
    ],
)
def test_explicit_recipe_rejects_every_individual_override(tmp_path, override):
    raw = _raw(tmp_path, "qf")
    with pytest.raises(ValueError, match="only the recorded processing parameters"):
        Experiment(raw).process(replay=_plan(raw), **override)
