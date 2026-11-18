"""One read retains all harmonics; selection and processing use physical data."""

import sys

import numpy as np
import pytest
import spectrochempy as scp

from test_multiplex_spam import make_spam_ser
from test_multiplex_spam import spam_ser as _spam_ser


@pytest.fixture
def spam_ser(tmp_path):
    return _spam_ser.__wrapped__(tmp_path)


def _read(path, **kwargs):
    return scp.nmr.read(path, phase_cycle="paired", allow_partial=True, **kwargs)


@pytest.mark.parametrize("target", ["directory", "ser", "root"])
@pytest.mark.parametrize("order", [3, 5])
def test_common_read_retains_phases_then_selects(spam_ser, target, order):
    path, a, b = spam_ser
    source = (path / "ser").read_bytes()
    # Labels are provenance: neither read nor processing dispatch on PULPROG.
    acqus = path / "acqus"
    acqus.write_text(acqus.read_text().replace("MQMultiplex52", "custom_cycle"))
    if target == "root":
        experiment = path / "17"
        experiment.mkdir()
        for item in ("acqus", "acqu2s", "acqu3s", "ser", "pdata"):
            (path / item).rename(experiment / item)
        raw = _read(path, expno=17, procno=1)
        path = experiment
    else:
        raw = _read(path / "ser" if target == "ser" else path, procno=1)
    before = raw.copy()
    assert raw.shape == (22, 2, 2)
    assert raw.meta.quadrature_mode == "paired-phase-cycle-2d"
    assert scp.nmr.Experiment(raw).encoding == ("PHASE-CYCLE", "DQD")
    assert "multiplex_spam" not in raw.meta.acquisition
    selected = scp.nmr.Experiment(raw).select_coherence(order)
    expected = np.stack((b - a, 1j * (a + b))) if order == 3 else np.stack((-b, 1j * b))
    np.testing.assert_allclose(selected.data, expected)
    np.testing.assert_array_equal(raw.data, before.data)
    assert raw.meta == before.meta
    assert (path / "ser").read_bytes() == source
    np.testing.assert_array_equal(selected.y.data, raw.y.data)
    np.testing.assert_array_equal(selected.x.data, raw.x.data)
    assert selected.meta.acquisition["multiplex_spam"]["coherence_order"] == order
    assert selected.meta.applied_operations[-1]["stage"] == "processing"


@pytest.mark.parametrize("order", [3, 5])
def test_saved_phase_cycle_and_selection_need_no_provider_or_source(
    spam_ser, tmp_path, monkeypatch, order
):
    path, _, _ = spam_ser
    raw = _read(path)
    expected = scp.nmr.read_multiplex_spam(
        path, coherence_order=order, allow_partial=True
    )
    raw.meta.readonly = False
    for key in ("source_metadata", "provenance", "sw_h", "sfo1", "sf", "nuc1"):
        raw.meta[key] = None
    raw.origin = "neutral"
    raw.filename = None
    raw.write(tmp_path / "cycle.scp", confirm=False)
    monkeypatch.setitem(sys.modules, "nmrglue", None)
    restored = scp.nmr.read(tmp_path / "cycle.scp")
    actual = scp.nmr.Experiment(restored).select_coherence(order)
    np.testing.assert_array_equal(actual.data, expected.data)
    actual.write(tmp_path / "selected.scp", confirm=False)
    actual = scp.nmr.read(tmp_path / "selected.scp")
    spectra = []
    for ds in (actual, expected):
        mixed = scp.nmr.Experiment(ds).xf2(size=8)
        phased = scp.nmr.Experiment(mixed).xf2p(phc0=23, phc1=-41, pivot=2)
        spectra.append(scp.nmr.Experiment(phased).xfshear(size=4))
    for attr in ("data", "x", "y"):
        left, right = (getattr(ds, attr) for ds in spectra)
        np.testing.assert_array_equal(
            left if attr == "data" else left.data,
            right if attr == "data" else right.data,
        )
    assert spectra[0].meta.acquisition["multiplex_spam"]["coherence_order"] == order


@pytest.mark.parametrize("operation", ["process", "fft", "xf2"])
def test_select_before_processing(spam_ser, operation):
    raw = _read(spam_ser[0])
    before = raw.copy()
    with pytest.raises(
        (ValueError, NotImplementedError),
        match="select_coherence|uncorrected, normalized States",
    ):
        if operation == "process":
            scp.nmr.Experiment(raw).process()
        elif operation == "xf2":
            scp.nmr.Experiment(raw).xf2(size=8)
        else:
            raw.fft(dim="x", inplace=True)
    np.testing.assert_array_equal(raw.data, before.data)
    assert raw.meta == before.meta


@pytest.mark.parametrize("order", [True, 1, 2, 7, 3.0])
def test_selection_rejects_unqualified_or_aliased_orders(spam_ser, order):
    with pytest.raises(ValueError, match="Unsupported MQ coherence"):
        scp.nmr.Experiment(_read(spam_ser[0])).select_coherence(order)


@pytest.mark.parametrize(
    "damage", ["phase", "pair", "grid", "mask", "nan", "corrected", "shape", "sampling"]
)
def test_selection_validates_physical_contract(spam_ser, damage):
    raw = _read(spam_ser[0])
    raw.meta.readonly = False
    acq = raw.meta.acquisition
    if damage == "phase":
        acq["phase_cycle"]["excitation_phases_degrees"] = [0] * 11
    elif damage == "pair":
        acq["phase_cycle"]["spam_pair_order"] = [180, 0]
    elif damage == "grid":
        raw.y.data = [0, 0.2]
    elif damage == "mask":
        mask = np.zeros(raw.shape, dtype=bool)
        mask[0, 0, 0] = True
        raw.mask = mask
    elif damage == "nan":
        raw.data[0, 0, 0] = np.nan
    elif damage == "corrected":
        acq["digital_filter"]["applied"] = True
    elif damage == "shape":
        acq["acquired_shape"] = [3, 2]
    else:
        acq["sampling"] = ["irregular", "regular"]
    with pytest.raises(ValueError, match="select_coherence"):
        scp.nmr.Experiment(raw).select_coherence()


def test_read_selection_and_calibration_are_explicit(spam_ser):
    path, _, _ = spam_ser
    from spectrochempy.utils.exceptions import SpectroChemPyError

    with pytest.raises(SpectroChemPyError, match="Unexpected extra SER data"):
        scp.nmr.read(path, phase_cycle="paired")
    raw = _read(path, indirect_dwell=0.0003)
    np.testing.assert_allclose(raw.y.data, [0, 0.0003])
    assert raw.meta.acquisition["spectral_width_hz"][0] == 1 / 0.0003
    for kwargs in (
        {"quadrature": "states"},
        {"processed": True},
        {"use_tdeff": True},
        {"remove_dc_offset": True},
        {"remove_digital_filter": True},
    ):
        with pytest.raises(ValueError, match="Read paired phase cycles"):
            _read(path, **kwargs)


@pytest.mark.parametrize("count", [7, 11])
def test_projection_against_scalar_oracle(tmp_path, count):
    path, _, _ = make_spam_ser(tmp_path, count)
    raw = _read(path)
    order = 3 if count == 7 else 5
    expected = np.zeros((2, *raw.shape[1:]), dtype=complex)
    # Explicit scalar MSM equations independently exercise channel indexing.
    for increment in range(raw.shape[1]):
        for point in range(raw.shape[2]):
            for j in range(count):
                first, second = raw.data[2 * j : 2 * j + 2, increment, point]
                angle = 2 * np.pi * order * j / count
                expected[0, increment, point] += (
                    (first - second) * np.cos(angle)
                    - 1j * (first + second) * np.sin(angle)
                ) / count
                expected[1, increment, point] += (
                    (first - second) * np.sin(angle)
                    + 1j * (first + second) * np.cos(angle)
                ) / count
    actual = scp.nmr.Experiment(raw).select_coherence(order)
    np.testing.assert_allclose(actual.data, expected, atol=1e-13)
    if count == 7:
        with pytest.raises(ValueError, match="Unsupported MQ coherence"):
            scp.nmr.Experiment(raw).select_coherence(5)
    actual.meta.acquisition["phase_cycle"]["phase_count"] = 1
    assert raw.meta.acquisition["phase_cycle"]["phase_count"] == count


@pytest.mark.parametrize("tail", ["zero", "partial", "hole", "complete"])
def test_tail_policy_is_read_side(spam_ser, tail):
    from spectrochempy.utils.exceptions import SpectroChemPyError

    path, _, _ = spam_ser
    words = np.fromfile(path / "ser", dtype=">f8").reshape(-1, 128)
    extra = np.zeros((22, 128), dtype=">f8")
    if tail == "partial":
        extra[:3, :4] = 1
    elif tail == "hole":
        extra[[0, 2], :4] = 1
    elif tail == "complete":
        extra[:, :4] = 1
    np.concatenate((words[:44], extra)).astype(">f8").tofile(path / "ser")
    if tail in ("hole", "complete"):
        with pytest.raises(SpectroChemPyError, match="Unexpected extra SER data"):
            _read(path)
    else:
        if tail == "partial":
            with pytest.raises(SpectroChemPyError, match="Unexpected extra SER data"):
                scp.nmr.read(path, phase_cycle="paired")
        raw = _read(path)
        cycle = raw.meta.acquisition["phase_cycle"]
        assert cycle["ignored_traces"] == 22
        assert cycle["ignored_nonzero_traces"] == (3 if tail == "partial" else 0)
        assert raw.shape == (22, 2, 2)


@pytest.mark.parametrize("delay", [None, -1, 0])
@pytest.mark.parametrize("dedicated", [False, True])
def test_no_decimation_preserves_zero_filter_delay(spam_ser, delay, dedicated):
    path, _, _ = spam_ser
    acqus = path / "acqus"
    text = acqus.read_text().replace(
        "##$GRPDLY= 2.0\n",
        "" if delay is None else f"##$GRPDLY= {delay}\n",
    )
    acqus.write_text("##$DECIM= 1\n" + text)
    if dedicated:
        states = scp.nmr.read_multiplex_spam(path, allow_partial=True)
    else:
        states = scp.nmr.Experiment(_read(path)).select_coherence()
    correction = states.meta.acquisition["digital_filter"]
    assert correction["group_delay_points"] == 0.0
    assert correction["delay_source"] == "no decimation"
    states.meta.readonly = False
    states.meta.source_metadata = {}
    corrected = scp.nmr.Experiment(states).xf2(size=8)
    uncorrected = scp.nmr.Experiment(states).xf2(size=8, correct_digital_filter=False)
    np.testing.assert_array_equal(corrected.data, uncorrected.data)
