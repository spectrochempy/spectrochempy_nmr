"""Parameterized pulse programs, units, diagnostics and simulator parity."""

from pathlib import Path

import numpy as np
import pytest
from spectrochempy.nmr import pulsar


PROGRAMS = Path(__file__).resolve().parents[1] / "examples/pulse_programs"
HAHN = (PROGRAMS / "hahn.seq").read_text()


def test_hahn_file_matches_manual_complex_echo_for_four_increments(tmp_path):
    path = tmp_path / "hahn.seq"
    path.write_text(HAHN)
    program = pulsar.read_sequence(path)
    parameters = {
        "rf1": "10 kHz",
        "p1": "8 us",
        "p2": "17 us",
        "ph1": 0,
        "ph2": 90,
        "l1": 1,
    }
    settings = dict(
        spin=2.5,
        larmor_hz=104.228e6,
        cq_hz=2.5e6,
        eta=0.3,
        spinning_hz=10000,
        transitions="central",
        points=1024,
        spectral_width_hz=10000,
        accuracy=2,
        t2_s=1e-3,
    )
    for count in (1, 6, 11, 16):
        d1, d2 = count * 100e-6 - 12.5e-6, count * 100e-6 - 8.5e-6
        events = program.build({**parameters, "d1": f"{d1} s", "d2": f"{d2} s"})
        manual = (
            pulsar.Pulse(8e-6, 10000),
            pulsar.Coherence(-1),
            pulsar.Delay(d1),
            pulsar.Pulse(17e-6, 10000, phase_deg=90),
            pulsar.Delay(d2),
        )
        actual = pulsar.simulate_pulse_sequence(events, **settings)
        expected = pulsar.simulate_pulse_sequence(manual, **settings)
        assert np.max(np.abs(expected.data)) > 1e-5
        np.testing.assert_allclose(actual.data, expected.data, atol=1e-12, rtol=1e-10)


def test_expressions_nested_loops_and_fresh_build_state():
    program = pulsar.parse_sequence(
        """
set rf=rf1
set p1=1/rf/4
set p2=p1*2
set n=0
repeat l1
    repeat 2
        pulse p1 phase=n*90
        set n=n+1
    end
    pulse p2 rf=2*rf
    pulse p1
end
"""
    )
    events = program.build({"rf1": "10 kHz", "l1": 2})
    assert len(events) == 8
    assert [e.phase_deg for e in events] == [0, 90, 0, 0, 180, 270, 0, 0]
    assert [e.rf_hz for e in events] == [10000, 10000, 20000, 10000] * 2
    assert events[0].duration_s == pytest.approx(25e-6)
    assert events[2].duration_s == pytest.approx(50e-6)
    assert program.build({"rf1": "10 kHz", "l1": 2}) == events
    faster = program.build({"rf1": "20 kHz", "l1": 1})
    assert faster[0].duration_s == pytest.approx(12.5e-6)


def test_mqmas_zfilter_rf_changes_match_manual_complex_spectrum():
    program = pulsar.read_sequence(PROGRAMS / "mqmas_zfilter.seq")
    parameters = {
        "rf_exc": "100 kHz",
        "rf_conv": "80 kHz",
        "rf_read": "10 kHz",
        "p1": "3 us",
        "p2": "2 us",
        "p3": "8 us",
        "dz": "50 us",
    }
    parameters.update({"q": 3, "d0": "12.5 us", "ph1": 0, "ph2": 90, "ph3": 0})
    manual = (
        pulsar.Pulse(3e-6, 100e3),
        pulsar.Coherence(3),
        pulsar.Delay(12.5e-6),
        pulsar.Pulse(2e-6, 80e3, 90),
        pulsar.Coherence(0),
        pulsar.Delay(50e-6),
        pulsar.Pulse(8e-6, 10e3),
        pulsar.Coherence(1),
    )
    settings = dict(
        spin=2.5,
        larmor_hz=104e6,
        cq_hz=2.5e6,
        eta=0.3,
        spinning_hz=10000,
        spectral_width_hz=20000,
        points=1024,
        accuracy=2,
        transitions="central",
    )
    actual = pulsar.simulate_pulse_sequence(program.build(parameters), **settings)
    expected = pulsar.simulate_pulse_sequence(manual, **settings)
    assert np.max(np.abs(expected.data)) > 1e-10
    np.testing.assert_allclose(actual.data, expected.data, atol=1e-12, rtol=1e-10)


def test_simultaneous_two_channel_rf_matches_coupled_density():
    program = pulsar.read_sequence(PROGRAMS / "two_channel.seq")
    parameters = {
        "rf1": "5 kHz",
        "rf2": "7 kHz",
        "p1": "25 us",
        "p2": "12 us",
        "p3": "8 us",
        "ph1": 30,
        "ph2": 70,
        "o1": "1 kHz",
        "o2": "-800 Hz",
    }
    manual = (
        pulsar.Pulse(25e-6, 5000, 30, 1000, 7000, 70, -800),
        pulsar.Pulse(12e-6, 0, partner_rf_hz=7000),
        pulsar.Pulse(8e-6, 5000),
    )
    settings = dict(
        observed=pulsar.SpinSite(0.5, 100e6),
        partner=pulsar.SpinSite(0.5, 25e6),
        j_hz=120,
        spectral_width_hz=10000,
        points=1024,
        accuracy=2,
        output="density",
    )
    actual = pulsar.simulate_coupled_sequence(program.build(parameters), **settings)
    expected = pulsar.simulate_coupled_sequence(manual, **settings)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
    # The same common interval is a simultaneous Pulse, not two sequential pulses.
    assert len(program.build(parameters)) == 3


@pytest.mark.parametrize(
    ("text", "parameters", "line", "message"),
    [
        ("delay d1", {}, 1, "missing parameter 'd1'"),
        ("# comment\ndelay d1", {"d1": "1 kHz"}, 2, "Cannot convert"),
        ("delay d1", {"d1": 1e-6}, 1, "explicit units"),
        ("delay -1 us", {}, 1, "nonnegative"),
        ("pulse 0 us rf=1 kHz", {}, 1, "positive"),
        ("pulse 1 us", {}, 1, "requires rf"),
        ("set rf=-1 kHz\npulse 1 us", {}, 1, "nonnegative"),
        ("pulse 1 us rf=1 us", {}, 1, "Cannot convert"),
        ("pulse 1 us rf=1 kHz phase=1 kHz", {}, 1, "Cannot convert"),
        ("repeat 1.5\nend", {}, 1, "integer"),
        ("repeat -1\nend", {}, 1, "nonnegative"),
        ("set p1=1/0\ndelay p1", {}, 1, "division by zero"),
        ("coherence 1.5", {}, 1, "integer"),
        ("delay 1e309 us", {}, 1, "finite"),
        ("set rf=missing", {}, 1, "missing parameter"),
    ],
)
def test_build_errors_report_source_and_line(text, parameters, line, message):
    program = pulsar.parse_sequence(text, source="test.seq")
    with pytest.raises(pulsar.SequenceError) as caught:
        program.build(parameters)
    assert caught.value.source == "test.seq"
    assert caught.value.line == line
    assert message in str(caught.value)
    assert str(caught.value).startswith(f"test.seq:{line}:")


@pytest.mark.parametrize(
    "text",
    [
        "end",
        "repeat n\ndelay d",
        "unknown p1",
        "set us=1",
        "pulse p1 typo=rf",
        "pulse p1 rf=a rf=b",
        "delay __import__('os')",
        "delay (1).__class__",
        "delay [1][0]",
        "delay 2**4",
        "delay True",
        "delay '1 us'",
    ],
)
def test_invalid_syntax_is_rejected_before_binding(text):
    with pytest.raises(pulsar.SequenceError):
        pulsar.parse_sequence(text)


def test_loop_failure_keeps_inner_line_and_expansion_is_bounded():
    program = pulsar.parse_sequence("repeat n\n delay -1 us\nend")
    with pytest.raises(pulsar.SequenceError, match=r"<sequence>:2:"):
        program.build({"n": 1})
    with pytest.raises(pulsar.SequenceError, match="max_steps"):
        pulsar.parse_sequence("repeat 1000\n delay 1 us\nend").build(max_steps=20)
    with pytest.raises(pulsar.SequenceError, match="max_steps"):
        pulsar.parse_sequence("repeat 20\n repeat 20\n end\nend").build(max_steps=50)
    assert pulsar.parse_sequence("repeat 0\n delay missing\nend").build() == ()


def test_external_quantities_and_per_pulse_partner_override():
    from spectrochempy.core.units import ur

    program = pulsar.parse_sequence(
        "set rf=rf1\npulse p1 partner_rf=rf2 partner_phase=phi\npulse p1"
    )
    parameters = {
        "rf1": 10 * ur.kHz,
        "rf2": 5 * ur.kHz,
        "p1": 8 * ur.us,
        "phi": np.pi / 2 * ur.rad,
    }
    first, second = program.build(parameters)
    assert first.partner_rf_hz == 5000
    assert first.partner_phase_deg == pytest.approx(90)
    assert second.partner_rf_hz == 0
    assert parameters["p1"] == 8 * ur.us


@pytest.mark.parametrize("coefficients", [(0.5, 0.5), (-0.5j, 0.5j)])
def test_true_2d_states_increments_match_full_t1_propagation(coefficients):
    """Both complex States branches retain simulated t1 and F2 dimensions."""
    from spectrochempy.core.units import ur

    program = pulsar.parse_sequence(
        """
set rf=rf_exc
pulse p1
coherence q
delay d0
set rf=rf_conv
pulse p2 phase=ph2
coherence 1
"""
    )
    parameters = {
        "rf_exc": "110 kHz",
        "rf_conv": "85 kHz",
        "p1": "12 us",
        "p2": "4 us",
        "ph2": 15,
    }
    times = np.arange(48) * 10e-6
    settings = dict(
        spin=2.5,
        larmor_hz=104.46851646e6,
        cq_hz=2.923e6,
        eta=0.136,
        spinning_hz=12500,
        sidebands=2,
        transitions="central",
        spectral_width_hz=120000,
        points=256,
        accuracy=1,
        receiver_phase_deg=37,
        t2_s=60e-6,
    )
    preparation = pulsar.PhaseCycle(
        (
            (pulsar.Pulse(12e-6, 110e3), pulsar.Coherence(3)),
            (pulsar.Pulse(12e-6, 110e3), pulsar.Coherence(-3)),
        ),
        coefficients,
    )
    reference = pulsar.simulate_t1_series(
        preparation,
        times,
        (pulsar.Pulse(4e-6, 85e3, 15), pulsar.Coherence(1)),
        **settings,
    )
    rows = []
    for time in times:
        scans = tuple(
            program.build({**parameters, "q": q, "d0": time * ur.s}) for q in (3, -3)
        )
        rows.append(
            pulsar.simulate_pulse_sequence(
                pulsar.PhaseCycle(scans, coefficients), **settings
            ).data
        )
    data = np.stack(rows)
    assert data.shape == (48, 256)
    assert np.iscomplexobj(data)
    assert np.max(np.abs(data)) > 1e-10
    assert not np.allclose(data[0], data[-1], atol=1e-10)
    np.testing.assert_allclose(data, reference.data, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize(
    "value", [True, np.nan, np.inf, np.array([1, 2]), "not a quantity"]
)
def test_bad_external_values_still_report_the_usage_line(value):
    program = pulsar.parse_sequence("# header\ndelay d1", source="bad.seq")
    with pytest.raises(pulsar.SequenceError) as caught:
        program.build({"d1": value})
    assert caught.value.line == 2
    assert caught.value.source == "bad.seq"


def test_tabs_are_whitespace_and_failed_build_does_not_leak_state():
    program = pulsar.parse_sequence("set\trf=rf1\nrepeat\tn\n\tpulse\tp1\nend")
    with pytest.raises(pulsar.SequenceError):
        program.build({"rf1": "10 kHz", "n": 1})
    events = program.build({"rf1": "20 kHz", "n": 1, "p1": "5 us"})
    assert len(events) == 1
    assert events[0].duration_s == pytest.approx(5e-6)
    assert events[0].rf_hz == 20000
