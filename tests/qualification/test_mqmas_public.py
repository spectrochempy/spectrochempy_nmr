"""Optional qualification against the historical PST-5 TopSpin references."""

import os

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment


@pytest.mark.parametrize(
    ("sx", "reference_env", "phase"),
    [
        (2048, "SCP_MQMAS_XFSHEAR_PDATA", {}),
        (4096, "SCP_MQMAS_XFSHEAR_ZF_PDATA", {}),
        (
            4096,
            "SCP_MQMAS_XFSHEAR_PHASE_PDATA",
            {"phc0": -30.0, "phc1": 60.0, "f1_shift": 5.0},
        ),
    ],
)
def test_replay_xfshear_matches_mqmas_51_four_planes(sx, reference_env, phase):
    ser = os.environ.get("SCP_MQMAS_SER")
    pdata = os.environ.get(reference_env)
    if not ser or not pdata:
        pytest.skip("requires MQMAS/51 SER and its rectangular XFSHEAR reference")
    source = scp.nmr.read_topspin(ser, local_only=True)
    assert source.meta.acquisition["digital_filter"]["applied"] is False
    assert source.meta.acquisition["dc_offset"]["applied"] is False
    assert source.fft(dim="x").shape == source.shape
    original = source.data.copy()
    ref = scp.nmr.read_topspin(pdata, local_only=True, proc_axis=True)
    mixed = Experiment(source).prepare_xfshear(
        size=(512, sx), direct_points=2048, correct_digital_filter=False
    )
    mixed_data = mixed.data.copy()
    automatic = Experiment(source).prepare_xfshear(size=(512, sx), direct_points=2048)
    corrected_spectrum = Experiment(automatic).replay_xfshear(spin=2.5, phc0=-58)
    manual_spectrum = Experiment(mixed).replay_xfshear(spin=2.5, phc0=-58, phc1=-27360)
    np.testing.assert_allclose(corrected_spectrum.data, manual_spectrum.data, atol=1e-8)
    if phase:
        preview = mixed[0, 0, :].squeeze().pk(phc0=-30.0, phc1=60.0, pivot=0)
        angle = np.deg2rad(-30.0 + 60.0 * np.arange(sx) / sx)
        np.testing.assert_allclose(preview.data, mixed_data[0, 0] * np.exp(1j * angle))
    result = Experiment(mixed).replay_xfshear(spin=2.5, **phase)
    np.testing.assert_array_equal(mixed.data, mixed_data)
    np.testing.assert_array_equal(source.data, original)
    np.testing.assert_allclose(result.x.data, ref.x.data, rtol=0, atol=1e-7)
    np.testing.assert_allclose(result.y.data, ref.y.data, rtol=0, atol=1e-7)

    # One amplitude factor for all four planes.
    rr = result.data[0].real
    gain = np.vdot(rr, ref.data[0].real) / np.vdot(rr, rr)
    assert gain == pytest.approx(7.5, rel=2e-6)
    for plane, channel, part, tolerance in (
        ("RR", 0, "real", 1e-3),
        ("RI", 1, "real", 2e-6),
        ("IR", 0, "imag", 2e-6),
        ("II", 1, "imag", 2e-6),
    ):
        expected = getattr(ref.data[channel], part)
        actual = getattr(result.data[channel], part) * gain
        assert (
            np.linalg.norm(actual - expected) / np.linalg.norm(expected) < tolerance
        ), plane

    corrected = scp.nmr.read_topspin(ser, local_only=True, remove_dc_offset=True)
    with pytest.raises(ValueError, match="uncorrected"):
        Experiment(corrected).prepare_xfshear(size=(512, 2048))


def test_echo_window_follows_each_f1_increment():
    ser = os.environ.get("SCP_MQMAS_SER")
    if not ser:
        pytest.skip("requires the PST-5 SER")
    raw = scp.nmr.read(ser, local_only=True)
    experiment = Experiment(raw)
    windowed = experiment.prepare_xfshear(
        size=(512, 4096),
        direct_points=2048,
        correct_digital_filter=False,
        echo_window=(19 / 12, 0.0008),
    )
    recovered = np.fft.fft(np.fft.ifftshift(windowed.data, axes=-1), axis=-1) / 4096
    # Sample each States pair at the predicted echo and at the first-row center.
    for row, center in [(0, 76), (16, 163), (32, 250)]:
        assert np.abs(recovered[:, row, center]) == pytest.approx(
            np.abs(raw.data[:, row, center]), rel=1e-4
        )
    assert np.all(np.abs(recovered[:, 32, 76]) < np.abs(raw.data[:, 32, 76]) * 0.001)


@pytest.mark.parametrize("echo_window", [None, (19 / 12, 0.0008)])
def test_xf2_without_decimation_has_zero_group_delay(echo_window):
    ser = os.environ.get("SCP_MQMAS_SER")
    if not ser:
        pytest.skip("requires the PST-5 SER")
    raw = scp.nmr.read(ser, local_only=True)
    raw.meta.readonly = False
    acquisition = dict(raw.meta.acquisition)
    acquisition["digital_filter"] = {"applied": False}
    raw.meta.acquisition = acquisition
    source = dict(raw.meta.source_metadata)
    source["acqus"] = {**source["acqus"], "DECIM": 1, "GRPDLY": 0.0}
    raw.meta.source_metadata = source

    experiment = Experiment(raw)
    xf2 = experiment.prepare_xfshear(size=(512, 2048), echo_window=echo_window)
    plan = xf2.meta.mqmas_xf2
    assert plan["digital_filter"]["applied"] is False
    if echo_window is None:
        expected = experiment.prepare_xfshear(
            size=(512, 2048), correct_digital_filter=False
        )
        np.testing.assert_array_equal(xf2.data, expected.data)
    else:
        assert plan["echo_window"]["group_delay_points"] == 0.0
