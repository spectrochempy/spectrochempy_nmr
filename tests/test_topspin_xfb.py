"""Always-on tests for the TopSpin XFB replay path."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.experiment import Experiment
from spectrochempy_nmr.processing.topspin_xfb import _gaussian_baseline
from spectrochempy_nmr.processing.topspin_xfb import _subtract_baseline


@pytest.fixture
def qf_fid():
    data = np.array([[1 + 2j, 3, -1j, 4], [2 - 1j, 0, 1j, -2]])
    source = scp.NDDataset(data)
    source.set_coordset(
        y=scp.Coord(np.arange(2) / 1000, units="s"),
        x=scp.Coord(np.arange(4) / 2000, units="s"),
    )
    status = {
        "ME_mod": 0,
        "SI": 4,
        "FTSIZE": 4,
        "STSI": 4,
        "STSR": 0,
        "TDeff": 2,
        "TDoff": 0,
        "FT_mod": 6,
        "PH_mod": 2,
        "REVERSE": False,
        "BC_mod": 0,
        "WDW": 0,
        "FCOR": 1,
        "PHC0": 0,
        "PHC1": 0,
        "OFFSET": 5,
        "SW_p": 1000,
        "SF": 400,
        "PKNL": False,
    }
    source.meta.update(
        {
            "normalization_profile": "bruker-ser-qf-2d-v1",
            "data_domain": "time",
            "interop_physical_dimensions": ("F1", "F2"),
            "acquisition": {"spectral_width_hz": (1000, 2000)},
            "source_metadata": {
                "proc2s": status,
                "procs": {**status, "TDeff": 8, "PH_mod": 0, "SW_p": 2000},
                "acqus": {"GRPDLY": 0},
            },
        }
    )
    return source


@pytest.mark.parametrize("predict", [False, True])
def test_qf_xfb_replaces_only_the_center_bin(qf_fid, predict):
    before = qf_fid.data.copy()
    f2 = np.fft.fftshift(np.fft.ifft(before, axis=1), axes=1) * before.shape[1]
    if predict:
        qf_fid.meta.source_metadata["proc2s"].update(
            ME_mod=2, LPBIN=3, NCOEF=2, TDeff=3
        )
        reflection = 2 * f2[0].conj() * f2[1] / np.sum(np.abs(f2) ** 2, axis=0)
        f2 = np.vstack((f2, reflection * f2[1]))
    expected = np.abs(np.fft.fftshift(np.fft.fft(f2, n=4, axis=0), axes=0))
    expected[2, 2] = (expected[1, 2] + expected[3, 2]) / 2

    result = Experiment(qf_fid).process(replay="topspin")

    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_array_equal(qf_fid.data, before)
    assert result.meta.nmr_processing["scp_processing"]["applied"]["command"] == "xfb"


@pytest.mark.parametrize(("mode", "lb"), [(1, 100), (1, -100), (3, 0)])
@pytest.mark.parametrize("axis", ["proc2s", "procs"])
def test_qf_xfb_windows(qf_fid, mode, lb, axis):
    status = qf_fid.meta.source_metadata
    status[axis].update(WDW=mode, LB=lb, SSB=2)
    status["acqus"]["GRPDLY"] = 1
    f1 = axis == "proc2s"
    points, delay, width = (2, 0, 1000) if f1 else (4, 1, 2000)
    positions = np.arange(points) - delay
    if mode == 1:
        weights = np.exp(-np.pi * lb * positions / width)
    else:
        weights = np.cos(np.pi / 2 * positions / (points - delay + 0.5))
    data = qf_fid.data if f1 else qf_fid.data * weights
    f2 = 4 * np.fft.fftshift(np.fft.ifft(data, axis=1), axes=1)
    if f1:
        f2 *= weights[:, None]
    expected = np.abs(np.fft.fftshift(np.fft.fft(f2, n=4, axis=0), axes=0))
    expected[2, 2] = (expected[1, 2] + expected[3, 2]) / 2

    result = Experiment(qf_fid).process(replay="topspin")

    np.testing.assert_allclose(result.data, expected)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"WDW": 1}, "exponential LB"),
        ({"WDW": 1, "LB": np.inf}, "exponential LB"),
        ({"WDW": 1, "LB": -1e9}, "window overflowed"),
        ({"WDW": 3}, "sine window shift"),
        ({"WDW": 3, "SSB": np.nan}, "sine window shift"),
        ({"WDW": 2}, "supported window"),
    ],
)
def test_qf_xfb_rejects_unqualified_windows(qf_fid, settings, message):
    qf_fid.meta.source_metadata["procs"].update(settings)

    with pytest.raises(ValueError, match=message):
        Experiment(qf_fid).process(replay="topspin")


@pytest.mark.parametrize("baseline", [1, 2])
def test_qf_xfb_f1_tail_baseline(qf_fid, baseline):
    data = np.vstack((qf_fid.data, 2 * qf_fid.data))
    qf_fid.delete_coordset()
    qf_fid.data = data
    qf_fid.set_coordset(
        y=scp.Coord(np.arange(4) / 1000, units="s"),
        x=scp.Coord(np.arange(4) / 2000, units="s"),
    )
    qf_fid.meta.source_metadata["proc2s"].update(TDeff=4, BC_mod=baseline)
    f2 = 4 * np.fft.fftshift(np.fft.ifft(data, axis=1), axes=1)
    tail = f2[-1]
    if baseline == 1:
        tail = (tail.real - tail.imag) * (1 - 1j) / 2
    expected = np.abs(np.fft.fftshift(np.fft.fft(f2 - tail, axis=0), axes=0))
    expected[2, 2] = (expected[1, 2] + expected[3, 2]) / 2

    result = Experiment(qf_fid).process(replay="topspin")

    np.testing.assert_allclose(result.data, expected)
    np.testing.assert_array_equal(qf_fid.data, data)
    assert result.meta.nmr_processing["scp_processing"]["applied"]["baseline"] == (
        baseline,
        0,
    )


@pytest.mark.parametrize(
    ("axis", "settings", "message"),
    [
        ("procs", {"BC_mod": 1}, "not supported on this axis"),
        ("procs", {"BC_mod": 5}, "not supported on this axis"),
        ("proc2s", {"BC_mod": 6, "COROFFS": 0}, "positive finite BCFW"),
        (
            "proc2s",
            {"BC_mod": 6, "COROFFS": 0, "BCFW": 0.2},
            "requires at least eight points",
        ),
        ("proc2s", {"BC_mod": 2}, "requires full quarters"),
        ("procs", {"BC_mod": 4, "COROFFS": 0}, "at least six points"),
    ],
)
def test_qf_xfb_rejects_unqualified_baselines(qf_fid, axis, settings, message):
    qf_fid.meta.source_metadata[axis].update(settings)

    with pytest.raises(ValueError, match=message):
        Experiment(qf_fid).process(replay="topspin")


def test_xfb_shifted_polynomial_baseline_from_endpoint_impulse():
    data = np.zeros(9, dtype=complex)
    data[-1] = 1
    cycles = 1 / 8
    expected_at_first = -(1 - 3 + 5 - 7 + 9 - 121 / 63) / 9
    expected_at_last = 1 - (1 + 3 + 5 + 7 + 9 + 121 / 63) / 9

    result = _subtract_baseline(data, 4, cycles)

    np.testing.assert_allclose(result[[0, -1]], [expected_at_first, expected_at_last])


@pytest.mark.parametrize("radius", [1, 2, 4])
def test_xfb_gaussian_edges(radius):
    data = np.arange(16, dtype=float)
    result = _gaussian_baseline(data, radius)

    # Symmetric convolution preserves the ramp in the interior.
    last = data.size - radius - 2
    np.testing.assert_allclose(result[radius : last + 1], data[radius : last + 1])
    assert result[0] == pytest.approx(radius / 2)
    right_anchor = data.size - 2 * radius if radius > 1 else 0
    assert result[-1] == pytest.approx(
        last + (last - right_anchor) * (data.size - 1 - last) / (2 * radius)
    )


def test_qf_xfb_rejects_missing_f1_center_neighbor(qf_fid):
    qf_fid.meta.source_metadata["proc2s"].update(SI=2, FTSIZE=2, STSI=2)

    with pytest.raises(ValueError, match="neighbors of the F1 center bin"):
        Experiment(qf_fid).process(replay="topspin")


@pytest.mark.parametrize("axis", ["proc2s", "procs"])
def test_qf_xfb_rejects_time_domain_offset(qf_fid, axis):
    qf_fid.meta.source_metadata[axis]["TDoff"] = 2

    with pytest.raises(ValueError, match="TDoff=0"):
        Experiment(qf_fid).process(replay="topspin")


@pytest.mark.parametrize(
    ("axis", "changes", "message"),
    [
        ("procs", {"ME_mod": 2}, "only supported on F1"),
        ("proc2s", {"ME_mod": 4}, "only LPfr"),
        ("proc2s", {"NCOEF": 3}, "even NCOEF"),
        ("proc2s", {"LPBIN": 2}, "valid F1 LPBIN"),
        ("proc2s", {"LPBIN": 5, "TDeff": 5}, "full grid"),
        ("proc2s", {"TDeff": 2}, "full grid"),
    ],
)
def test_qf_xfb_rejects_unqualified_prediction(qf_fid, axis, changes, message):
    qf_fid.meta.source_metadata["proc2s"].update(ME_mod=2, LPBIN=3, NCOEF=2, TDeff=3)
    qf_fid.meta.source_metadata[axis].update(changes)

    with pytest.raises(ValueError, match=message):
        Experiment(qf_fid).process(replay="topspin")
