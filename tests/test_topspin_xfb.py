"""Always-on tests for the TopSpin XFB replay path."""

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.experiment import Experiment


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
