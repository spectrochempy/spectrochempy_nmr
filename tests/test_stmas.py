"""Public STMAS replay of a synthetic line with known F1 phase evolution."""

import numpy as np
import pytest
import spectrochempy as scp
from nmrglue import bruker


def test_stmas_removes_f2_dependent_f1_evolution(tmp_path):
    """Keep complete States pairs and shear a known off-center frequency bin."""
    (tmp_path / "pdata/1").mkdir(parents=True)
    nx, ny, width, dwell = 16, 16, 1600, 1e-4
    carrier, sf = 130.0, 129.999
    for name, parameters in {
        "acqus": {
            "PULPROG": "dqf-stmas-old.av",
            "PARMODE": 1,
            "AQ_mod": 1,
            "TD": 2 * nx,
            "IN": [dwell, 1.0],
            "SW_h": width,
            "SFO1": carrier,
            "NUC1": "87Rb",
            "BYTORDA": 0,
            "DTYPA": 2,
        },
        "acqu2s": {"TD": 2 * ny + 1, "FnMODE": 4},
        "pdata/1/procs": {"SF": sf},
        "pdata/1/proc2s": {"TDeff": 2 * ny},
    }.items():
        bruker.write_jcamp(
            {"_coreheader": [], "_comments": [], **parameters}, tmp_path / name
        )

    kx, ky = 3, -2
    f2_hz = kx * width / nx
    indirect_phase = -2 * np.pi * (ky / ny - 8 / 9 * f2_hz * dwell) * np.arange(ny)
    direct = np.exp(-2j * np.pi * kx / nx * np.arange(nx))
    pairs = np.stack((np.cos(indirect_phase), np.sin(indirect_phase)), axis=1)
    traces = pairs.reshape(-1, 1) * direct
    stored = np.zeros((2 * ny + 2, 128), dtype="<f8")
    stored[: 2 * ny, : 2 * nx : 2] = traces.real
    stored[: 2 * ny, 1 : 2 * nx : 2] = traces.imag
    stored[2 * ny, : 2 * nx] = 10
    stored.tofile(tmp_path / "ser")

    with pytest.raises(ValueError, match="complete States pairs"):
        scp.nmr.read_stmas(tmp_path)
    raw = scp.nmr.read_stmas(tmp_path, use_tdeff=True)
    assert raw.shape == (2, ny, nx)
    np.testing.assert_array_equal(
        raw.data, traces.reshape(ny, 2, nx).transpose(1, 0, 2)
    )

    xf2 = scp.nmr.Experiment(raw).xf2(size=nx, correct_digital_filter=False)
    phased = scp.nmr.Experiment(xf2).xf2p()
    with pytest.raises(ValueError, match="Use stmas_shear"):
        scp.nmr.Experiment(phased).xfshear(size=ny, spin=1.5)
    out = scp.nmr.Experiment(phased).stmas_shear(size=ny)
    iy, ix = ny // 2 + ky, nx // 2 + kx
    expected = np.zeros((ny, nx))
    expected[iy, ix] = nx * ny
    np.testing.assert_allclose(out.data, expected, atol=1e-10)
    center = (carrier - sf) * 1e6 / sf
    assert out.x.data[ix] == pytest.approx(center - f2_hz / sf)
    assert out.y.data[iy] == pytest.approx(center - ky / (ny * dwell * sf * 17 / 9))
    np.testing.assert_array_equal(
        raw.data, traces.reshape(ny, 2, nx).transpose(1, 0, 2)
    )

    acqus = tmp_path / "acqus"
    acqus.write_text(acqus.read_text().replace("dqf-stmas-old.av", "mp3qzqf"))
    with pytest.raises(ValueError, match="qualified dqf-stmas-old.av"):
        scp.nmr.read_stmas(tmp_path, use_tdeff=True)
