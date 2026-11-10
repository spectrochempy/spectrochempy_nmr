"""Opt-in qualification on a local TopSpin installation."""

import os
from pathlib import Path
import shutil
import subprocess

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment


@pytest.fixture
def examdata():
    root = os.environ.get("NMR_TOPSPIN_EXAMDATA")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_EXAMDATA to the TopSpin examdata directory")
    return Path(root).resolve(strict=True)


def _local_read(path, *, filtered=False):
    return scp.read_topspin(
        path, local_only=True, remove_digital_filter=filtered, remove_dc_offset=False
    )


@pytest.mark.parametrize(
    ("experiment", "mode", "size", "nuclei"),
    [
        ("Menthyl-Anthranilate/20", 1, (128, 1024), ("1H", "1H")),
        ("Menthyl-Anthranilate/30", 5, (128, 1024), ("1H", "1H")),
        ("Menthyl-Anthranilate/40", 6, (128, 1024), ("13C", "1H")),
        ("Menthyl-Anthranilate/60", 6, (128, 2048), ("13C", "1H")),
        ("Menthyl-Anthranilate/71", 1, (256, 1024), ("15N", "1H")),
        ("Cyclosporine/206", 5, (256, 1024), ("1H", "1H")),
        ("Cyclosporine/209", 5, (256, 1024), ("1H", "1H")),
    ],
)
def test_uniform_ser(examdata, monkeypatch, tmp_path, experiment, mode, size, nuclei):
    path = examdata / experiment
    monkeypatch.chdir(examdata)
    fid = _local_read(path / "ser")
    assert fid.shape == (size if mode == 1 else (2, *size))
    for target in (Path(experiment) / "ser", path, Path(experiment)):
        np.testing.assert_array_equal(_local_read(target).data, fid.data)

    records = ng.bruker.read_acqus_file(path)
    x, y = records["acqus"], records["acqu2s"]
    assert (y["FnMODE"], x["FnTYPE"], x["AQ_mod"]) == (mode, 0, 3)
    assert (y["NUC1"], x["NUC1"]) == nuclei
    assert x["BYTORDA"] == 0 and x["DTYPA"] in (0, 2)
    words = np.fromfile(path / "ser", dtype="<f8" if x["DTYPA"] == 2 else "<i4")
    words = words.reshape(y["TD"], x["TD"])
    traces = words[:, ::2] + 1j * words[:, 1::2]
    divisor = x["NS"] * x["RG"]
    expected = traces if mode == 1 else np.stack((traces[::2], traces[1::2]))
    np.testing.assert_allclose(fid.data, expected / divisor, rtol=2e-14, atol=1e-12)
    _check_filter(path, fid, x["GRPDLY"])
    _check_historical_phase(path, fid, monkeypatch)

    options = {"echo_antiecho": "bruker-rance-kay"} if mode == 6 else {}
    spectrum = Experiment(fid).process(**options)
    assert spectrum.shape == (size if mode == 1 else (4, *size))
    if mode != 1:
        assert tuple(spectrum.c.labels) == ("rr", "ri", "ir", "ii")
    widths = (y["SW"] * y["SFO1"], x["SW_h"])
    for axis, width, count in zip((spectrum.y, spectrum.x), widths, size, strict=True):
        assert axis.units == scp.ur.Hz
        np.testing.assert_allclose(
            axis.data, (count // 2 - np.arange(count)) * width / count, atol=1e-8
        )

    tools = [shutil.which(name) for name in ("bruk2pipe", "nmrPipe")]
    if all(tools):
        _compare_nmrpipe(tmp_path, tools, words, spectrum, x, y, widths, mode, divisor)


def _check_historical_phase(path, fid, monkeypatch):
    reader = "spectrochempy_nmr.readers.read_topspin._read_normalized_ser"
    with monkeypatch.context() as legacy:
        legacy.setattr(reader, lambda *args, **kwargs: None)
        historical = _local_read(path / "ser")
    for dim in ("x", "y"):
        fid, historical = fid.fft(dim=dim), historical.fft(dim=dim)
        np.testing.assert_allclose(fid.data, historical.data, rtol=2e-12, atol=1e-7)
        axis, expected = fid.coordset[dim], historical.coordset[dim]
        assert axis.units == expected.units == scp.ur.ppm
        np.testing.assert_allclose(axis.data, expected.data, rtol=0, atol=1e-3)


def _check_filter(path, fid, delay):
    filtered = _local_read(path, filtered=True)
    assert delay == int(delay)
    delay = int(delay)
    skip = delay + 2
    fold = max(skip - 6, 0)
    shifted = np.roll(fid.data, -delay, axis=-1)
    if fold:
        shifted[..., :fold] += shifted[..., : -fold - 1 : -1]
    np.testing.assert_allclose(
        filtered.data, shifted[..., :-skip], rtol=1e-11, atol=3e-9
    )
    correction = filtered.meta.acquisition["digital_filter"]
    assert correction["applied"] and correction["removed_points"] == skip


def _compare_nmrpipe(tmp_path, tools, words, spectrum, x, y, widths, mode, divisor):
    bruk2pipe, nmrpipe = tools
    environment = os.environ.copy()
    environment.setdefault(
        "NMRTXT", str(Path(bruk2pipe).resolve().parents[1] / "nmrtxt")
    )

    def run(binary, *args):
        completed = subprocess.run(
            [binary, *map(str, args)], capture_output=True, env=environment
        )
        assert completed.returncode == 0, completed.stderr.decode(errors="replace")

    # The Bruker converter takes int32 words. Scale float64 originals in
    # a temporary copy so quantization stays below the comparison tolerance.
    scale = 2**23 / np.max(np.abs(words)) if x["DTYPA"] == 2 else 1
    ser = tmp_path / "ser"
    np.rint(words * scale).astype("<i4").tofile(ser)
    raw = tmp_path / "raw.fid"
    swy, swx = widths
    run(
        bruk2pipe,
        "-in",
        ser,
        "-out",
        raw,
        "-ov",
        "-AMX",
        "-noswap",
        "-bad",
        0,
        "-ndim",
        2,
        "-aq2D",
        "Magnitude" if mode == 1 else "States",
        "-xN",
        x["TD"],
        "-xT",
        x["TD"] // 2,
        "-xMODE",
        "DQD",
        "-xSW",
        swx,
        "-xOBS",
        x["SFO1"],
        "-xCAR",
        0,
        "-xLAB",
        x["NUC1"],
        "-yN",
        y["TD"],
        "-yT",
        y["TD"] // (1 if mode == 1 else 2),
        "-yMODE",
        {1: "Real", 5: "States-TPPI", 6: "Echo-AntiEcho"}[mode],
        "-ySW",
        swy,
        "-yOBS",
        y["SFO1"],
        "-yCAR",
        0,
        "-yLAB",
        y["NUC1"],
    )
    stages = (
        ("FT", "-nodf"),
        ("TP",),
        ("FT", {1: "-neg", 5: "-alt", 6: "-nodf"}[mode]),
    )
    for index, args in enumerate(stages):
        target = tmp_path / f"stage-{index}.ft2"
        run(nmrpipe, "-in", raw, "-fn", *args, "-out", target, "-ov")
        raw = target

    header, transformed = ng.pipe.read(str(raw))
    if mode == 1:
        pairs = [(spectrum.data, transformed.T)]
    else:
        first, second = transformed[::2].T, transformed[1::2].T
        pairs = zip(
            spectrum.data,
            (first.real, second.real, first.imag, second.imag),
            strict=True,
        )
    for actual, reference in pairs:
        reference = reference / (divisor * scale)
        assert np.linalg.norm(actual - reference) / np.linalg.norm(reference) < 1e-6

    for dimension, axis in (("F1", spectrum.y), ("F2", spectrum.x)):
        step = header[f"FD{dimension}SW"] / axis.size
        origin = header[f"FD{dimension}ORIG"]
        # NMRPipe's real F1 origin after TP and FT -neg is one bin higher.
        if mode == 1 and dimension == "F1":
            origin -= step
        expected = origin + (axis.size - 1 - np.arange(axis.size)) * step
        np.testing.assert_allclose(axis.data, expected, atol=2e-3, rtol=0)


@pytest.mark.parametrize(("experiment", "rows"), [("42", 98), ("62", 102)])
def test_nus_ser_retains_storage_without_uniform_fft(examdata, experiment, rows):
    path = examdata / "Menthyl-Anthranilate" / experiment
    stored = ng.interop.read_bruker_ser_storage(path)
    assert stored.metadata["acqus"]["FnTYPE"] == 2
    assert stored.metadata["acqu2s"]["TD"] == rows
    assert stored.data.shape[0] == 256
    assert len(stored.metadata["nuslist_raw"].splitlines()) == 128
    with pytest.raises(ValueError, match="uniform"):
        ng.interop.read_bruker_ser_storage(path, echo_antiecho_only=True)
    fid = _local_read(path)
    with pytest.raises(NotImplementedError, match="normalized raw 2D"):
        Experiment(fid).process()


def test_ubiquitin_3d_storage(examdata):
    path = examdata / "Ubiquitin" / "203"
    metadata, data = ng.bruker.read(path, read_pulseprogram=False)
    x, y, z = (metadata[key] for key in ("acqus", "acqu2s", "acqu3s"))
    assert x["PARMODE"] == 2
    assert (z["FnMODE"], y["FnMODE"], x["AQ_mod"]) == (5, 6, 3)
    assert (z["NUC1"], y["NUC1"], x["NUC1"]) == ("13C", "15N", "1H")
    raw = np.memmap(path / "ser", dtype="<f8", mode="r", shape=(160, 84, 1024))
    assert data.shape == (160, 84, 512)
    np.testing.assert_array_equal(data.real, raw[..., ::2])
    np.testing.assert_array_equal(data.imag, raw[..., 1::2])


def test_dqd_fid_filter_manual_phase_and_ppm(examdata):
    path = examdata / "Menthyl-Anthranilate" / "1"
    params = ng.bruker.read_acqus_file(path)["acqus"]
    assert (params["AQ_mod"], params["DTYPA"], params["BYTORDA"]) == (3, 2, 0)
    fid = _local_read(path)
    words = np.fromfile(path / "fid", dtype="<f8")
    expected = (words[::2] - 1j * words[1::2]) / (params["NS"] * params["RG"])
    np.testing.assert_allclose(fid.data, expected, rtol=2e-14, atol=1e-12)
    _check_filter(path, fid, params["GRPDLY"])
    filtered = _local_read(path, filtered=True)
    size = fid.x.size
    phased = Experiment(filtered).process(
        size=size, reference="acquisition", phase="manual", phc0=23
    )
    index = np.arange(size)
    rotation = np.exp(1j * np.deg2rad(23))
    expected_fft = np.fft.fftshift(np.fft.fft(filtered.data, n=size)) * rotation
    np.testing.assert_allclose(phased.data, expected_fft, rtol=1e-11, atol=1e-9)
    assert phased.x.units == scp.ur.ppm
    expected_ppm = (
        params["O1"] + (size // 2 - index) * params["SW_h"] / size
    ) / params["BF1"]
    np.testing.assert_allclose(phased.x.data, expected_ppm, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    ("experiment", "delay", "reference_scale"),
    [
        ("Cyclosporine/1", 76.0, 256),
        ("exam_CMCse_1/1", 67.9842376708984, 0.25),
    ],
)
def test_dqd_topspin_efp_replay(experiment, delay, reference_scale):
    root = os.environ.get("NMR_TOPSPIN_EFP_REFERENCES")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_EFP_REFERENCES to the TopSpin-processed copies")
    path = Path(root).resolve(strict=True) / experiment

    fid = _local_read(path / "fid")
    original = fid.data.copy()
    replay = Experiment(fid).process(replay="topspin")

    pdata = path / "pdata" / "1"
    pars, (real, imag) = ng.bruker.read_pdata(pdata, bin_files=["1r", "1i"])
    reference = real + 1j * imag
    # Fixed positive scales measured for these references. Do not fit a
    # complex scale, which would hide a phase error in the delay correction.
    residual = replay.data * reference_scale - reference
    assert np.linalg.norm(residual) / np.linalg.norm(reference) < 1e-7

    top_axis = scp.read_topspin(pdata / "1r", proc_axis=True, local_only=True).x
    assert replay.x.units == top_axis.units == scp.ur.ppm
    np.testing.assert_allclose(replay.x.data, top_axis.data, atol=2e-12, rtol=0)
    assert pars["acqus"]["GRPDLY"] == pytest.approx(delay)
    assert replay.meta.nmr_processing["scp_processing"]["applied"][
        "group_delay_points"
    ] == pytest.approx(delay)
    assert Experiment(replay).is_frequency_domain
    np.testing.assert_array_equal(fid.data, original)

    with pytest.raises(ValueError, match="remove_digital_filter=False"):
        Experiment(_local_read(path / "fid", filtered=True)).process(replay="topspin")
    with pytest.raises(ValueError, match="recorded processing parameters"):
        Experiment(fid).process(replay="topspin", phase="manual")
    with pytest.raises(ValueError, match="original calibrated time grid"):
        Experiment(fid[1:]).process(replay="topspin")


@pytest.mark.parametrize("experiment", ["20", "30", "40"])
@pytest.mark.parametrize(
    ("variable", "with_prediction"),
    [
        ("NMR_TOPSPIN_XFB_REFERENCES", False),
        ("NMR_TOPSPIN_XFB_LP_REFERENCES", True),
    ],
)
def test_topspin_xfb_replay(experiment, variable, with_prediction):
    root = os.environ.get(variable)
    if root is None:
        pytest.skip(f"set {variable} to the corresponding TopSpin XFB copies")
    path = Path(root).resolve(strict=True) / experiment
    _assert_topspin_xfb(path, with_prediction)


@pytest.mark.parametrize("experiment", ["20", "30", "40"])
@pytest.mark.parametrize(
    ("case", "windows", "combined"),
    [
        ("em-f1", (1, 0), False),
        ("em-f2", (0, 1), False),
        ("sine-f1", (3, 0), False),
        ("sine-f2", (0, 3), False),
        ("em-sine-lp-pknl", (1, 3), True),
        ("sine-em-lp-pknl", (3, 1), True),
    ],
)
def test_topspin_xfb_windows(experiment, case, windows, combined):
    root = os.environ.get("NMR_TOPSPIN_XFB_WINDOW_REFERENCES")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_XFB_WINDOW_REFERENCES to the window copies")
    path = Path(root).resolve(strict=True) / case / experiment
    trace = _assert_topspin_xfb(path, combined)
    assert trace["window"] == windows
    assert trace["pknl"] is combined


@pytest.mark.parametrize("experiment", ["20", "30", "40"])
@pytest.mark.parametrize(
    ("folder", "case", "modes"),
    [
        ("bc-modes", "bc1-f1", (1, 0)),
        ("bc-modes", "bc2-f1", (2, 0)),
        ("bc-modes", "bc3-f1", (3, 0)),
        ("bc-modes", "bc4-f1", (4, 0)),
        ("bc-modes", "bc1-f2", (0, 0)),
        ("bc-modes", "bc2-f2", (0, 0)),
        ("bc-modes", "bc3-f2", (0, 4)),
        ("bc-modes", "bc4-f2", (0, 4)),
        ("bc-offset", "bc4-f1-offset", (4, 0)),
        ("bc-offset", "bc4-f2-offset", (0, 4)),
        ("bc-combined", "bc1-combined", (1, 4)),
        ("bc-combined", "bc2-combined", (2, 4)),
        ("bc-combined", "bc3-combined", (3, 4)),
        ("bc-combined", "bc4-combined", (4, 4)),
    ],
)
def test_topspin_xfb_baseline(experiment, folder, case, modes):
    root = os.environ.get("NMR_TOPSPIN_XFB_BASELINE_REFERENCES")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_XFB_BASELINE_REFERENCES to the baseline copies")
    path = Path(root).resolve(strict=True) / folder / case / experiment
    combined = folder == "bc-combined"
    trace = _assert_topspin_xfb(path, combined)
    assert trace.get("baseline", (0, 0)) == modes
    assert trace["pknl"] is combined
    if combined:
        assert trace["window"] == (4, 1)


@pytest.mark.parametrize("experiment", ["20", "30", "40"])
@pytest.mark.parametrize(
    ("folder", "case", "modes"),
    [
        ("bc-modes", "bc5-f1", (5, 0)),
        ("bc-modes", "bc6-f1", (6, 0)),
        ("bc-modes", "bc5-f2", (0, 6)),
        ("bc-modes", "bc6-f2", (0, 6)),
        ("bc-gaussian-combined", "sfil-combined", (5, 6)),
        ("bc-gaussian-combined", "qfil-combined", (6, 6)),
        ("bc-gaussian-combined", "qfil-radius-one", (6, 6)),
    ],
)
def test_topspin_xfb_gaussian_baseline(experiment, folder, case, modes):
    root = os.environ.get("NMR_TOPSPIN_XFB_GAUSSIAN_REFERENCES")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_XFB_GAUSSIAN_REFERENCES to the Gaussian copies")
    path = Path(root).resolve(strict=True) / folder / case / experiment
    combined = folder == "bc-gaussian-combined"
    trace = _assert_topspin_xfb(path, combined)
    assert trace["baseline"] == modes
    assert trace["pknl"] is combined


@pytest.mark.parametrize("experiment", ["20", "30", "40"])
@pytest.mark.parametrize(
    ("case", "windows", "baseline", "pknl"),
    [
        ("plain", (0, 0), 0, False),
        ("phase", (0, 0), 0, True),
        ("em", (0, 1), 0, False),
        ("sine", (0, 3), 0, False),
        ("qpol", (0, 0), 4, False),
        ("qfil", (0, 0), 6, False),
        ("combined", (4, 3), 6, True),
    ],
)
def test_topspin_xfb_fractional_delay(experiment, case, windows, baseline, pknl):
    root = os.environ.get("NMR_TOPSPIN_XFB_FRACTIONAL_REFERENCES")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_XFB_FRACTIONAL_REFERENCES to the CMCse copies")
    path = Path(root).resolve(strict=True) / case / experiment
    trace = _assert_topspin_xfb(path, case == "combined")
    assert trace["pknl"] is pknl
    assert trace["window"] == windows
    assert trace.get("baseline", (0, 0)) == (0, baseline)
    delay = trace["group_delay_points"]
    assert 67 < delay < 68
    assert trace["f2_time_origin_points"] == 68


def _assert_topspin_xfb(path, with_prediction):
    experiment = path.name
    raw = _local_read(path / "ser")
    before = raw.data.copy()

    result = Experiment(raw).process(replay="topspin")

    names = ["2rr"] if experiment == "20" else ["2rr", "2ri", "2ir", "2ii"]
    pdata = path / "pdata" / "1"
    meta, planes = ng.bruker.read_pdata(pdata, bin_files=names, scale_data=True)
    if experiment == "20":
        actual = result.data[None]
        expected = planes[None]
    else:
        assert tuple(result.c.labels) == ("S_R", "S_I")
        with pytest.raises(ValueError, match="branches in acquired order"):
            Experiment(raw[::-1, :, :]).process(replay="topspin")
        first, second = result.data
        actual = np.stack((first.real, second.real, first.imag, second.imag))
        expected = np.asarray(planes)

    gain = meta["acqus"]["NS"] * meta["acqus"]["RG"]
    if meta["acqus"]["DTYPA"] == 0:
        gain *= 2.0 ** meta["acqus"]["NC"]
    residual = actual * gain - expected
    relative = np.linalg.norm(residual, axis=(1, 2)) / np.linalg.norm(
        expected, axis=(1, 2)
    )
    assert np.max(relative) < 2e-6
    trace = result.meta.nmr_processing["scp_processing"]["applied"]
    if with_prediction:
        pars = meta["proc2s"]
        assert pars["ME_mod"] == 2
        assert trace["linear_prediction"] == {
            "axis": "F1",
            "mode": "LPfr",
            "order": pars["NCOEF"] // 2,
            "size": pars["LPBIN"] // (1 if experiment == "20" else 2),
        }
    else:
        assert meta["proc2s"]["ME_mod"] == 0
        assert "linear_prediction" not in trace
    for axis, key in ((result.y, "proc2s"), (result.x, "procs")):
        pars = meta[key]
        grid = pars["OFFSET"] - np.arange(pars["SI"]) * pars["SW_p"] / (
            pars["SF"] * pars["SI"]
        )
        np.testing.assert_allclose(axis.data, grid, atol=2e-12, rtol=0)
        assert axis.units == scp.ur.ppm
    assert Experiment(result).is_frequency_domain
    np.testing.assert_array_equal(raw.data, before)

    with pytest.raises(ValueError, match="uncorrected SER"):
        Experiment(_local_read(path / "ser", filtered=True)).process(replay="topspin")
    return trace


@pytest.mark.parametrize("expno", range(80, 95))
def test_qsim_cp_pknl_matches_topspin_ft(expno):
    """Compare em → ft (BC_mod=0, PKNL on) on copies of the CP FIDs."""
    root = os.environ.get("NMR_TOPSPIN_CP_REFERENCES")
    if root is None:
        pytest.skip("set NMR_TOPSPIN_CP_REFERENCES to the captured CP series")
    root = Path(root).resolve(strict=True)
    path = root / "cp" / str(expno) / "fid"
    fid = _local_read(path)
    original = fid.data.copy()
    spectrum = Experiment(fid).process(
        digital_filter="pknl", apodization="em", lb=50, size=8192
    )

    pars, (real, imag) = ng.bruker.read_pdata(
        root / "ft" / str(expno), bin_files=["1r", "1i"]
    )
    reference = real + 1j * imag
    # The reader and TopSpin use different intensity scales. Fit a single
    # positive real factor, which cannot hide a phase or quadrature error.
    scale = (
        np.vdot(spectrum.data, reference).real
        / np.vdot(spectrum.data, spectrum.data).real
    )
    assert scale > 0
    assert (
        np.linalg.norm(scale * spectrum.data - reference) / np.linalg.norm(reference)
        < 1e-7
    )
    p = pars["procs"]
    assert p["BC_mod"] == 0 and p["PKNL"] is True
    bins = np.arange(p["SI"]) / p["SI"]
    np.testing.assert_allclose(
        spectrum.x.data, p["OFFSET"] - bins * p["SW_p"] / p["SF"], atol=5e-5, rtol=0
    )
    assert spectrum.meta.acquisition["digital_filter"]["group_delay_points"] == 66.625
    np.testing.assert_array_equal(fid.data, original)
    with pytest.raises(ValueError, match="already applied"):
        Experiment(_local_read(path, filtered=True)).process(digital_filter="pknl")
