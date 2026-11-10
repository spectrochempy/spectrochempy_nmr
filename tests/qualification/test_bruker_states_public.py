"""Compare a native Bruker States SER with NMRPipe."""

import hashlib
from importlib import import_module
import os
from pathlib import Path
import shutil
import subprocess

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment


def test_real_bruker_states_nmrpipe(monkeypatch, tmp_path):
    root = os.environ.get("NMR_BRUKER_PUBLIC_STATES_DATA")
    nmrpipe = os.environ.get("NMR_PIPE_EXECUTABLE") or shutil.which("nmrPipe")
    bruk2pipe = os.environ.get("NMR_BRUK2PIPE_EXECUTABLE") or shutil.which("bruk2pipe")
    if not root or not nmrpipe or not bruk2pipe:
        pytest.skip("configure the Bruker States reference and NMRPipe tools")
    if not hasattr(ng.interop, "normalize_bruker_ser_states"):
        pytest.skip("requires the provider States normalization contract")
    path = Path(root)
    checksums = {
        "ser": "eca52a5418efc10cb615f1477376f6d8f6a1f2632c9848e4a32f7dc1cfb7e0a8",
        "acqus": "778ba31375056a0f0153c96053a230501a09db8f7d87a519b5f617dfd242becc",
        "acqu2s": "229bc17cb4ffe8f0c593ff327c5e1d3aafb931517277dbf88632a98571b96075",
        "pdata/1/procs": "76705a5686ce5271541524418f67a98fc4bfc455b982077397a7e00ff96d381c",
        "pdata/1/proc2s": "be97bb78aa0133cba0c82b518ce64d5a51d721f0c2b43107cc136264c20a8e84",
    }

    def digests():
        return {
            name: hashlib.sha256((path / name).read_bytes()).hexdigest()
            for name in checksums
        }

    assert digests() == checksums

    converter = import_module("spectrochempy_nmr.interop")
    convert = converter.from_interop
    captured = []

    def capture(result, **kwargs):
        dataset = convert(result, **kwargs)
        captured.append(dataset)
        return dataset

    monkeypatch.setattr(converter, "from_interop", capture)
    options = {"remove_dc_offset": False}
    fid = scp.read_topspin(path / "ser", remove_digital_filter=False, **options)
    filtered = scp.read_topspin(path / "ser", remove_digital_filter=True, **options)
    assert len(captured) == 2 and captured[0] is fid and captured[1] is filtered
    assert Experiment(fid).encoding == ("STATES", "DQD")
    assert fid.meta.acquisition["digital_filter"]["applied"] is False
    assert filtered.meta.acquisition["digital_filter"]["applied"] is True
    assert filtered.meta.acquisition["digital_filter"]["removed_points"] == 78
    shifted = np.roll(fid.data, -76, axis=-1)
    shifted[..., :72] += shifted[..., -1:-73:-1]
    np.testing.assert_allclose(filtered.data, shifted[..., :-78], atol=4e-13)
    np.testing.assert_array_equal(filtered.x.data, fid.x.data[:-78])

    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    with monkeypatch.context() as baseline:
        baseline.setattr(reader, "_read_normalized_ser", lambda *args, **kwargs: None)
        historical = scp.read_topspin(
            path / "ser", remove_digital_filter=False, **options
        )
    public_fft = fid
    for dim in ("x", "y"):
        public_fft, historical = public_fft.fft(dim=dim), historical.fft(dim=dim)
        np.testing.assert_allclose(
            public_fft.data, historical.data, rtol=2e-12, atol=1e-8
        )
        axis = public_fft.coordset[dim]
        assert axis.units == historical.coordset[dim].units == scp.ur.ppm
        np.testing.assert_allclose(
            axis.data, historical.coordset[dim].data, rtol=0, atol=0.00101
        )

    raw = tmp_path / "raw.fid"
    geometry = (
        "-bad 0.0 -noswap -AMX -decim 416 -dspfvs 21 -grpdly 76 -ndim 2 "
        "-xN 5376 -xT 2688 -xMODE Complex -xSW 48076.9230769 "
        "-xOBS 130.3042724 -xCAR 0 -xLAB Al27 "
        "-yN 128 -yT 64 -yMODE States -ySW 14005.6022409 "
        "-yOBS 184.5977192 -yCAR 0 -yLAB Al27 -aq2D States"
    )
    subprocess.run(
        [
            bruk2pipe,
            "-in",
            str(path / "ser"),
            *geometry.split(),
            "-out",
            str(raw),
            "-ov",
        ],
        check=True,
        capture_output=True,
    )
    _, traces = ng.pipe.read(str(raw))
    divisor = fid.meta.acquisition["scan_gain"]["divisor"]
    assert divisor == 194880
    np.testing.assert_allclose(
        fid.data, np.stack((traces[::2], traces[1::2])) / divisor
    )

    source = raw
    for label, args in (
        ("f2", ("FT", "-nodf")),
        ("tp", ("TP",)),
        ("f1", ("FT", "-nodf")),
    ):
        output = tmp_path / f"{label}.pipe"
        subprocess.run(
            [nmrpipe, "-in", str(source), "-fn", *args, "-out", str(output), "-ov"],
            check=True,
            capture_output=True,
        )
        source = output

    header, final = ng.pipe.read(str(source))
    first, second = final[::2].T, final[1::2].T
    spectrum = Experiment(fid).process()
    assert tuple(spectrum.c.labels) == ("rr", "ri", "ir", "ii")
    for actual, plane in zip(
        spectrum.data, (first.real, second.real, first.imag, second.imag), strict=True
    ):
        expected = plane / divisor
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-6
    for dim, axis in (("F1", spectrum.y), ("F2", spectrum.x)):
        width = header[f"FD{dim}SW"]
        origin = header[f"FD{dim}ORIG"]
        expected = origin + (axis.size - 1 - np.arange(axis.size)) * width / axis.size
        assert axis.units == scp.ur.Hz
        np.testing.assert_allclose(axis.data, expected, atol=2e-3, rtol=0)
    assert spectrum.meta.get("tppi_alternation_state") is None
    assert digests() == checksums


@pytest.mark.parametrize("paired_ser", [4], indirect=True)
@pytest.mark.parametrize("filtered", [False, True])
def test_synthetic_public_states_asymmetric_branches(paired_ser, monkeypatch, filtered):
    if not hasattr(ng.interop, "normalize_bruker_ser_states"):
        pytest.skip("requires the provider States normalization contract")

    converter = import_module("spectrochempy_nmr.interop")
    convert = converter.from_interop
    captured = []

    def capture(result, **kwargs):
        dataset = convert(result, **kwargs)
        captured.append(dataset)
        return dataset

    monkeypatch.setattr(converter, "from_interop", capture)
    actual = scp.read_topspin(
        paired_ser / "ser", remove_dc_offset=False, remove_digital_filter=filtered
    )
    assert len(captured) == 1 and actual is captured[0]
    assert Experiment(actual).encoding == ("STATES", "DQD")
    words = np.fromfile(paired_ser / "ser", dtype=">i4").reshape(8, 256)
    traces = words[:, ::2] + 1j * words[:, 1::2]
    if filtered:
        traces = np.roll(traces, -2, axis=-1)
        traces = traces[:, :-4]
    traces /= 288
    branches = np.stack((traces[::2], traces[1::2]))
    np.testing.assert_allclose(actual.data, branches, atol=1e-12)
    np.testing.assert_allclose(actual.y.data, np.arange(4) / 1000)
    np.testing.assert_allclose(actual.x.data, np.arange(traces.shape[1]) / 2000)
    assert actual.meta.acquisition["digital_filter"]["applied"] is filtered
    assert actual.meta.get("tppi_alternation_state") is None

    # Direct positive-sign DFT matrices give an oracle independent of the FFT.
    def kernel(size):
        bins = np.arange(size) - size // 2
        return np.exp(2j * np.pi * np.outer(bins, np.arange(size)) / size)

    f2 = branches @ kernel(traces.shape[1]).T
    f1 = kernel(4)
    real = f1 @ (f2[0].real + 1j * f2[1].real)
    imag = f1 @ (f2[0].imag + 1j * f2[1].imag)
    spectrum = Experiment(actual).process()
    np.testing.assert_allclose(
        spectrum.data, [real.real, imag.real, real.imag, imag.imag], atol=1e-9
    )
    assert tuple(spectrum.c.labels) == ("rr", "ri", "ir", "ii")
    for axis, width in ((spectrum.y, 1000), (spectrum.x, 2000)):
        assert axis.units == scp.ur.Hz
        np.testing.assert_allclose(
            axis.data, (axis.size // 2 - np.arange(axis.size)) * width / axis.size
        )
    with pytest.raises(NotImplementedError, match="PENDING-F2"):
        actual.fft(dim="y")
    direct = actual.fft(dim="x").fft(dim="y")
    assert tuple(direct.c.labels) == ("F1_R", "F1_I")
