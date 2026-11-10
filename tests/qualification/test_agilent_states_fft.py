"""Opt-in component-level comparison after explicit Agilent NMRPipe transport."""

import hashlib
import os
from inspect import signature
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr import Experiment
from nmrglue.interop import normalize_agilent_states_2d


@pytest.mark.skipif(
    "quadrature" not in signature(normalize_agilent_states_2d).parameters,
    reason="provider does not support Agilent quadrature selection",
)
def test_agilent_states_nmrpipe_transport_and_fft(tmp_path):
    root = os.environ.get("NMR_AGILENT_STATES_REFERENCE")
    nmrpipe = os.environ.get("NMR_PIPE_EXECUTABLE") or shutil.which("nmrPipe")
    var2pipe = os.environ.get("NMR_VAR2PIPE_EXECUTABLE") or shutil.which("var2pipe")
    if not root or not nmrpipe or not var2pipe:
        pytest.skip("configure the historical Agilent States FID and NMRPipe tools")
    import nmrglue as ng

    directory = Path(root)
    checksums = {
        "fid": "4a35f4b98f9116ba67760d1c0c83cc48a17b4e53897b5dbe1e25f85f28b1eac8",
        "procpar": "54cf12841987ec3a05f5ca76f2b09dff09b7648d84b2936627f2c9757472c817",
    }
    for name, digest in checksums.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest

    native = scp.nmr.read_agilent(directory)
    assert native.meta.quadrature_mode == "states-stored-2d"
    with pytest.raises(NotImplementedError, match="UNVALIDATED"):
        native.fft(dim="y")

    transported = tmp_path / "agilent.fid"
    geometry = (
        "-noaswap -ndim 2 "
        "-xN 3000 -xT 1500 -xMODE Complex -xSW 50000 -xOBS 125.691 -xCAR 0 -xLAB C13 "
        "-yN 332 -yT 166 -yMODE States -ySW 5555.55555556 -yOBS 50.648 -yCAR 0 -yLAB N15 "
        "-aq2D States"
    )
    subprocess.run(
        [
            var2pipe,
            "-in",
            str(directory / "fid"),
            *geometry.split(),
            "-out",
            str(transported),
            "-ov",
        ],
        check=True,
        capture_output=True,
    )
    fid = scp.nmr.read(
        transported,
        protocol="interop",
        normalize_options={"confirm_time_grid": True},
        conversion_profile="states",
    )
    np.testing.assert_array_equal(fid.data[0], native.data[0].conj())
    np.testing.assert_array_equal(fid.data[1], -native.data[1].conj())
    selected = scp.nmr.read_agilent(directory, quadrature="states")
    np.testing.assert_array_equal(selected.data, native.data)
    assert selected.meta.acquisition["quadrature_selection"] == {
        "mode": "states",
        "source": "manual",
        "sequence": "cpjdcptan2d",
    }

    source = transported
    stages = (("f2", ("FT", "-nodf")), ("tp", ("TP",)), ("f1", ("FT", "-nodf")))
    for label, args in stages:
        output = tmp_path / f"{label}.pipe"
        subprocess.run(
            [nmrpipe, "-in", str(source), "-fn", *args, "-out", str(output), "-ov"],
            check=True,
            capture_output=True,
        )
        source = output

    header, final = ng.pipe.read(str(source))
    first, second = final[::2].T, final[1::2].T
    spectrum = Experiment(selected).process()
    np.testing.assert_array_equal(spectrum.data, Experiment(fid).process().data)
    np.testing.assert_array_equal(selected.data, native.data)
    assert tuple(spectrum.c.labels) == ("rr", "ri", "ir", "ii")
    for actual, expected in zip(
        spectrum.data, (first.real, second.real, first.imag, second.imag), strict=True
    ):
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-6
    for dim, axis in (("F1", spectrum.y), ("F2", spectrum.x)):
        width = header[f"FD{dim}SW"]
        origin = header[f"FD{dim}ORIG"]
        expected = origin + (axis.size - 1 - np.arange(axis.size)) * width / axis.size
        assert axis.units == scp.ur.Hz
        np.testing.assert_allclose(axis.data, expected, atol=2e-3, rtol=0)
    assert spectrum.meta.get("tppi_alternation_state") is None
    for name, digest in checksums.items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
