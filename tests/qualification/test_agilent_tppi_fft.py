"""Compare the public Agilent TPPI workflow with native NMRPipe checkpoints."""

import os
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy_nmr.experiment import Experiment


def test_native_agilent_tppi():
    source = os.getenv("SCP_AGILENT_TPPI_FID_DIR")
    oracle = os.getenv("SCP_AGILENT_TPPI_ORACLE_DIR")
    if not source or not oracle:
        pytest.skip("native Agilent TPPI acquisition and NMRPipe checkpoints required")

    dataset = scp.nmr.read_agilent(source)
    folder = Path(oracle)
    h2, f2 = ng.pipe.read(str(folder / "f2.ft"))
    h1, f1 = ng.pipe.read(str(folder / "f1-zf.ft"))
    _, real_branch = ng.pipe.read(str(folder / "RR-IR.ft"))
    _, imag_branch = ng.pipe.read(str(folder / "RI-II.ft"))
    reference = (
        (h1["FDF1OBS"], h1["FDF1CAR"]),
        (h2["FDF2OBS"], h2["FDF2CAR"]),
    )
    direct = dataset.fft(dim="x")
    assert np.linalg.norm(direct.data.conj() - f2) / np.linalg.norm(f2) < 1e-6

    experiment = Experiment(dataset)
    spectrum = experiment.process(tppi_reference=reference)
    assert spectrum.meta.component_labels == ("rr", "ri", "ir", "ii")
    assert spectrum.units == dataset.units
    for plane, expected in zip(
        spectrum.data,
        (
            real_branch.real.T,
            imag_branch.real.T,
            real_branch.imag.T,
            imag_branch.imag.T,
        ),
        strict=True,
    ):
        assert np.linalg.norm(plane - expected) / np.linalg.norm(expected) < 1e-6
    for axis, header, checkpoint in ((spectrum.y, h1, f1), (spectrum.x, h2, f2)):
        assert axis.units == scp.ur.ppm
        np.testing.assert_allclose(
            axis.data,
            ng.pipe.make_uc(header, checkpoint).ppm_scale(),
            rtol=0,
            atol=1e-5,
        )
    unreferenced = experiment.process()
    assert unreferenced.y.units == unreferenced.x.units == scp.ur.Hz
    assert experiment.is_time_domain
