"""Public Bruker echo/anti-echo reader and FFT parity."""

import hashlib
import os
from importlib import import_module
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.experiment import Experiment


@pytest.mark.parametrize("dc", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_public_echo_antiecho_uses_normalized_dataset(monkeypatch, dc, filtered):
    configured = os.environ.get("NMR_BRUKER_PUBLIC_ECHO_ANTIECHO_DATA")
    if not configured:
        pytest.skip("configure the real public echo/anti-echo reference")
    if not hasattr(ng.interop, "normalize_bruker_ser_echo_antiecho"):
        pytest.skip("requires the provider echo/anti-echo normalization contract")
    path = Path(configured)
    files = [
        path / name
        for name in ("ser", "acqus", "acqu2s", "pdata/1/procs", "pdata/1/proc2s")
    ]
    before = [hashlib.sha256(file.read_bytes()).hexdigest() for file in files]
    options = dict(remove_dc_offset=dc, remove_digital_filter=filtered)
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    converter = import_module("spectrochempy_nmr.interop")
    with monkeypatch.context() as baseline:
        baseline.setattr(reader, "_read_normalized_ser", lambda *args, **kwargs: None)
        expected = scp.read_topspin(path / "ser", **options)

    captured = []
    convert = converter.from_interop

    def capture(result, **kwargs):
        dataset = convert(result, **kwargs)
        captured.append(dataset)
        return dataset

    with monkeypatch.context() as migrated:
        migrated.setattr(converter, "from_interop", capture)
        actual = scp.read_topspin(path / "ser", **options)

    assert len(captured) == 1 and actual is captured[0]
    assert tuple(actual.c.labels) == ("first", "second")
    assert actual.meta.echo_antiecho_reconstruction is None
    assert actual.meta.acquisition["dc_offset"]["applied"] is dc
    assert actual.meta.acquisition["digital_filter"]["applied"] is filtered
    assert actual.meta.acquisition["scan_gain"]["divisor"] == (
        expected.meta.rgold[-1] * expected.meta.nsold[-1]
    )
    np.testing.assert_allclose(actual.data.conj(), expected.data, rtol=2e-12, atol=1e-7)
    experiment = Experiment(actual)
    if filtered:
        with pytest.raises(ValueError, match="already been corrected"):
            experiment.process(echo_antiecho="bruker-rance-kay", digital_filter="phase")
        physical = experiment.process(echo_antiecho="bruker-rance-kay")
    else:
        physical = experiment.process(
            echo_antiecho="bruker-rance-kay", digital_filter="phase"
        )
        assert (
            physical.meta.bruker_digital_filter["delay_source"]
            == "normalized acquisition"
        )
    assert physical.x.units == scp.ur.Hz
    assert physical.meta.echo_antiecho_reconstruction == "bruker-rance-kay"
    with pytest.raises(NotImplementedError, match="PENDING-F2"):
        actual.fft(dim="y")

    for dim, domains, reconstruction in (
        ("x", ("time", "frequency"), None),
        ("y", ("frequency", "frequency"), "public-branch-fft"),
    ):
        actual, expected = actual.fft(dim=dim), expected.fft(dim=dim)
        assert Experiment(actual).domains == domains
        assert actual.meta.echo_antiecho_reconstruction == reconstruction
        assert actual.meta.tppi_alternation_state is None
        assert tuple(actual.c.labels) == ("first", "second")
        np.testing.assert_allclose(actual.data, expected.data, rtol=2e-12, atol=2e-6)
        assert actual.coordset[dim].units == expected.coordset[dim].units == scp.ur.ppm
        # Derive the unrounded grid from the historical acquisition metadata.
        # Coord linearization in the legacy path can round the step itself.
        axis = expected.dims.index(dim)
        meta = expected.meta
        center = (meta.sfo1[axis] - meta.sf[axis]).to("Hz").m
        width = meta.sw_h[axis].to("Hz").m
        reference = meta.bf1[axis].to("MHz").m
        grid = np.linspace(center + width / 2, center - width / 2, actual.shape[axis])
        np.testing.assert_allclose(
            actual.coordset[dim].data, grid / reference, rtol=0, atol=1e-8
        )
        np.testing.assert_allclose(
            actual.coordset[dim].data, expected.coordset[dim].data, rtol=0, atol=0.01
        )
    assert [hashlib.sha256(file.read_bytes()).hexdigest() for file in files] == before
