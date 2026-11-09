"""Public QF SER route and historical two-stage FFT compatibility."""

from importlib import import_module
from inspect import signature
import hashlib
import os
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp

from spectrochempy_nmr.experiment import Experiment


def _check_route(path, monkeypatch, dc, filtered):
    normalizer = getattr(ng.interop, "normalize_bruker_ser_qf", None)
    if (
        normalizer is None
        or "qf_only" not in signature(ng.interop.read_bruker_ser).parameters
    ):
        pytest.skip("requires the provider QF FFT presentation contract")
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    converter = import_module("spectrochempy_nmr.interop")
    options = {"remove_dc_offset": dc, "remove_digital_filter": filtered}
    with monkeypatch.context() as old:
        old.setattr(reader, "_read_normalized_ser", lambda *args, **kwargs: None)
        expected = scp.read_topspin(path / "ser", **options)

    converted = []
    convert = converter.from_interop

    def capture(result, **kwargs):
        dataset = convert(result, **kwargs)
        converted.append(dataset)
        return dataset

    with monkeypatch.context() as new:
        new.setattr(converter, "from_interop", capture)
        actual = scp.read_topspin(path / "ser", **options)

    assert len(converted) == 1 and actual is converted[0]
    np.testing.assert_allclose(actual.data.conj(), expected.data, rtol=2e-12, atol=1e-7)
    assert actual.meta.acquisition["digital_filter"]["applied"] is filtered
    assert actual.meta.acquisition["dc_offset"]["applied"] is dc
    assert actual.meta.acquisition["scan_gain"]["divisor"] == (
        expected.meta.rgold[-1] * expected.meta.nsold[-1]
    )
    assert Experiment(actual).process().x.units == scp.ur.Hz
    before = actual.data.copy()
    dims = actual.dims.copy()
    with pytest.raises(NotImplementedError, match="PENDING-F2"):
        actual.fft(dim="y", size=2 * actual.y.size, inplace=True)
    np.testing.assert_array_equal(actual.data, before)
    assert actual.dims == dims
    assert actual.meta.isfreq == [False, False]

    first_stage = actual.fft(dim="x", ppm=False)
    inverse = first_stage.ifft(dim="x")
    assert Experiment(inverse).domains == ("time", "time")
    if not actual.meta.phc0[1].magnitude and not actual.meta.phc1[1].magnitude:
        np.testing.assert_allclose(inverse.data, actual.data, rtol=2e-12, atol=1e-7)
    np.testing.assert_allclose(
        inverse.fft(dim="x", ppm=False).data,
        first_stage.data,
        rtol=2e-12,
        atol=2e-6,
    )

    for dimension in ("x", "y"):
        actual = actual.fft(dim=dimension)
        expected = expected.fft(dim=dimension)
        np.testing.assert_allclose(actual.data, expected.data, rtol=2e-12, atol=2e-6)
        assert actual.coordset[dimension].units == expected.coordset[dimension].units
        np.testing.assert_allclose(
            actual.coordset[dimension].data,
            expected.coordset[dimension].data,
            rtol=0,
            atol=0.00101,
        )
        domains = (
            ("time", "frequency") if dimension == "x" else ("frequency", "frequency")
        )
        assert Experiment(actual).domains == domains

    partial = actual.ifft(dim="y")
    assert Experiment(partial).domains == ("time", "frequency")
    assert partial.y.units == scp.ur.s
    assert Experiment(partial.ifft(dim="x")).domains == ("time", "time")


@pytest.mark.parametrize("dc", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_synthetic_qf_public_with_nonzero_phase(tmp_path, monkeypatch, dc, filtered):
    path = tmp_path / "bruker" / "1"
    processing = path / "pdata" / "1"
    processing.mkdir(parents=True)
    (path / "acqus").write_text(
        "##TITLE= Synthetic direct\n"
        "##$TD= 256\n##$SW= 16\n##$SW_h= 2000\n##$SFO1= 125\n"
        "##$BF1= 124.9995\n##$O1= 500\n##$AQ_mod= 3\n##$DECIM= 16\n"
        "##$DSPFVS= 12\n##$GRPDLY= 2.5\n##$BYTORDA= 1\n##$DTYPA= 0\n"
        "##$PARMODE= 1\n##$NUC1= <13C>\n##$NS= 4\n##$RG= 3\n"
        "##$FnTYPE= 0\n##END=\n"
    )
    (path / "acqu2s").write_text(
        "##TITLE= Synthetic indirect\n"
        "##$TD= 8\n##$SW= 8\n##$SFO1= 125\n##$BF1= 124.9995\n"
        "##$O1= 500\n##$FnMODE= 1\n##$NUC1= <13C>\n##END=\n"
    )
    for name, zero, first in (("procs", 21, -8), ("proc2s", -30, 14)):
        (processing / name).write_text(
            f"##TITLE= Synthetic processing\n##$SF= 124.999\n"
            f"##$PHC0= {zero}\n##$PHC1= {first}\n##END=\n"
        )
    values = np.arange(8 * 256, dtype=np.int32)
    (path / "ser").write_bytes((values * (values % 7 - 3)).astype(">i4").tobytes())
    _check_route(path, monkeypatch, dc, filtered)
    if not dc and not filtered:
        core_fft = import_module("spectrochempy.processing.fft.fft")
        with monkeypatch.context() as old_core:
            old_core.setattr(core_fft, "FFT_PREVALIDATION_API", 0)
            legacy = scp.read_topspin(path / "ser")
            assert legacy.meta.normalization_profile is None
        (processing / "proc2s").unlink()
        reader = import_module("spectrochempy_nmr.readers.read_topspin")
        with monkeypatch.context() as legacy:
            legacy.setattr(reader, "_read_normalized_ser", lambda *args, **kwargs: None)
            expected = scp.read_topspin(path / "ser")
        actual = scp.read_topspin(path / "ser")
        assert actual.meta.normalization_profile is None
        np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("dc", [False, True])
@pytest.mark.parametrize("filtered", [False, True])
def test_real_qf_public_cosy(monkeypatch, dc, filtered):
    configured = os.environ.get("NMR_BRUKER_PUBLIC_QF_DATA")
    if not configured:
        pytest.skip("configure the Bruker COSY QF reference")
    path = Path(configured)
    files = [
        path / name
        for name in ("ser", "acqus", "acqu2s", "pdata/1/procs", "pdata/1/proc2s")
    ]
    before = [hashlib.sha256(file.read_bytes()).hexdigest() for file in files]
    _check_route(path, monkeypatch, dc, filtered)
    assert [hashlib.sha256(file.read_bytes()).hexdigest() for file in files] == before
