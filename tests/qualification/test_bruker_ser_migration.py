"""Public TopSpin fallback parity with independently decoded SER traces."""

from functools import wraps
from importlib import import_module
import os
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy.utils.exceptions import SpectroChemPyError


def _check_parity(path, monkeypatch, *, storage, rows, points, filtered, dc):
    reader = import_module("spectrochempy_nmr.readers.read_topspin")
    calls = []
    decode = ng.bruker.read

    @wraps(decode)
    def capture(directory, **kwargs):
        calls.append((directory, kwargs))
        return decode(directory, **kwargs)

    monkeypatch.setattr(reader, "_read_normalized_ser", lambda *args, **kwargs: None)
    monkeypatch.setattr(ng.bruker, "read", capture)
    result = scp.read_topspin(
        path / "ser", remove_digital_filter=filtered, remove_dc_offset=dc
    )

    assert len(calls) == 1
    directory, kwargs = calls[0]
    assert directory == path
    assert kwargs["recover_shape"] is False
    assert kwargs["read_nus_schedule"] is False
    assert kwargs["legacy_layout"] is True
    assert result is not None

    meta = ng.bruker.read_acqus_file(path)
    direct, indirect = meta["acqus"], meta["acqu2s"]
    words = np.fromfile(path / "ser", dtype=storage).reshape(rows, -1)
    expected = words[:, : 2 * points : 2] + 1j * words[:, 1 : 2 * points : 2]
    if dc:
        expected -= expected.mean(axis=-1, keepdims=True)
    if filtered:
        delay = int(direct["GRPDLY"])
        expected = np.roll(expected, -delay, axis=-1)
        removed = delay + 2
        fold = max(removed - 6, 0)
        if fold:
            expected[:, :fold] += expected[:, -fold:][:, ::-1]
        expected = expected[:, :-removed]
    expected = expected.conj() / (direct["RG"] * direct["NS"])
    paired = indirect["FnMODE"] in (5, 6)
    if paired:
        expected = np.stack((expected[::2], expected[1::2]))
        assert tuple(result.c.labels) == ("branch_0", "branch_1")
    np.testing.assert_allclose(result.data, expected, rtol=1e-12, atol=1e-8)
    assert result.dims == (["c", "y", "x"] if paired else ["y", "x"])
    for dim, params in (("x", direct), ("y", indirect)):
        coord = result.coordset[dim]
        width = params["SW"] * params["SFO1"]
        actual_seconds = (coord.data * coord.units).to("s").m
        np.testing.assert_allclose(
            actual_seconds, np.arange(coord.size) / width, atol=1e-6
        )


def _synthetic_ser(tmp_path, mode, complete=True, nus=False, floating=False):
    path = tmp_path / "bruker" / "1"
    path.mkdir(parents=True)
    td = 120 if floating else (240 if complete else 256)
    (path / "acqus").write_text(
        "##TITLE= Synthetic Bruker acquisition\n"
        f"##$TD= {td}\n##$SW= 16\n##$SW_h= 2000\n##$SFO1= 125\n"
        "##$BF1= 125\n##$O1= 0\n##$AQ_mod= 3\n##$DECIM= 16\n"
        f"##$DSPFVS= 12\n##$GRPDLY= 2.5\n##$BYTORDA= 1\n##$DTYPA= {2 if floating else 0}\n"
        "##$PARMODE= 1\n##$NUC1= <13C>\n##$NS= 4\n##$RG= 3\n"
        f"##$FnTYPE= {2 if nus else 0}\n##END=\n"
    )
    (path / "acqu2s").write_text(
        "##TITLE= Synthetic Bruker indirect acquisition\n"
        f"##$TD= {8 if complete else 6}\n##$SW= 8\n##$SFO1= 125\n"
        + (f"##$FnMODE= {mode}\n" if mode is not None else "")
        + "##$NUC1= <13C>\n##END=\n"
    )
    samples = np.arange(8 * (128 if floating else 256), dtype=np.int32)
    dtype = ">f8" if floating else ">i4"
    (path / "ser").write_bytes((samples * (samples % 7 - 3)).astype(dtype).tobytes())
    if nus:
        # The legacy path ignores this directory, while nmrglue-ng opens it.
        (path / "nuslist").mkdir()
    return path


@pytest.mark.parametrize(
    ("mode", "complete", "nus", "floating"),
    [(mode, complete, False, False) for mode in (1, 5, 6) for complete in (True, False)]
    + [(5, True, True, False), (1, True, False, True)],
)
def test_synthetic_public_ser_parity(
    tmp_path, monkeypatch, mode, complete, nus, floating
):
    path = _synthetic_ser(tmp_path, mode, complete, nus, floating)
    dtype = ">f8" if floating else ">i4"
    _check_parity(
        path,
        monkeypatch,
        storage=dtype,
        rows=8,
        points=60 if floating else 128,
        filtered=True,
        dc=True,
    )


@pytest.mark.parametrize(
    ("reference", "rows", "points"),
    [
        ("exam2d_HH/1", 128, 1024),
        ("exam2d_HC/3", 256, 512),
        ("topspin_2d/1", 192, 1024),
    ],
)
@pytest.mark.parametrize("filtered", [False, True])
def test_local_public_ser_parity(reference, rows, points, filtered, monkeypatch):
    root = os.environ.get("NMR_BRUKER_SER_DATA")
    if not root:
        pytest.skip("set NMR_BRUKER_SER_DATA to the Bruker test-data directory")
    _check_parity(
        Path(root) / reference,
        monkeypatch,
        storage="<i4",
        rows=rows,
        points=points,
        filtered=filtered,
        dc=False,
    )


def test_synthetic_public_fid_ignores_stale_indirect_status(tmp_path):
    path = tmp_path / "bruker" / "1"
    path.mkdir(parents=True)
    (path / "acqus").write_text(
        "##TITLE= Synthetic FID\n##$TD= 256\n##$AQ_mod= 3\n"
        "##$BYTORDA= 1\n##$DTYPA= 0\n##$PARMODE= 0\n"
        "##$DECIM= 1\n##$DSPFVS= 12\n##$SW= 16\n##$SW_h= 2000\n"
        "##$SFO1= 125\n##$BF1= 125\n##$O1= 0\n##$NUC1= <13C>\n"
        "##$RG= 2\n##$NS= 4\n##END=\n"
    )
    (path / "acqu2s").write_text("##TITLE= Stale indirect\n##$TD= 2\n##END=\n")
    words = np.arange(512, dtype=">i4")
    (path / "fid").write_bytes(words.tobytes())
    dataset = scp.read_topspin(path / "fid", remove_digital_filter=False)
    expected = (words[::2] + 1j * words[1::2]) / 8
    np.testing.assert_array_equal(dataset.data, expected)
    assert dataset.dims == ["x"]


@pytest.mark.parametrize("paired_ser", [5], indirect=True)
def test_synthetic_experiment_folder_with_auxiliary_files(paired_ser, monkeypatch):
    """A complete experiment folder selects SER instead of its text files."""
    (paired_ser / "audita.txt").write_text("Bruker acquisition audit")
    monkeypatch.chdir(paired_ser.parent.parent)
    options = {"local_only": True, "remove_digital_filter": False}
    expected = scp.read_topspin(paired_ser / "ser", **options)
    for path in (paired_ser, paired_ser.relative_to(Path.cwd())):
        actual = scp.read_topspin(path, **options)
        np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("mode", [None, 0, 99, "yes"])
def test_synthetic_unknown_bruker_f1_is_only_required_for_f1(tmp_path, mode):
    path = _synthetic_ser(tmp_path, mode) / "ser"
    fid = scp.nmr.read_topspin(path)
    assert fid.dims == ["y", "x"]
    f2 = fid.fft(dim="x")
    assert f2.x.units == scp.ur.Hz
    before = f2.copy()
    with pytest.raises(NotImplementedError, match="Bruker F1.*quadrature"):
        f2.fft(dim="y", inplace=True, size=12)
    assert f2.dims == before.dims
    np.testing.assert_array_equal(f2.data, before.data)
    with pytest.raises(NotImplementedError, match="Bruker F1.*quadrature"):
        scp.nmr.Experiment(fid).process()

    resolved = scp.nmr.read_topspin(path, quadrature="qf")
    np.testing.assert_array_equal(resolved.data, fid.data)
    assert resolved.fft(dim="x").fft(dim="y").shape == fid.shape


@pytest.mark.parametrize(
    ("fnmode", "sw", "quadrature"),
    [(None, 8, None), (1, 0, None), (None, 8, "qf"), (1, 8, None)],
)
def test_synthetic_bruker_ser_on_core_without_prevalidation(
    tmp_path, monkeypatch, fnmode, sw, quadrature
):
    core_fft = import_module("spectrochempy.processing.fft.fft")
    monkeypatch.setattr(core_fft, "FFT_PREVALIDATION_API", 0, raising=False)
    path = _synthetic_ser(tmp_path, fnmode)
    acqu2s = path / "acqu2s"
    acqu2s.write_text(acqu2s.read_text().replace("##$SW= 8", f"##$SW= {sw}"))

    if (fnmode is None and quadrature is None) or sw == 0:
        with pytest.raises(SpectroChemPyError, match="Bruker SER.*1.2.0"):
            scp.nmr.read_topspin(path / "ser", quadrature=quadrature)
    else:
        fid = scp.nmr.read_topspin(path / "ser", quadrature=quadrature)
        assert fid.fft(dim="x").fft(dim="y").shape == fid.shape


@pytest.mark.parametrize("invalid_sw", ["0", "<BAD>", "yes", "(0..-1)", "(0..1)\n1 2"])
def test_synthetic_bruker_ser_uses_stored_width_if_sw_invalid(tmp_path, invalid_sw):
    path = _synthetic_ser(tmp_path, 1)
    acqu2s = path / "acqu2s"
    acqu2s.write_text(
        acqu2s.read_text().replace("##$SW= 8", f"##$SW= {invalid_sw}\n##$SW_h= 2000")
    )
    fid = scp.nmr.read_topspin(path / "ser")
    np.testing.assert_allclose(fid.y.to("s").data, np.arange(fid.shape[0]) / 2000)
    assert fid.fft(dim="x").fft(dim="y").shape == fid.shape


@pytest.mark.parametrize("fnmode", ["yes", "1.0"])
def test_synthetic_invalid_fnmode_with_processing_parameters(tmp_path, fnmode):
    path = _synthetic_ser(tmp_path, fnmode)
    acqu2s = path / "acqu2s"
    acqu2s.write_text(acqu2s.read_text().replace("##END=", "##$O1= 0\n##END="))
    status = path / "pdata" / "1"
    status.mkdir(parents=True)
    for name, width, size in (("procs", 2000, 128), ("proc2s", 1000, 8)):
        (status / name).write_text(
            f"##TITLE= Processing status\n##$SF= 125\n##$SW_p= {width}\n"
            f"##$OFFSET= 4\n##$SI= {size}\n##END=\n"
        )

    fid = scp.nmr.read_topspin(path / "ser")
    with pytest.raises(NotImplementedError, match="Bruker F1.*quadrature"):
        fid.fft(dim="x").fft(dim="y")


@pytest.mark.parametrize("sfo1", ["yes", "<BAD>", "NaN", "0", "1e999"])
def test_synthetic_bruker_f1_fft_with_invalid_reference(tmp_path, sfo1):
    path = _synthetic_ser(tmp_path, 1)
    acqu2s = path / "acqu2s"
    acqu2s.write_text(
        acqu2s.read_text().replace(
            "##$SFO1= 125", f"##$SFO1= {sfo1}\n##$BF1= 125\n##$SW_h= 2000"
        )
    )
    status = path / "pdata" / "1" / "proc2s"
    status.parent.mkdir(parents=True)
    status.write_text("##TITLE= Processing status\n##$SF= 125\n##END=\n")

    spectrum = scp.nmr.read_topspin(path / "ser").fft(dim="x").fft(dim="y")
    assert spectrum.y.units == scp.ur.Hz
    assert np.isfinite(spectrum.y.data).all()
    assert spectrum.meta.isfreq == [True, True]


def test_synthetic_bruker_ser_ignores_boolean_mc2(tmp_path):
    path = _synthetic_ser(tmp_path, None)
    status = path / "pdata" / "1" / "proc2s"
    status.parent.mkdir(parents=True)
    status.write_text("##TITLE= Processing status\n##$MC2= no\n##END=\n")
    fid = scp.nmr.read_topspin(path / "ser")
    with pytest.raises(NotImplementedError, match="Bruker F1.*quadrature"):
        fid.fft(dim="x").fft(dim="y")


@pytest.mark.parametrize(("fnmode", "expected"), [(None, "STATES-TPPI"), (4, "STATES")])
def test_synthetic_bruker_f1_prefers_fnmode_to_mc2(tmp_path, fnmode, expected):
    path = _synthetic_ser(tmp_path, fnmode)
    status = path / "pdata" / "1" / "proc2s"
    status.parent.mkdir(parents=True)
    status.write_text("##TITLE= Processing status\n##END=\n")
    status.with_name("proc2").write_text("##TITLE= Processing\n##$MC2= 4\n##END=\n")
    fid = scp.nmr.read_topspin(path / "ser")
    assert fid.meta.acquisition_encoding[0] == expected
    assert fid.fft(dim="x").fft(dim="y").shape == fid.shape


def test_synthetic_bruker_ser_uses_mc2_over_fnmode_in_processing_parameters(tmp_path):
    path = _synthetic_ser(tmp_path, None)
    status = path / "pdata" / "1" / "proc2s"
    status.parent.mkdir(parents=True)
    status.write_text("##TITLE= Processing status\n##$FnMODE= 5\n##$MC2= 0\n##END=\n")
    fid = scp.nmr.read_topspin(path / "ser")
    assert fid.dims == ["y", "x"]
    assert fid.meta.encoding[0] == "QF"


def test_synthetic_bruker_f2_allows_missing_direct_nucleus(tmp_path):
    path = _synthetic_ser(tmp_path, None)
    acqus = path / "acqus"
    acqus.write_text(acqus.read_text().replace("##$NUC1= <13C>\n", ""))
    status = path / "pdata" / "1" / "procs"
    status.parent.mkdir(parents=True)
    status.write_text("##TITLE= Processing status\n##$SF= 125\n##END=\n")

    fid = scp.nmr.read_topspin(path / "ser")
    f2 = fid.fft(dim="x")
    assert f2.x.units == scp.ur.ppm
    with pytest.raises(NotImplementedError, match="Bruker F1.*quadrature"):
        f2.fft(dim="y")


@pytest.mark.parametrize("prevalidation_api", [0, 1])
def test_synthetic_bruker_ser_list_does_not_supply_f1_width(
    tmp_path, monkeypatch, prevalidation_api
):
    core_fft = import_module("spectrochempy.processing.fft.fft")
    monkeypatch.setattr(
        core_fft, "FFT_PREVALIDATION_API", prevalidation_api, raising=False
    )
    path = _synthetic_ser(tmp_path, 1)
    acqu2s = path / "acqu2s"
    acqu2s.write_text(acqu2s.read_text().replace("##$SW= 8", "##$SW= 0"))
    (path / "vdlist").write_text("1\n2\n3\n4\n5\n6\n7\n8\n")

    if prevalidation_api == 0:
        with pytest.raises(SpectroChemPyError, match="Bruker SER.*1.2.0"):
            scp.nmr.read_topspin(path / "ser", use_list=True)
    else:
        fid = scp.nmr.read_topspin(path / "ser", use_list=True)
        np.testing.assert_array_equal(fid.y.data, np.arange(1, 9))
        with pytest.raises(
            ValueError, match="Bruker F1 FFT requires an acquisition spectral width"
        ):
            fid.fft(dim="x").fft(dim="y")


@pytest.mark.parametrize("stored_width", [None, 1000])
def test_synthetic_bruker_ser_width_uses_only_acquisition_parameters(
    tmp_path, stored_width
):
    path = _synthetic_ser(tmp_path, 1)
    acqu2s = path / "acqu2s"
    params = acqu2s.read_text().replace("##$SW= 8\n", "")
    if stored_width is not None:
        params = params.replace("##END=", f"##$SW_h= {stored_width}\n##END=")
    acqu2s.write_text(params)
    status = path / "pdata" / "1" / "proc2s"
    status.parent.mkdir(parents=True)
    status.write_text(
        "##TITLE= Processing status\n##$SW_h= 4000\n##$SW= 32\n##$SFO1= 125\n##END=\n"
    )

    fid = scp.nmr.read_topspin(path / "ser")
    f2 = fid.fft(dim="x")
    if stored_width is None:
        np.testing.assert_array_equal(fid.y.data, np.arange(fid.shape[0]))
        with pytest.raises(
            ValueError, match="Bruker F1 FFT requires an acquisition spectral width"
        ):
            f2.fft(dim="y")
    else:
        np.testing.assert_allclose(
            fid.y.to("s").data, np.arange(fid.shape[0]) / stored_width
        )
        assert f2.fft(dim="y").y.units == scp.ur.Hz


@pytest.mark.parametrize("width_hz", [None, 0, -1000, "NaN", "1e999", 1000])
def test_synthetic_bruker_ser_without_indirect_sfo1(tmp_path, width_hz):
    path = _synthetic_ser(tmp_path, 1)
    acqu2s = path / "acqu2s"
    params = acqu2s.read_text().replace("##$SFO1= 125\n", "")
    if width_hz is not None:
        params = params.replace("##END=", f"##$SW_h= {width_hz}\n##END=")
    acqu2s.write_text(params)

    fid = scp.nmr.read_topspin(path / "ser")
    f2 = fid.fft(dim="x")
    assert f2.x.units == scp.ur.Hz
    if width_hz != 1000:
        np.testing.assert_array_equal(fid.y.data, np.arange(fid.shape[0]))
        with pytest.raises(
            ValueError, match="Bruker F1 FFT requires an acquisition spectral width"
        ):
            f2.fft(dim="y")
    else:
        np.testing.assert_allclose(
            fid.y.to("s").data, np.arange(fid.shape[0]) / width_hz
        )
        assert f2.fft(dim="y").y.units == scp.ur.Hz
