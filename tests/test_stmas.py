"""Public STMAS replay of a synthetic line with known F1 phase evolution."""

import numpy as np
import pytest
import spectrochempy as scp
from nmrglue import bruker
from spectrochempy.utils.exceptions import SpectroChemPyError


@pytest.fixture
def states_path(tmp_path):
    """Uniform States storage with one unpaired trace and padding."""
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

    return tmp_path, traces, (nx, ny, width, dwell, carrier, sf, kx, ky)


def test_stmas_removes_f2_dependent_f1_evolution(states_path):
    """Keep complete pairs and shear a known off-center frequency bin."""
    path, traces, (nx, ny, width, dwell, carrier, sf, kx, ky) = states_path
    with pytest.raises(SpectroChemPyError, match="complete States pairs"):
        scp.nmr.read(path, indirect_dwell="IN[0]")
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    assert raw.shape == (2, ny, nx)
    np.testing.assert_array_equal(
        raw.data, traces.reshape(ny, 2, nx).transpose(1, 0, 2)
    )
    assert "stmas" not in raw.meta.acquisition
    shortcut = scp.nmr.read_stmas(path, use_tdeff=True)
    np.testing.assert_array_equal(shortcut.data, raw.data)
    np.testing.assert_array_equal(shortcut.y.data, raw.y.data)

    xf2 = scp.nmr.Experiment(raw).xf2(size=nx, correct_digital_filter=False)
    phased = scp.nmr.Experiment(xf2).xf2p()
    out = scp.nmr.Experiment(phased).stmas_shear(size=ny)
    iy, ix = ny // 2 + ky, nx // 2 + kx
    expected = np.zeros((ny, nx))
    # The shared XFSHEAR path weights the first F1 sample by 1/2.
    expected[:, ix] = -nx / 2
    expected[iy, ix] += nx * ny
    view = scp.nmr.Experiment(out)
    assert out.shape == (2, ny, nx)
    np.testing.assert_allclose(view.RR.data, expected, atol=1e-10)
    assert np.max(np.abs(view.IR.data)) > 1
    np.testing.assert_allclose(view.RI.data, 0, atol=1e-10)
    np.testing.assert_allclose(view.II.data, 0, atol=1e-10)
    center = (carrier - sf) * 1e6 / sf
    f2_hz = kx * width / nx
    assert out.x.data[ix] == pytest.approx(center - f2_hz / sf)
    assert out.y.data[iy] == pytest.approx(center - ky / (ny * dwell * sf * 17 / 9))
    np.testing.assert_array_equal(
        raw.data, traces.reshape(ny, 2, nx).transpose(1, 0, 2)
    )


@pytest.mark.parametrize("nucleus", ["87Rb", "23Na", "27Al", "99Xx"])
def test_reader_does_not_select_experiment_or_spin(states_path, nucleus):
    path, _, _ = states_path
    acqus = path / "acqus"
    acqus.write_text(
        acqus.read_text()
        .replace("dqf-stmas-old.av", "unrelated-program")
        .replace("87Rb", nucleus)
    )
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    xf2 = scp.nmr.Experiment(raw).xf2(size=16, correct_digital_filter=False)
    phased = scp.nmr.Experiment(xf2).xf2p()
    if nucleus in ("87Rb", "23Na", "27Al"):
        assert scp.nmr.Experiment(phased).stmas_shear(size=16).shape == (2, 16, 16)
    else:
        with pytest.raises(ValueError, match="pass spin="):
            scp.nmr.Experiment(phased).stmas_shear(size=16)


def _workflow(raw, *, shift=0.0):
    mixed = scp.nmr.Experiment(raw).xf2(size=16, correct_digital_filter=False)
    phased = scp.nmr.Experiment(mixed).xf2p(phc0=-17, phc1=24, pivot=2)
    return (
        mixed,
        phased,
        scp.nmr.Experiment(phased).stmas_shear(size=32, f1_shift=shift),
    )


def test_stmas_uses_common_states_contract_and_survives_scp(
    states_path, tmp_path, monkeypatch
):
    from nmrglue import interop
    from test_mqmas_common import _neutral
    from spectrochempy_nmr.nmr_metadata import extract_nmr_metadata

    path, _, _ = states_path
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    expected = _workflow(raw, shift=0.1)
    generic = _neutral(raw)
    generic.origin = "other-vendor"
    generic.meta.readonly = False
    generic.meta.normalization_profile = "unrelated-profile"
    generic.meta.source_metadata = {"acqus": "opaque provenance"}
    for key in ("sw_h", "sfo1", "sf", "tdeff", "nuc1"):
        generic.meta[key] = [None, "opaque", "opaque"]
    generic.write(tmp_path / "raw.scp", confirm=False)

    def unavailable(*args, **kwargs):
        pytest.fail("Processing must not decode acquisition files")

    monkeypatch.setattr(interop, "read_bruker_ser_storage", unavailable)
    restored = scp.nmr.read(tmp_path / "raw.scp")
    before = restored.copy()
    actual = _workflow(restored, shift=0.1)
    for got, ref in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(got.data, ref.data)
        np.testing.assert_array_equal(got.x.data, ref.x.data)
        np.testing.assert_array_equal(got.y.data, ref.y.data)
    np.testing.assert_array_equal(restored.data, before.data)
    assert restored.meta == before.meta
    actual[1].write(tmp_path / "phased.scp", confirm=False)
    phased = scp.nmr.read(tmp_path / "phased.scp")
    resumed = scp.nmr.Experiment(phased).stmas_shear(size=32, f1_shift=0.1)
    np.testing.assert_array_equal(resumed.data, expected[-1].data)
    final = extract_nmr_metadata(resumed.meta, ndim=resumed.ndim)
    assert final.normalized_interop
    assert final.domains == ("frequency", "frequency")
    assert final.iscomplex == (True, True)
    assert resumed.shape == (2, 32, 16)
    assert final.nuclei == ("87Rb", "87Rb")
    assert scp.nmr.Experiment(resumed).source_kind == "processed_2d"


@pytest.mark.parametrize("option", ["numeric", "status-width"])
def test_indirect_time_calibration_is_explicit(states_path, option):
    path, _, (_, _, _, dwell, *_) = states_path
    if option == "numeric":
        raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell=dwell)
    else:
        file = path / "acqu2s"
        file.write_text(
            file.read_text().replace("##END", f"##$SW_h= {1 / dwell}\n##END")
        )
        raw = scp.nmr.read(path, use_tdeff=True)
    np.testing.assert_allclose(raw.y.data, np.arange(16) * dwell)
    assert raw.meta.acquisition["acquired_shape"] == (16, 16)


@pytest.mark.parametrize(
    "dwell", [True, np.bool_(True), 0, -1, np.nan, np.inf, "unknown"]
)
def test_invalid_dwell_is_rejected(states_path, dwell):
    path, _, _ = states_path
    with pytest.raises(ValueError, match="indirect_dwell"):
        scp.nmr.read(path, use_tdeff=True, indirect_dwell=dwell)


@pytest.mark.parametrize(
    "key,value", [("FnMODE", 5), ("TDeff", 31), ("TDeff", 36), ("FnTYPE", 2)]
)
def test_incomplete_or_nonuniform_selection_is_rejected(states_path, key, value):
    path, _, _ = states_path
    file = path / ({"FnMODE": "acqu2s", "FnTYPE": "acqus"}.get(key, "pdata/1/proc2s"))
    text = file.read_text()
    if key in text:
        import re

        text = re.sub(rf"(##\${key}= )[^\n]*", rf"\g<1>{value}", text)
    else:
        text = text.replace("##END", f"##${key}= {value}\n##END")
    file.write_text(text)
    with pytest.raises(SpectroChemPyError, match="uniform|complete States pairs"):
        scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")


def test_stmas_accepts_default_common_reader(tmp_path):
    from test_mqmas_common import states_path as common_states_path

    path = common_states_path.__wrapped__(tmp_path)
    for name in ("acqus", "acqu2s"):
        file = path / name
        file.write_text(file.read_text().replace("27Al", "87Rb"))
    raw = scp.nmr.read(path)
    assert "stmas" not in raw.meta.acquisition
    mixed = scp.nmr.Experiment(raw).xf2(size=256)
    phased = scp.nmr.Experiment(mixed).xf2p()
    spectrum = scp.nmr.Experiment(phased).stmas_shear(size=16)
    assert spectrum.shape == (2, 16, 256)
    assert np.isfinite(spectrum.data).all()


def test_selection_retains_indirect_nucleus_and_frequency(states_path):
    path, _, _ = states_path
    file = path / "acqu2s"
    file.write_text(
        file.read_text().replace("##END", "##$NUC1= <1H>\n##$SFO1= 500\n##END")
    )
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    assert raw.meta.acquisition["nuclei"] == ("1H", "87Rb")
    assert raw.meta.acquisition["acquisition_frequency_mhz"] == (500, 130)
    mixed = scp.nmr.Experiment(raw).xf2(size=16, correct_digital_filter=False)
    phased = scp.nmr.Experiment(mixed).xf2p()
    actual = scp.nmr.Experiment(phased).stmas_shear(size=32)
    expected = scp.nmr.Experiment(phased).xfshear(
        size=32, ratio=-8 / 9, f1_scale=17 / 9
    )
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)
    assert actual.meta.mqmas_shearing["spin"] == 1.5
    assert raw.meta.acquisition["nuclei"] == ("1H", "87Rb")


@pytest.mark.parametrize(
    "kwargs",
    [{"use_list": True}, {"remove_digital_filter": True}, {"quadrature": "qf"}],
)
def test_selection_refuses_incompatible_reader_options(states_path, kwargs):
    path, _, _ = states_path
    with pytest.raises(SpectroChemPyError, match="States|uncorrected"):
        scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]", **kwargs)


def test_unknown_mode_needs_explicit_quadrature(states_path):
    path, _, _ = states_path
    file = path / "acqu2s"
    file.write_text(file.read_text().replace("##$FnMODE= 4", "##$FnMODE= 0"))
    with pytest.raises(SpectroChemPyError, match="States acquisition"):
        scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    raw = scp.nmr.read(
        path, use_tdeff=True, indirect_dwell="IN[0]", quadrature="states"
    )
    assert raw.meta.quadrature_mode == "states"
    assert raw.shape == (2, 16, 16)


def test_missing_selection_status_is_not_silently_ignored(states_path):
    path, _, _ = states_path
    (path / "pdata/1/proc2s").unlink()
    with pytest.raises(SpectroChemPyError, match="Missing States processing status"):
        scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")


@pytest.mark.parametrize("mc2", [3, 4])
def test_processing_quadrature_fallback_is_respected(states_path, mc2):
    path, _, _ = states_path
    file = path / "acqu2s"
    file.write_text(file.read_text().replace("##$FnMODE= 4", "##$FnMODE= 0"))
    file = path / "pdata/1/proc2s"
    file.write_text(file.read_text().replace("##END", f"##$MC2= {mc2}\n##END"))
    if mc2 == 3:
        raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
        assert raw.meta.quadrature_mode == "states"
    else:
        with pytest.raises(SpectroChemPyError, match="States acquisition"):
            scp.nmr.read(
                path, use_tdeff=True, indirect_dwell="IN[0]", quadrature="states"
            )


@pytest.mark.parametrize("alias", ["Rb-87", "Rb87", "87rb"])
def test_stmas_ignores_indirect_isotope_alias(states_path, alias):
    path, _, _ = states_path
    expected = _workflow(
        scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]"), shift=0.1
    )[-1]
    file = path / "acqu2s"
    file.write_text(file.read_text().replace("##END", f"##$NUC1= <{alias}>\n##END"))
    raw = scp.nmr.read(path, use_tdeff=True, indirect_dwell="IN[0]")
    actual = _workflow(raw, shift=0.1)[-1]
    np.testing.assert_array_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.x.data, expected.x.data)
    np.testing.assert_array_equal(actual.y.data, expected.y.data)


@pytest.mark.parametrize("procno", [2, "2"])
@pytest.mark.parametrize("relative", [False, True])
def test_states_directory_selects_raw_ser_and_requested_status(
    states_path, procno, relative
):
    import shutil

    path, _, _ = states_path
    shutil.copytree(path / "pdata/1", path / "pdata/2")
    file = path / "pdata/2/procs"
    file.write_text(file.read_text().replace("129.999", "129.998"))
    file = path / "pdata/2/proc2s"
    file.write_text(file.read_text().replace("TDeff= 32", "TDeff= 30"))
    options = {"directory": path.parent} if relative else {}
    source = path.name if relative else path
    raw = scp.nmr.read(
        source, procno=procno, use_tdeff=True, indirect_dwell="IN[0]", **options
    )
    assert raw.shape == (2, 15, 16)
    assert raw.meta.acquisition["reference_frequency_mhz"] == (129.998, 129.998)
    shortcut = scp.nmr.read_stmas(path, procno=int(procno), use_tdeff=True)
    np.testing.assert_array_equal(raw.data, shortcut.data)
    np.testing.assert_array_equal(raw.x.data, shortcut.x.data)
    np.testing.assert_array_equal(raw.y.data, shortcut.y.data)


def test_states_selection_does_not_silently_read_processed_data(states_path):
    path, _, _ = states_path
    with pytest.raises(ValueError, match="raw SER"):
        scp.nmr.read(path, processed=True, use_tdeff=True, indirect_dwell="IN[0]")


def test_states_dataset_root_respects_experiment_and_processing_selection(states_path):
    import shutil

    path, _, _ = states_path
    target = path / "dataset" / "42"
    target.parent.mkdir()
    target.mkdir()
    for name in ("acqus", "acqu2s", "ser"):
        shutil.copy2(path / name, target / name)
    shutil.copytree(path / "pdata/1", target / "pdata/2")
    raw = scp.nmr.read(
        target.parent,
        expno=42,
        procno=2,
        use_tdeff=True,
        indirect_dwell="IN[0]",
    )
    expected = scp.nmr.read_stmas(target, procno=2, use_tdeff=True)
    np.testing.assert_array_equal(raw.data, expected.data)
    assert raw.shape == (2, 16, 16)
