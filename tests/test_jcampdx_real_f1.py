"""Public real-F1 JCAMP processing against independent matrix DFTs."""
import sys
import numpy as np
import pytest
from nmrglue import interop
from spectrochempy_nmr import Experiment, read
from test_jcampdx_states import _write

pytestmark = pytest.mark.skipif(
    '"qseq"'
    not in (getattr(interop, "normalize_jcampdx_fid_2d", lambda: None).__doc__ or ""),
    reason="requires development JCAMP TPPI/QSEQ normalization",
)


def _read(path, mode, delay=0, direct="SIMULTANEOUS (DQD)"):
    from spectrochempy_nmr.processing.fft_validation import _require_fft_prevalidation

    try:
        _require_fft_prevalidation()
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    samples = np.random.default_rng(7063).integers(-100, 101, size=(8, 6, 2))
    raw = samples[..., 0] + 1j * samples[..., 1]
    _write(path, raw, mode.upper())
    path.write_text(path.read_text().replace("SIMULTANEOUS (DQD)", direct))
    record = "" if delay is None else f"##$GRPDLY={delay}\n"
    path.write_text(path.read_text().replace("##NTUPLES=", record + "##NTUPLES="))
    return read(
        path,
        protocol="interop",
        profile="jcampdx-fid-2d",
        normalize_options={
            "quadrature": mode,
            "conjugation": "conjugate",
            "indirect_dwell_s": 0.0005,
        },
    ), raw


@pytest.mark.parametrize("mode", ["tppi", "qseq"])
@pytest.mark.parametrize("delay,method", [(0, None), (2, "none"), (1.5, "phase")])
@pytest.mark.parametrize("direct", ["SIMULTANEOUS (DQD)", "SIMULTANEOUS"])
def test_real_f1_public_processing_and_scp_without_provider(
    tmp_path, monkeypatch, mode, delay, method, direct
):
    path = tmp_path / "real.dx"
    source, raw = _read(path, mode, delay, direct)
    options = {"digital_filter": method} if method else {}
    before = source.data.copy()
    if mode == "qseq":
        raw[2::4] *= -1
        raw[3::4] *= -1
    ny, nx = raw.shape
    direct = np.exp(2j * np.pi * np.outer(np.arange(nx) - nx // 2, np.arange(nx)) / nx)
    f2 = raw @ direct.T
    if method == "phase":
        f2 *= np.exp(-2j * np.pi * delay * np.arange(nx) / nx)
    indirect = np.exp(-2j * np.pi * np.outer(np.arange(ny), np.arange(ny)) / (2 * ny))
    a, b = indirect @ f2.real, indirect @ f2.imag
    expected = np.stack((a.real, b.real, -a.imag, -b.imag))
    result = Experiment(source).process(**options)
    np.testing.assert_allclose(result.data, expected, rtol=0, atol=1e-9)
    np.testing.assert_array_equal(source.data, before)
    np.testing.assert_allclose(source.y.data, np.arange(ny) / 2000, rtol=0, atol=1e-15)
    assert result.meta.component_labels == ("rr", "ri", "ir", "ii")
    anchors = {
        role: {"reference_frequency_mhz": freq, "point_index": 0, "ppm": ppm}
        for role, freq, ppm in [("F1", 100, 8), ("F2", 400, 4)]
    }
    referenced = Experiment(source).process(reference=anchors, **options)
    np.testing.assert_array_equal(referenced.data, result.data)
    for role, dim in [("F1", "y"), ("F2", "x")]:
        assert referenced.coordset[dim].data[0] == anchors[role]["ppm"]
    if delay:
        trace = result.meta.nmr_processing["scp_processing"]
        assert trace["requested"]["digital_filter"] == method
        assert trace["applied"]["digital_filter"]["applied"] is (method == "phase")
        assert trace["applied"]["digital_filter"]["group_delay_points"] == delay
        assert source.meta.acquisition["digital_filter"]["applied"] is False
    saved = tmp_path / "raw.scp"
    source.write(saved, confirm=False)
    restored = read(saved)
    monkeypatch.setitem(sys.modules, "nmrglue", None)
    restored.origin = "unknown provenance"
    restored.meta.normalization_profile = "unknown provenance"
    again = Experiment(restored).process(reference=anchors, **options)
    np.testing.assert_array_equal(again.data, referenced.data)
    np.testing.assert_array_equal(again.y.data, referenced.y.data)
    np.testing.assert_array_equal(again.x.data, referenced.x.data)


@pytest.mark.parametrize("mode", ["tppi", "qseq"])
@pytest.mark.parametrize("delay", [None, -1, 68])
def test_public_real_f1_does_not_ignore_unknown_or_nonzero_filter(
    tmp_path, mode, delay
):
    source, _ = _read(tmp_path / "filter.dx", mode, delay)
    before = source.data.copy()
    with pytest.raises(ValueError, match="digital-filter delay"):
        Experiment(source).process()
    np.testing.assert_array_equal(source.data, before)


@pytest.mark.parametrize("delay", [None, -1, 6, 68, float("nan"), float("inf")])
def test_explicit_unfiltered_tppi_still_requires_a_valid_delay(tmp_path, delay):
    source, _ = _read(tmp_path / "filter.dx", "tppi", 0)
    source.meta.acquisition["digital_filter"] = {
        "applied": False,
        "group_delay_points": delay,
    }
    with pytest.raises(ValueError, match="known unapplied digital-filter delay"):
        Experiment(source).process(digital_filter="none")


@pytest.mark.parametrize("method", ["time", "pknl", "unknown"])
def test_tppi_does_not_advertise_filter_correction(tmp_path, method):
    source, _ = _read(tmp_path / "filter.dx", "tppi", 2)
    with pytest.raises(ValueError, match="TPPI digital_filter"):
        Experiment(source).process(digital_filter=method)
