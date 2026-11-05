"""NMR FFT conventions on ordinary complex arrays and paired channels."""

from __future__ import annotations

import numpy as np


def _states_fft(data, tppi=False):
    """FFT transform according to STATES encoding."""
    from spectrochempy_nmr.processing.encoding_channels import _extract_components
    from spectrochempy_nmr.processing.encoding_channels import _prepare_states
    from spectrochempy_nmr.processing.encoding_channels import _pack_channels

    RR, RI, IR, II = _extract_components(data)
    sr, si = _prepare_states(RR, RI, IR, II)

    if tppi:
        sr[..., 1::2] = -sr[..., 1::2]
        si[..., 1::2] = -si[..., 1::2]

    fr = np.fft.fftshift(np.fft.fft(sr), -1)
    fi = np.fft.fftshift(np.fft.fft(si), -1)

    return _pack_channels(fr, fi)


def _echoanti_fft(data):
    """FFT transform according to ECHO-ANTIECHO encoding."""
    from spectrochempy_nmr.processing.encoding_channels import _extract_components
    from spectrochempy_nmr.processing.encoding_channels import _prepare_echoanti
    from spectrochempy_nmr.processing.encoding_channels import _pack_channels

    RR, RI, IR, II = _extract_components(data)
    c, s = _prepare_echoanti(RR, RI, IR, II)

    fc = np.fft.fftshift(np.fft.fft(c / 2.0), -1)
    fs = np.fft.fftshift(np.fft.fft(s / 2.0), -1)
    return _pack_channels(fc, fs)


def _tppi_fft(data):
    """FFT transform according to TPPI encoding."""
    from spectrochempy_nmr.processing.encoding_channels import _extract_components
    from spectrochempy_nmr.processing.encoding_channels import _prepare_tppi
    from spectrochempy_nmr.processing.encoding_channels import _pack_channels

    RR, RI, IR, II = _extract_components(data)
    sx, sy = _prepare_tppi(RR, RI, IR, II)

    sx[..., 1::2] = -sx[..., 1::2]
    sy[..., 1::2] = -sy[..., 1::2]

    fx = np.fft.fftshift(np.fft.fft(sx), -1)
    fy = np.fft.fftshift(np.fft.fft(sy), -1)

    return _pack_channels(fx, fy)


def _qf_fft(data):
    """FFT transform according to QF encoding."""
    return np.fft.fftshift(np.fft.fft(np.conjugate(data)), -1)


def _complex_fft(data):
    """Apply a standard complex FFT with NMR-centered output ordering."""
    return np.fft.fftshift(np.fft.fft(data), -1)


def _states_second_pass_fft(data, *, tppi=False):
    """
    Process the indirect dimension for STATES / STATES-TPPI data.

    After the first pass the two channels store the direct-dimension complex
    result as ``[Re(R), Im(R), Re(I), Im(I)]``.  The indirect complex axis is
    therefore represented by cross-pairs ``Re(R)+1j*Re(I)`` and
    ``Im(R)+1j*Im(I)``.  Transform those pairs along F1 and rebuild the
    two channels as the indirect complex results.
    """
    from spectrochempy_nmr.processing.encoding_channels import _extract_components
    from spectrochempy_nmr.processing.encoding_channels import _pack_channels

    RR, RI, IR, II = _extract_components(data)
    fr = RR + 1j * IR
    fi = RI + 1j * II

    if tppi:
        fr[..., 1::2] = -fr[..., 1::2]
        fi[..., 1::2] = -fi[..., 1::2]

    fr = np.fft.fftshift(np.fft.fft(np.conjugate(fr)), -1)
    fi = np.fft.fftshift(np.fft.fft(np.conjugate(fi)), -1)
    return _pack_channels(fr, fi)


def _echoanti_second_pass_fft(data):
    """
    Process the indirect dimension for ECHO-ANTIECHO data.

    After the first pass the channels store two complex subspectra directly.
    On real Bruker Echo-Antiecho data the indirect-dimension branch must be
    reconstructed with the conjugated orientation; using the plain complex FFT
    mirrors the F1 peak relative to the TopSpin processed reference.
    """
    from spectrochempy_nmr.processing.encoding_channels import _extract_components
    from spectrochempy_nmr.processing.encoding_channels import _pack_channels

    RR, RI, IR, II = _extract_components(data)
    fr = RR + 1j * RI
    fi = IR + 1j * II

    fr = np.fft.fftshift(np.fft.fft(np.conjugate(fr)), -1)
    fi = np.fft.fftshift(np.fft.fft(np.conjugate(fi)), -1)
    return _pack_channels(fr, fi)


def _fft_encoding_handler(data, encoding, **kwargs):
    """Dispatch explicit acquisition conventions on ordinary numeric arrays."""
    data = np.asarray(data)
    if data.dtype.kind not in "iufc":
        raise ValueError("NMR FFT encoding requires real or complex numeric data")
    encoding = "" if encoding is None else str(encoding)
    if encoding.startswith(("UNVALIDATED:", "PENDING-F2:")) or encoding == "CHANNELS":
        raise NotImplementedError(
            f"NMR encoding {encoding!r} is not ready for this transform"
        )
    if encoding.startswith("DIRECT:"):
        return _complex_fft(data)
    tppi = kwargs.get("tppi", False)
    original_axis = kwargs.get("original_axis", -1)
    if encoding.strip().lower() in {"", "none", "unknown"}:
        if data.ndim == 3 and data.shape[0] == 2:
            raise NotImplementedError("channel encoding metadata is missing")
        return _qf_fft(data)
    if original_axis != -1 or encoding.startswith("INDIRECT:"):
        if "QF" in encoding or "QSIM" in encoding or "DQD" in encoding:
            return _qf_fft(data)
        if "STATES" in encoding:
            return _states_second_pass_fft(data, tppi=tppi or "TPPI" in encoding)
        if "ECHO-ANTIECHO" in encoding:
            return _echoanti_second_pass_fft(data)
        raise NotImplementedError(
            f"Indirect NMR encoding {encoding!r} is not validated"
        )
    if "QF" in encoding or "QSIM" in encoding or "DQD" in encoding:
        return _qf_fft(data) if data.ndim == 1 else _complex_fft(data)
    if "QSEQ" in encoding:
        raise NotImplementedError("QSEQ NMR encoding is not yet implemented")
    if "STATES" in encoding:
        return _states_fft(data, tppi=tppi)
    if "TPPI" in encoding:
        return _complex_fft(data)
    if "ECHO-ANTIECHO" in encoding:
        return _echoanti_fft(data)
    raise NotImplementedError(f"NMR encoding {encoding!r} is not supported")
