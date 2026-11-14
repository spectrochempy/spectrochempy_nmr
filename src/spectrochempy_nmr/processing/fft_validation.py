"""Read-only acquisition validation before any core FFT preparation."""

from importlib import import_module

import numpy as np


def _positive_scalar(value):
    magnitude = getattr(value, "m", None)
    return (
        isinstance(magnitude, (int, float, np.integer, np.floating))
        and np.isfinite(magnitude)
        and magnitude > 0
    )


def _require_fft_prevalidation():
    module = import_module("spectrochempy.processing.fft.fft")
    if getattr(module, "FFT_PREVALIDATION_API", 0) < 1:
        raise NotImplementedError(
            "Native NMR channels and unvalidated indirect profiles require a core "
            "with FFT prevalidation (FFT_PREVALIDATION_API >= 1). "
            "The released 1D NMR workflow remains available."
        )


def _require_known_f1_quadrature(encoding):
    if encoding == "UNVALIDATED:AGILENT-STATES":
        raise NotImplementedError(
            "Agilent F1 acquisition mode is ambiguous. Reopen with "
            'read_agilent(..., quadrature="states") or quadrature="states-tppi".'
        )
    if encoding == "UNVALIDATED:BRUKER-F1":
        raise NotImplementedError(
            "Bruker F1 acquisition mode is undefined. Reopen with "
            'read_topspin(..., quadrature=mode), where mode is "qf", "states", "states-tppi", or '
            '"echo-antiecho".'
        )


def _fft_validate_input(dataset, *, dim, inv=False, **kwargs):
    """Reject unsupported transforms without modifying the original dataset."""
    if kwargs.get("convention") == "topspin-xf2":
        from .fft_postprocess import _topspin_xf2_reference  # noqa: PLC0415

        if (
            inv
            or dim != "x"
            or tuple(dataset.dims) != ("c", "y", "x")
            or dataset.meta.get("normalization_profile") != "bruker-ser-states-2d-v1"
        ):
            raise ValueError(
                "TopSpin XF2 convention requires direct Bruker States F2 data"
            )
        _topspin_xf2_reference(dataset)
    component = getattr(dataset.meta, "component_axis", None)
    if component == "c":
        if dim == component:
            raise NotImplementedError("The component axis is not a Fourier axis")
        if inv:
            raise NotImplementedError("Inverse channel reconstruction is not validated")
    if not inv:
        index = dataset.dims.index(dim)
        coord = dataset.coordset[dim] if dataset.coordset is not None else None
        widths = dataset.meta.get("sw_h")
        missing_ser_width = (
            dataset.meta.get("datatype") == "SER"
            and widths is not None
            and len(widths) == dataset.ndim
            and not _positive_scalar(widths[index])
        )
        if missing_ser_width or getattr(
            getattr(coord, "meta", None), "bruker_f1_width_missing", False
        ):
            raise ValueError(
                "Bruker F1 FFT requires an acquisition spectral width. "
                "Supply SW_h in Hz or restore SW and SFO1 in the indirect acquisition parameters."
            )
        if getattr(getattr(coord, "meta", None), "bruker_delay_list", False):
            raise ValueError(
                "Bruker F1 is a delay-list axis. Reopen without use_list to "
                "Fourier transform the indirect acquisition axis."
            )
        encodings = getattr(dataset.meta, "encoding", None)
        if encodings is not None:
            encoding = encodings[index]
            _require_known_f1_quadrature(encoding)
            if isinstance(encoding, str) and (
                encoding.startswith(("UNVALIDATED:", "PENDING-F2:"))
                or encoding == "CHANNELS"
            ):
                raise NotImplementedError(
                    f"NMR encoding {encoding!r} is not ready for this transform"
                )
