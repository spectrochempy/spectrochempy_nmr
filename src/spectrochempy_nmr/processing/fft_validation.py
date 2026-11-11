"""Read-only acquisition validation before any core FFT preparation."""

from importlib import import_module


def _require_fft_prevalidation():
    module = import_module("spectrochempy.processing.fft.fft")
    if getattr(module, "FFT_PREVALIDATION_API", 0) < 1:
        raise NotImplementedError(
            "Native NMR channels and unvalidated indirect profiles require a core "
            "with FFT prevalidation (FFT_PREVALIDATION_API >= 1). "
            "The released 1D NMR workflow remains available."
        )


def _fft_validate_input(dataset, *, dim, inv=False, **kwargs):
    """Reject unsupported transforms without modifying the original dataset."""
    if kwargs.get("convention") == "topspin-xf2" and (
        inv
        or dim != "x"
        or tuple(dataset.dims) != ("c", "y", "x")
        or dataset.meta.get("normalization_profile") != "bruker-ser-states-2d-v1"
    ):
        raise ValueError("TopSpin XF2 convention requires direct Bruker States F2 data")
    component = getattr(dataset.meta, "component_axis", None)
    if component == "c":
        if dim == component:
            raise NotImplementedError("The component axis is not a Fourier axis")
        if inv:
            raise NotImplementedError("Inverse channel reconstruction is not validated")
    if not inv:
        encodings = getattr(dataset.meta, "encoding", None)
        if encodings is not None:
            encoding = encodings[dataset.dims.index(dim)]
            if isinstance(encoding, str) and (
                encoding.startswith(("UNVALIDATED:", "PENDING-F2:"))
                or encoding == "CHANNELS"
            ):
                raise NotImplementedError(
                    f"NMR encoding {encoding!r} is not ready for this transform"
                )
