"""NMR-specific post-processing for FFT results (axis labels, ppm conversion)."""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np

from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415


def _meta_dim_index(new, dim):
    """Resolve the metadata index matching the target dataset dimension."""
    if isinstance(dim, str):
        return new.dims.index(dim)
    axis, _ = new.get_axis(dim, negative_axis=False)
    return axis


def _apply_interop_fft_presentation(dataset, dim, *, ppm, physical, acquisition=None):
    if acquisition is None:
        acquisition = dataset.meta.get("acquisition", {})
    display = acquisition.get("fft_display_axis")
    if physical:
        width = acquisition.get("spectral_width_hz")
        if width is not None:
            size = dataset.shape[dataset.dims.index(dim)]
            bins = (np.arange(size, dtype=float) - size // 2) * (float(width) / size)
            dataset.coordset[dim] = Coord(
                bins, units="Hz", title="frequency", rounding=False, sigdigits=15
            )
        if acquisition.get("fft_initial_phase") is not None:
            from spectrochempy import ur  # noqa: PLC0415

            dataset.meta.phc0 = [0 * ur.degree]
            dataset.meta.phc1 = [0 * ur.degree]
        return
    if display is None:
        return
    from spectrochempy import ur  # noqa: PLC0415

    if (
        not isinstance(display, Mapping)
        or display.get("grid") != "inclusive-endpoints"
        or display.get("direction") != "decreasing"
    ):
        raise ValueError("unsupported interop FFT display grid")
    center = float(display["center_hz"])
    width = float(display["spectral_width_hz"])
    frequency = float(display["reference_frequency_mhz"])
    if not np.isfinite([center, width, frequency]).all() or min(width, frequency) <= 0:
        raise ValueError("invalid interop FFT display calibration")
    axis = dataset.coordset[dim]
    size = axis.size
    acquisition_width = acquisition.get("spectral_width_hz")
    if size > 1 and acquisition_width is not None:
        measured_width = np.ptp(axis.data) * size / (size - 1)
        if not np.isclose(measured_width, acquisition_width, rtol=1e-3, atol=0):
            return
    step = -width / max(size - 1, 1)
    if "physical_axis" in acquisition:
        coord = Coord(
            center + width / 2 + np.arange(size) * step,
            units="Hz",
            rounding=False,
            sigdigits=15,
        )
    else:
        coord = Coord.arange(size) * (step * ur.Hz) + (center + width / 2) * ur.Hz
    coord.name = dim
    coord.show_datapoints = False
    coord.meta["acquisition_frequency"] = frequency * ur.MHz
    nucleus = acquisition.get("nucleus") or ""
    match = re.fullmatch(r"(\d+)([a-zA-Z]+)", nucleus)
    label = f"^{{{match[1]}}}{match[2]}" if match else nucleus
    coord.title = f"${label}$ frequency"
    if ppm:
        coord.ito("ppm")
        coord.title = rf"$\delta\ {label}$"
    dataset.coordset[dim] = coord


def _fft_postprocess_result(new, dim=-1, inv=False, **kwargs):
    """
    Apply NMR-specific coordinate creation and unit conversion after FFT.

    This handler is registered on ``fft.postprocess_result`` and recreates
    the spectral axis when the dataset carries NMR metadata (``sfo1``,
    ``bf1``, ``sw_h``, ``nuc1``).
    """
    if getattr(new.meta, "component_axis", None) == "c":
        transformed = new.dims[-1]
        if transformed == "x" and new.meta.stage == "acquisition":
            new.meta.readonly = False
            new.meta.stage = "F2-spectrum"
            new.meta.readonly = True
        if new.meta.storage_profile == "bruker-states-trace-pairs":
            new.meta.readonly = False
            if transformed == "x":
                idx = new.dims.index("y")
                pending = new.meta.encoding[idx]
                if pending.startswith("PENDING-F2:"):
                    new.meta.encoding[idx] = "INDIRECT:" + pending.removeprefix(
                        "PENDING-F2:"
                    )
                    new.meta.stage = "F2-spectrum"
            elif transformed == "y":
                new.meta.stage = "spectrum"
                new.meta.component_labels = ("F1_R", "F1_I")
                new.coordset["c"] = Coord.arange(2, dims="c", labels=["F1_R", "F1_I"])
            new.meta.readonly = True

    if new.ndim == 1 and (
        new.meta.get("quadrature_mode") == "complex-1d"
        or new.meta.get("acquisition", {}).get("frequency_axis") is not None
    ):
        readonly = new.meta.readonly
        new.meta.readonly = False
        try:
            new.meta.data_domain = "time" if inv else "frequency"
            if not inv:
                axis = new.dims[_meta_dim_index(new, dim)]
                _apply_interop_fft_presentation(
                    new,
                    axis,
                    ppm=kwargs.get("ppm", True),
                    physical=kwargs.get("_interop_physical_fft", False),
                )
            reference = new.meta.get("frequency_reference")
            if inv and reference:
                size = new.shape[0]
                dwell = 1 / (size * reference["bin_width_hz"])
                coord = Coord(
                    np.arange(size) * dwell,
                    units="s",
                    rounding=False,
                    sigdigits=15,
                )
                coord.title = "time"
                new.set_coordset(coord)
                new.meta.last_frequency_reference = dict(reference)
                new.meta.frequency_reference = None
            elif inv:
                acquisition = new.meta.get("acquisition", {})
                if acquisition.get("fft_display_axis") is not None:
                    width = acquisition["spectral_width_hz"]
                    axis = new.dims[_meta_dim_index(new, dim)]
                    new.coordset[axis] = Coord(
                        np.arange(new.shape[0]) * (1e6 / width),
                        units="us",
                        title="time",
                    )
        finally:
            new.meta.readonly = readonly
        return new

    presentation = new.meta.get("fft_presentation")
    if presentation is not None and new.ndim == 2:
        index = _meta_dim_index(new, dim)
        dimension = new.dims[index]
        descriptor = presentation[dimension]
        if inv:
            size = new.shape[index]
            new.coordset[dimension] = Coord(
                np.arange(size) / descriptor["spectral_width_hz"],
                units="s",
                title="time",
                rounding=False,
                sigdigits=15,
            )
        else:
            _apply_interop_fft_presentation(
                new,
                dimension,
                ppm=kwargs.get("ppm", True),
                physical=False,
                acquisition=descriptor,
            )
        readonly = new.meta.readonly
        new.meta.readonly = False
        flags = list(new.meta.isfreq)
        flags[index] = not inv
        f2 = next(
            i
            for i, name in enumerate(new.dims)
            if presentation[name]["physical_axis"] == "F2"
        )
        new.meta.encoding = ["CONJUGATE-FFT"] * 2
        if not flags[f2]:
            new.meta.encoding[1 - f2] = "PENDING-F2:CONJUGATE-FFT"
        new.meta.data_domain = (
            "frequency" if all(flags) else "mixed" if any(flags) else "time"
        )
        new.meta.readonly = readonly
        return new

    # Only act when NMR metadata is present
    if not hasattr(new.meta, "sfo1") or new.meta.sfo1 is None:
        return new

    x = new.coordset[dim]
    size = x.size
    meta_idx = _meta_dim_index(new, dim)

    sfo1 = new.meta.sfo1[meta_idx]
    bf1 = new.meta.bf1[meta_idx]
    sf = new.meta.sf[meta_idx]
    sw = new.meta.sw_h[meta_idx]

    if new.meta.nuc1 is not None:
        nuc1 = new.meta.nuc1[meta_idx]
        m = re.match(r"([^a-zA-Z]+)([a-zA-Z]+)", nuc1)
        nucleus = "^{" + m[1] + "}" + m[2] if m is not None else ""
    else:
        nucleus = ""

    if not inv:
        # time → frequency
        sizem = max(size - 1, 1)
        deltaf = -sw / sizem
        first = sfo1 - sf - deltaf * sizem / 2.0

        newcoord = Coord.arange(size) * deltaf + first
        newcoord.show_datapoints = False
        newcoord.name = x.name
        newcoord.title = f"${nucleus}$ frequency"
        newcoord.ito("Hz")

        # Store acquisition frequency for ppm conversion context
        newcoord.meta["acquisition_frequency"] = bf1

        ppm = kwargs.get("ppm", True)
        if ppm:
            newcoord.ito("ppm")
            newcoord.title = rf"$\delta\ {nucleus}$"

        new.coordset[dim] = newcoord
    else:
        # frequency/ppm → time
        # Replace the generic core-created coordinate with a proper NMR time axis.
        deltat = (1.0 / sw.to("Hz")).to("us")

        newcoord = Coord.arange(size) * deltat
        newcoord.name = x.name
        newcoord.title = "time"
        newcoord.ito("us")
        new.coordset[dim] = newcoord

    if getattr(new.meta, "isfreq", None) is not None:
        new.meta.isfreq[meta_idx] = not inv

    return new
