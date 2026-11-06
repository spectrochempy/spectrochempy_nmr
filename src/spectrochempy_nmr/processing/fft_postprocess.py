"""NMR-specific post-processing for FFT results (axis labels, ppm conversion)."""

from __future__ import annotations

import re

import numpy as np

from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415


def _meta_dim_index(new, dim):
    """Resolve the metadata index matching the target dataset dimension."""
    if isinstance(dim, str):
        return new.dims.index(dim)
    axis, _ = new.get_axis(dim, negative_axis=False)
    return axis


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

    if (
        new.ndim == 1
        and new.meta.get("normalization_profile") == "agilent-fid-1d-complex-v1"
    ):
        new.meta.data_domain = "time" if inv else "frequency"
        reference = new.meta.get("frequency_reference")
        if inv and reference:
            size = new.shape[0]
            dwell = 1 / (size * reference["bin_width_hz"])
            coord = Coord(
                np.arange(size) * dwell, units="s", rounding=False, sigdigits=15
            )
            coord.title = "time"
            new.set_coordset(coord)
            new.meta.last_frequency_reference = dict(reference)
            new.meta.frequency_reference = None
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
