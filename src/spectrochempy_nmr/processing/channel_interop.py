"""NMRPipe-convention FFTs for explicitly imported complex channel pairs."""

from collections.abc import Mapping
from numbers import Real

import numpy as np

from .qf_interop import _fft_stage


def _cross_pairs(f2):
    cross = np.empty_like(f2)
    cross[0].real, cross[0].imag = f2.real
    cross[1].real, cross[1].imag = f2.imag
    return cross


def _rance_kay(data):
    first, second = data
    return np.stack((first - second, 1j * (first + second)))


def process_states_tppi_fft(dataset, widths):
    return _process_channel_fft(dataset, widths, ("E1", "E2"), "States-TPPI")


def _bruker_delay(dataset, override):
    acquisition = dataset.meta.get("acquisition", {})
    operations = dataset.meta.get("applied_operations", ())
    for record in (acquisition, *operations):
        correction = record.get("digital_filter")
        if isinstance(correction, Mapping) and correction.get("applied"):
            raise ValueError("Bruker digital filter has already been corrected")

    if override is not None:
        return override, "caller"

    correction = acquisition.get("digital_filter")
    if (
        isinstance(correction, Mapping)
        and correction.get("group_delay_points") is not None
    ):
        return correction["group_delay_points"], "normalized acquisition"

    acqus = dataset.meta.get("source_metadata", {}).get("acqus", {})
    if isinstance(acqus, Mapping):
        delay = acqus.get("GRPDLY")
        if (
            isinstance(delay, Real)
            and not isinstance(delay, (bool, np.bool_))
            and delay > 0
        ):
            return delay, "GRPDLY"
        firmware, decimation = acqus.get("DSPFVS"), acqus.get("DECIM")
        if (
            all(
                isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                for value in (firmware, decimation)
            )
            and 0 <= firmware < 20
        ):
            from ..readers.read_topspin import bruker_dsp_table  # noqa: PLC0415

            delay = bruker_dsp_table.get(firmware, {}).get(decimation)
            if delay is not None:
                return delay, "DSPFVS/DECIM"
    raise ValueError("Unknown Bruker group delay; supply bruker_group_delay explicitly")


def process_echo_antiecho_fft(
    dataset, widths, *, digital_filter=None, group_delay=None
):
    if digital_filter is not None:
        group_delay, source = _bruker_delay(dataset, group_delay)
        if (
            isinstance(group_delay, (bool, np.bool_))
            or not isinstance(group_delay, Real)
            or not 0 < group_delay < dataset.shape[-1]
        ):
            raise ValueError("Bruker group delay must be positive and shorter than F2")
        group_delay = float(group_delay)

    result = _process_channel_fft(
        dataset,
        widths,
        ("first", "second"),
        "Echo/anti-echo",
        reconstruct=_rance_kay,
        digital_filter=digital_filter,
        group_delay=group_delay,
    )
    result.meta.echo_antiecho_reconstruction = "bruker-rance-kay"
    if digital_filter is not None:
        result.meta.bruker_digital_filter = {
            "method": digital_filter,
            "group_delay_points": group_delay,
            "delay_source": source,
            "removed_points": dataset.shape[-1] - result.shape[-1],
        }
    return result


def _process_channel_fft(
    dataset,
    widths,
    labels,
    mode,
    *,
    reconstruct=None,
    digital_filter=None,
    group_delay=None,
):
    import spectrochempy as scp  # noqa: PLC0415

    data = np.asarray(dataset.data)
    if (
        tuple(dataset.dims) != ("c", "y", "x")
        or data.shape[0] != 2
        or dataset.c is None
        or dataset.c.labels is None
        or tuple(dataset.c.labels) != labels
        or not np.iscomplexobj(data)
        or np.any(dataset.mask)
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            f"{mode} FFT requires the imported unmasked {'/'.join(labels)} channels"
        )
    if widths is None or len(widths) != 2:
        raise ValueError(f"{mode} FFT requires two spectral widths")

    for dim, size, width in zip(("y", "x"), data.shape[1:], widths, strict=True):
        axis = dataset.coordset[dim]
        if (
            size < 2
            or axis is None
            or axis.units != scp.ur.s
            or isinstance(width, (bool, np.bool_))
            or not isinstance(width, (int, float, np.integer, np.floating))
            or not np.isfinite(width)
            or width <= 0
            or not np.allclose(axis.data, np.arange(size) / width, rtol=1e-9, atol=0)
        ):
            raise ValueError(f"{mode} FFT requires the declared uniform time grids")
    if digital_filter == "time":
        from ..readers.read_topspin import _correct_digital_filter_time  # noqa: PLC0415

        if data.shape[-1] - int(group_delay) - 2 < 2:
            raise ValueError("Bruker group delay leaves fewer than two F2 points")
        data = _correct_digital_filter_time(data, group_delay)

    coordinates = {}
    for dim, size, width in zip(("y", "x"), data.shape[1:], widths, strict=True):
        axis = dataset.coordset[dim]
        hz = (size // 2 - np.arange(size)) * (width / size)
        coordinates[dim] = scp.Coord(
            hz, units="Hz", title=axis.title, rounding=False, sigdigits=15
        )

    if reconstruct is not None:
        data = reconstruct(data)
    f2 = _fft_stage(data, -1, direct=True)
    if digital_filter == "phase":
        # FT -df references the phase ramp to the first bin of the shifted FFT.
        bins = np.arange(f2.shape[-1]) / f2.shape[-1]
        f2 *= np.exp(-2j * np.pi * group_delay * bins)
    cross = _cross_pairs(f2)
    if reconstruct is None:
        cross[:, 1::2] *= -1  # States-TPPI uses NMRPipe FT -alt.
    f1 = _fft_stage(cross, -2, direct=True)
    planes = np.stack((f1[0].real, f1[1].real, f1[0].imag, f1[1].imag))
    labels = ("rr", "ri", "ir", "ii")
    result = scp.NDDataset(planes, dtype=planes.dtype, dims=["c", "y", "x"])
    result.set_coordset(c=scp.Coord(labels=list(labels)), **coordinates)
    result.meta.update(dict(dataset.meta.copy()))
    result.meta.data_domain = "frequency"
    result.meta.stage = "spectrum"
    result.meta.component_labels = labels
    result.meta.isfreq = [False, True, True]
    result.meta.td = list(planes.shape)
    if reconstruct is None:
        result.meta.tppi_alternation_state = "applied"
    return result
