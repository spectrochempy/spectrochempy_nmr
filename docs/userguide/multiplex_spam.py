# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reconstructing Multiplex SPAM MQMAS

# %% [raw] raw_mimetype="text/restructuredtext"
# Multiplex SPAM stores each phase-cycle scan as a separate FID in the
# Bruker ``ser``. Use ``scp.nmr.read_multiplex_spam()`` to select one MQ
# coherence and reconstruct two States FIDs per indirect increment. The
# returned dataset is ready for the staged :doc:`mqmas` workflow.
#
# Acquisition and pulse program
# -----------------------------
#
# In ``MQMultiplex32`` and ``MQMultiplex52``, the first pulse excites MQ
# coherence and the delay ``d0`` lets it evolve. The second pulse converts
# the coherence and the selective third pulse mixes the SPAM pathways.
# The innermost loop records a pair of FIDs with the third pulse at 0°
# and 180°. It then resets the third phase and advances the first pulse
# phase. After a full phase cycle it resets the first phase and increments
# ``d0`` by ``IN[0]``.
#
# The SER order is indirect time, excitation phase, SPAM pair, then direct
# time. For spin 3/2 the cycle contains 7 excitation phases and 14 FIDs.
# For spin 5/2 it contains 11 phases and 22 FIDs. Bruker stores the number
# of FIDs per cycle in ``acqu3s:TD`` and the number of indirect increments
# in ``acqu2s:TD``. Retain the acquisition status files, the archived
# ``pulseprogram`` and the processing status file ``pdata/1/procs`` with the
# SER. The reader takes the indirect dwell from ``acqus:IN[0]``, the direct
# width from ``acqus:SW_h`` and the direct reference from ``procs:SF``.
#
# For each increment, the reader projects the difference and sum of the
# two SPAM scans onto sine and cosine at the selected coherence order.
# It recombines the projections with the relative complex phase used by
# the MSM AU program. The division by the number of excitation phases
# keeps the projection on a per-phase scale. The two complex outputs
# retain the cosine and sine information as States channels ``E1`` and
# ``E2``. Digital filter correction is handled later by XF2.
#
# Writing the paired FIDs as :math:`A_j,B_j` and the selected phase as
# :math:`\phi_j=2\pi p j/N`, the States channels for each indirect
# increment are
#
# .. math::
#
#    E_1 &= \frac{1}{N}\sum_j [(A_j-B_j)\cos\phi_j
#          -i(A_j+B_j)\sin\phi_j], \\
#    E_2 &= \frac{1}{N}\sum_j [(A_j-B_j)\sin\phi_j
#          +i(A_j+B_j)\cos\phi_j].
#
# Here :math:`N` is the number of excitation phases. The phase projection
# distinguishes harmonics modulo :math:`N`. The reader accepts the MQ
# orders supported by the archived pulse programs, 3Q for MQMultiplex32
# and 3Q or 5Q for MQMultiplex52.
#
# Reconstruct and process a 3Q spectrum
# -------------------------------------
#
# For an archived spin-3/2 ⁸⁷Rb acquisition, pass the experiment directory
# or its SER directly.
#
# The :doc:`gallery workflow </examples/gallery/nmr/plot_multiplex_spam>`
# runs the reconstruction, phases F2 and plots the RR plane. Inspect the
# F2 phase before interpreting the map. For spin 5/2, use ``spin=2.5``
# and select 3Q or 5Q consistently when reading and shearing.

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spectrochempy as scp
from nmrglue import bruker
from spectrochempy_nmr.processing.bruker_filter import resolve_bruker_delay
from spectrochempy_nmr.processing.mqmas import _mqecho_weights


path = os.environ.get("SCP_MSM_DATA")
if path:
    states = scp.nmr.read_multiplex_spam(path, coherence_order=3)
    xf2 = scp.nmr.Experiment(states).xf2(size=4096)
    row = xf2.data[0, 0]
    phc0 = -np.degrees(np.angle(row[np.argmax(np.abs(row))]))
    phased = scp.nmr.Experiment(xf2).xf2p(phc0=phc0)
    spectrum = scp.nmr.Experiment(phased).xfshear(size=256, spin=1.5, coherence_order=3)
    spectrum[0].squeeze().real.plot(method="contour")
else:
    print("Set SCP_MSM_DATA to the MQMultiplex32 experiment directory.")

# %% [raw] raw_mimetype="text/restructuredtext"
# If the SER contains a single interrupted cycle after the declared
# increments, ``allow_partial=True`` explicitly permits ignoring its
# nonzero traces and the remaining padding. Check
# ``states.meta.acquisition["multiplex_spam"]`` for the selection and the
# count of skipped traces.
#
# The pulse program and method are described by :cite:t:`malicki2005multiplex`.
#
#
# Isotropic referencing and echo weighting
# -----------------------------------------
#
# For the ⁸⁷Rb 3Q pathway, the shear coefficient with the positive FFT
# convention is −7/9 and the isotropic F1 scale divisor is 34/9.
# ``f1_shift`` positions the spectrum within the isotropic window.
# The historical RbNO₃/206 comparison used −7 ppm. Confirm the phase
# and referencing for each acquisition before choosing a shift.
#
# .. _mqecho-pst5-rbno3-comparison:
#
# Symmetric echo and antiecho apodization
# ---------------------------------------
#
# The two rows compare PST-5 and the selected RbNO₃/206 3Q component.
# Each row shows the same data with no window, with an echo-centered
# exponential, and with a symmetric average of the echo and antiecho
# exponentials. Contours within a row use the same levels relative
# to the unwindowed height of peak A.
#
# .. image:: /_static/mqecho-pst5-rbno3.png
#    :alt: Six contour maps showing the three apodization treatments for each sample
#
# The echo window is :math:`w_+(t_1,t_2)=\exp(-\pi L|t_2-|k|t_1|)`.
# The symmetric window is :math:`[w_+(t_1,t_2)+w_+(t_1,-t_2)]/2`.
# The time origin accounts for the digital group delay. We used
# :math:`|k|=19/12` and :math:`L=40\,\mathrm{Hz}` for PST-5, and
# :math:`|k|=7/9` and :math:`L=50\,\mathrm{Hz}` for RbNO₃/206.
# These are direct-dimension settings from the Bruker processing
# parameters. This variant applies the same window to both States
# branches before F2 transformation, with no additional F1 window.
# For RbNO₃, the 3Q selection precedes the window. The phase is held
# fixed within each acquisition. PST-5 shows positive RR contours,
# and RbNO₃ shows magnitude.
#
# A and B mark the same two resonances across each row. Widths are full
# widths at half maximum (FWHM) in ppm, measured on F1 and F2 cuts
# through the local 2D maxima. B/A is the ratio of their peak heights.
#
# .. csv-table::
#    :header: Acquisition, Window, "F1 FWHM A / B (ppm)", "F2 FWHM A / B (ppm)", "Height B/A"
#    :widths: 16, 24, 22, 22, 16
#
#    PST-5, None, 1.62 / 1.92, 1.89 / 2.22, 0.881
#    PST-5, Echo only (40 Hz), 1.70 / 2.00, 2.06 / 2.34, 0.910
#    PST-5, Symmetric (40 Hz), 1.91 / 2.24, 2.06 / 2.36, 0.925
#    RbNO₃/206, None, 0.87 / 0.90, 3.30 / 3.75, 0.761
#    RbNO₃/206, Echo only (50 Hz), 0.86 / 0.82, 5.55 / 4.32, 0.774
#    RbNO₃/206, Symmetric (50 Hz), 0.88 / 0.85, 5.55 / 4.29, 0.778
#
# On PST-5, symmetrization adds about 12% to both F1 widths relative
# to the echo-only treatment. On RbNO₃/206, echo-only weighting already
# accounts for nearly all of the F2 broadening. The F2 cut through
# RbNO₃ peak A can span two nearby lobes at half height, so its FWHM
# characterizes the cut as a whole.
#
# .. note::
#
#    Both acquisitions start after the conversion pulse. The predicted
#    antiecho center lies at negative physical time for nonzero t1.
#    The raw grid used for weighting begins at the negative digital group
#    delay, so it still contains the mirrored center for increments
#    1–13 of PST-5 and 1–8 of RbNO₃/206 (counting from zero).
#    For later increments only the tail overlaps the raw grid. Overlap
#    with this grid does not establish that a physical antiecho was acquired.

# %% [raw] raw_mimetype="text/restructuredtext"
# Reproduce the archived comparisons
# ----------------------------------
#
# Set ``SCP_RBNO3_DATA`` to the directory containing experiments 15 and 206.
# Make the public PST-5 experiment available in the SpectroChemPy data directory.
# To use a local copy, set ``SCP_MQMAS_SER`` to its SER file.
# The cells below replay the historical calculations used for the figures and
# the peak-width table above. They use the original phase and filtering conventions
# for that comparison. The short workflow above uses the current plugin API.
#
# .. _rbno3-comparison:
#
# .. image:: /_static/rbno3-mqmas-stmas.png
#    :alt: Conventional PST-5 MQMAS, RbNO₃ 3Q multiplex MQMAS and RbNO₃ STMAS

# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    <span class="fold-notebook-input" data-summary="Historical processing functions" hidden></span>

# %% jupyter={"source_hidden": true}
PST5_SER = os.environ.get(
    "SCP_MQMAS_SER", "nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/51/ser"
)


def read_traces(path):
    """Read the 32-bit little-endian storage used by these two acquisitions."""
    acqu = bruker.read_acqus_file(str(path))
    direct = acqu["acqus"]
    td = int(direct["TD"])
    if direct["BYTORDA"] != 0 or direct["DTYPA"] != 0 or td % 256:
        raise ValueError("Expected aligned little-endian int32 Bruker traces")
    words = np.fromfile(path / "ser", dtype="<i4").reshape(-1, td)
    traces = words[:, ::2] + 1j * words[:, 1::2]
    return acqu, traces


def select_multiplex_order(traces, order):
    """Select one coherence from (increment, phase step, SPAM, point)."""
    phase_count = traces.shape[1]
    if not 0 < order < (phase_count + 1) // 2:
        raise ValueError("The requested coherence aliases on this phase grid")
    phase = 2 * np.pi * order * np.arange(phase_count) / phase_count
    difference = traces[:, :, 0] - traces[:, :, 1]
    total = traces[:, :, 0] + traces[:, :, 1]
    cosine = np.einsum("p,ipt->it", np.cos(phase), difference)
    cosine -= 1j * np.einsum("p,ipt->it", np.sin(phase), total)
    sine = np.einsum("p,ipt->it", np.sin(phase), difference)
    sine += 1j * np.einsum("p,ipt->it", np.cos(phase), total)
    return np.stack((cosine, sine)) / phase_count


def fft_positive(data, size, axis):
    return size * np.fft.fftshift(np.fft.ifft(data, n=size, axis=axis), axes=axis)


def spectrum(
    branches, acqu, processing, *, order, scale, shear, shift, direct_size, phase=None
):
    """Return the magnitude spectrum on calibrated frequency axes."""
    direct = acqu["acqus"]
    sw2 = direct["SW_h"]
    dwell1 = direct["IN"][0]
    sf = processing["procs"]["SF"]
    carrier = direct["SFO1"]
    branches = bruker.remove_digital_filter(acqu, branches)
    f2 = fft_positive(branches, direct_size, -1)
    # A common zero-order phase makes the strongest F2 point real.
    if phase is None:
        phase = np.angle(f2.flat[np.argmax(np.abs(f2))])
    f2 *= np.exp(-1j * phase)
    states = f2[0].real + 1j * f2[1].real
    hz = (np.arange(direct_size) - direct_size // 2) * sw2 / direct_size
    t1 = np.arange(states.shape[0]) * dwell1
    ramp = 2 * np.pi * t1[:, None] * (shear * hz - shift * carrier * scale)
    result = fft_positive(states * np.exp(1j * ramp), 512, -2)
    reference = scale * (sf - shift * carrier * 1e-6)
    center = 1e6 * (scale * carrier - reference) / reference
    f1 = center - (np.arange(512) - 256) / (512 * dwell1 * reference)
    f2_ppm = 1e6 * (carrier - sf) / sf - hz / sf
    return scp.NDDataset(
        np.abs(result),
        name=order,
        coordset=[
            scp.Coord(
                f1, units="ppm", title="isotropic chemical shift", rounding=False
            ),
            scp.Coord(
                f2_ppm, units="ppm", title="direct chemical shift", rounding=False
            ),
        ],
    )


def read_multiplex(path):
    """Read experiment 206 and decode its two 3Q States branches."""
    mqmas, traces = read_traces(path)
    mqmas_proc = bruker.read_procs_file(str(path))
    if (
        mqmas["acqus"]["PULPROG"] != "MQMultiplex32"
        or mqmas["acqu3s"]["TD"] != 14
        or mqmas["acqu2s"]["TD"] != 32
        or traces.shape[0] != 32 * 14
    ):
        raise ValueError("Unexpected phase-cycle layout for RbNO3 experiment 206")
    multiplex = select_multiplex_order(traces.reshape(32, 7, 2, -1), 3)
    return mqmas, mqmas_proc, multiplex


def replay(root):
    """Reconstruct the two archived experiments without changing the SER files."""
    stmas, traces = read_traces(root / "15")
    stmas_proc = bruker.read_procs_file(str(root / "15"))
    usable = stmas_proc["proc2s"]["TDeff"]
    if usable != 110 or not np.any(traces[110]) or np.any(traces[111:]):
        raise ValueError("Unexpected trace count for RbNO3 experiment 15")
    stmas_branches = np.stack((traces[:usable:2], traces[1:usable:2]))

    mqmas, mqmas_proc, multiplex = read_multiplex(root / "206")
    return (
        spectrum(
            multiplex,
            mqmas,
            mqmas_proc,
            order="3Q MQMAS from multiplex",
            scale=34 / 9,
            shear=-7 / 9,
            shift=-7,
            direct_size=4096,
        ),
        spectrum(
            stmas_branches,
            stmas,
            stmas_proc,
            order="STMAS",
            scale=17 / 9,
            shear=-8 / 9,
            shift=0.1,
            direct_size=1024,
        ),
    )


def replay_pst5(raw=None, *, mqecho=None):
    """Return the conventional PST-5 MQMAS reference in its resonance region."""
    if raw is None:
        raw = scp.nmr.read(PST5_SER)
    xf2 = scp.nmr.Experiment(raw).xf2(size=4096, direct_points=2048, mqecho=mqecho)
    phased = scp.nmr.Experiment(xf2).xf2p(phc0=-58.0, phc1=0.0, pivot=0)
    spectrum = scp.nmr.Experiment(phased).xfshear(size=512, spin=2.5, f1_shift=-20.0)
    return spectrum[0, :, :].squeeze().real[52.0:25.0, 65.0:20.0]


def mqecho_window(shape, dwell1, sw2, delay, slope, lb, *, symmetric):
    """Weight the acquired time grid around the echo and optionally its mirror."""
    t1 = np.arange(shape[-2])[:, None] * dwell1
    t2 = (np.arange(shape[-1]) - delay) / sw2
    if symmetric:
        return _mqecho_weights(t1[:, 0], t2, slope, lb)
    return np.exp(-np.pi * lb * abs(t2 - slope * t1))


def half_height_width(axis, signal, peak):
    """Measure the contiguous half-height width around a local maximum."""
    half = signal[peak] / 2
    left = np.flatnonzero(signal[:peak] <= half)
    right = np.flatnonzero(signal[peak + 1 :] <= half)
    if not left.size or not right.size:
        return np.nan
    a, b = left[-1], peak + 1 + right[0]
    lo = np.interp(half, signal[a : a + 2], axis[a : a + 2])
    hi = np.interp(half, signal[b - 1 : b + 1][::-1], axis[b - 1 : b + 1][::-1])
    return abs(hi - lo)


def measure_peaks(dataset, centers):
    """Find the same resonances in each map and measure cuts through their maxima."""
    x, y, z = dataset.x.data, dataset.y.data, dataset.data
    peaks = []
    for xc, yc in centers:
        rows = np.flatnonzero(abs(y - yc) < 1.2)
        cols = np.flatnonzero(abs(x - xc) < 2.2)
        row, col = np.unravel_index(
            z[np.ix_(rows, cols)].argmax(), (rows.size, cols.size)
        )
        row, col = rows[row], cols[col]
        peaks.append(
            (
                x[col],
                y[row],
                z[row, col],
                half_height_width(y, z[:, col], row),
                half_height_width(x, z[row, :], col),
            )
        )
    return peaks


def compare_mqecho(root):
    """Recompute the three treatments and the peak measurements for each sample."""
    raw = scp.nmr.read(PST5_SER)
    sw1, sw2 = (float(v.to("Hz").magnitude) for v in raw.meta.sw_h[-2:])
    delay, _ = resolve_bruker_delay(raw, None)
    weighted = raw.copy()
    weighted.data = raw.data * mqecho_window(
        raw.shape, 1 / sw1, sw2, delay, 19 / 12, 40, symmetric=False
    )
    pst5_maps = [
        replay_pst5(raw),
        replay_pst5(weighted),
        replay_pst5(raw, mqecho=(19 / 12, 40)),
    ]

    acqu, processing, branches = read_multiplex(root / "206")
    direct = acqu["acqus"]
    f2 = fft_positive(bruker.remove_digital_filter(acqu, branches), 4096, -1)
    phase = np.angle(f2.flat[np.argmax(np.abs(f2))])
    rb_maps = []
    for weights in (
        1,
        *(
            mqecho_window(
                branches.shape,
                direct["IN"][0],
                direct["SW_h"],
                direct["GRPDLY"],
                7 / 9,
                50,
                symmetric=symmetric,
            )
            for symmetric in (False, True)
        ),
    ):
        rb_maps.append(
            spectrum(
                branches * weights,
                acqu,
                processing,
                order="3Q MQMAS from multiplex",
                scale=34 / 9,
                shear=-7 / 9,
                shift=-7,
                direct_size=4096,
                phase=phase,
            )
        )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for axrow, name, maps, centers, xlim, ylim, lb in (
        (axes[0], "PST-5", pst5_maps, [(36, 38), (40, 42)], (49, 30), (50, 33), 40),
        (
            axes[1],
            "RbNO₃/206",
            rb_maps,
            [(-27, -22.3), (-31.5, -26.3)],
            (-21, -38),
            (-19, -30),
            50,
        ),
    ):
        reference = measure_peaks(maps[0], centers)[0][2]
        labels = ("No window", f"Echo only ({lb} Hz)", f"Symmetric ({lb} Hz)")
        for ax, dataset, label in zip(axrow, maps, labels, strict=True):
            a, b = measure_peaks(dataset, centers)
            print(
                f"{name} | {label} | F1 A/B {a[3]:.2f}/{b[3]:.2f} ppm | "
                f"F2 A/B {a[4]:.2f}/{b[4]:.2f} ppm | B/A {b[2] / a[2]:.3f}"
            )
            ax.contour(
                dataset.x.data,
                dataset.y.data,
                dataset.data / reference,
                levels=[0.08, 0.15, 0.25, 0.4, 0.6, 0.8, 0.95],
            )
            ax.set(
                title=f"{name} · {label}",
                xlim=xlim,
                ylim=ylim,
                xlabel="F2 (ppm)",
                ylabel="Isotropic F1 (ppm)",
            )
            for peak_label, peak in zip("AB", (a, b), strict=True):
                ax.annotate(
                    peak_label, peak[:2], xytext=(7, 7), textcoords="offset points"
                )
    return fig


# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    <span class="fold-notebook-input" data-summary="Code for the three-spectrum comparison" hidden></span>

# %% jupyter={"source_hidden": true}
data_root = os.environ.get("SCP_RBNO3_DATA")
if data_root:
    pst5 = replay_pst5()
    multiplex, stmas = replay(Path(data_root))
    panels = (
        ("²⁷Al PST-5\nConventional 3Q MQMAS (RR)", pst5),
        ("⁸⁷Rb RbNO₃/206\n3Q MQMAS from multiplex", multiplex[:, -10.0:-60.0]),
        ("⁸⁷Rb RbNO₃/15\nSTMAS", stmas[:, -10.0:-60.0]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, (title, dataset) in zip(axes, panels, strict=True):
        ax.contour(
            dataset.x.data,
            dataset.y.data,
            dataset.data / dataset.data.max(),
            levels=[0.10, 0.18, 0.30, 0.50, 0.70, 0.90],
        )
        ax.set(title=title, xlabel="F2 (ppm)", ylabel="Isotropic F1 (ppm)")
        ax.invert_xaxis()
        ax.invert_yaxis()
    plt.show()
else:
    print("Set SCP_RBNO3_DATA to the directory containing RbNO3/15 and RbNO3/206.")

# %% [raw] raw_mimetype="text/restructuredtext"
# .. raw:: html
#
#    <span class="fold-notebook-input" data-summary="Code for the echo weighting comparison" hidden></span>

# %% jupyter={"source_hidden": true}
if data_root:
    mqecho_figure = compare_mqecho(Path(data_root))
    plt.show()
