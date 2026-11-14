"""
Process ⁸⁷Rb MQMAS and STMAS acquisitions
=========================================

Replay the ⁸⁷Rb 3Q MQMAS and STMAS workflows from Bruker SER files.
Set ``SCP_RBNO3_DATA`` to the directory containing experiments ``206``
and ``15``. Keep their acquisition parameters and ``pdata/1`` directories.
See :doc:`/userguide/stmas_multiplex` for the comparison with PST-5.

.. image:: /_static/rbno3-replay.png
   :alt: RbNO₃ multiplex 3Q MQMAS and STMAS maps from experiments 206 and 15
"""

# %%
# Read the stored traces
# ----------------------
# Experiment 206 stores each indirect increment as seven phase steps with
# a pair of SPAM scans at each step. Project the SPAM sums and differences
# onto the third harmonic of the phase cycle to recover the 3Q States pair.
# Experiment 15 stores States pairs
# directly. Its processing status specifies 110 usable traces.

import os
from pathlib import Path

import numpy as np
import spectrochempy as scp
from nmrglue import bruker


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


# %%
# Transform and reference the isotropic axis
# ------------------------------------------
# Transform F2, combine the States branches and apply the shear before
# transforming F1. The historical signed coherence is -3 for this
# spin-3/2 MQMAS experiment, yielding a scale divisor of 34/9.
# The shear sign below follows the positive FFT convention. STMAS
# uses the (1, 2) satellite pathway.


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


# %%
# Display the processed spectra
# -----------------------------
data_root = os.environ.get("SCP_RBNO3_DATA")
if data_root:
    mqmas, stmas = replay(Path(data_root))
    maps = [dataset[:, -10.0:-60.0] for dataset in (mqmas, stmas)]
    _ = scp.multiplot(
        [dataset / dataset.max() for dataset in maps],
        labels=["RbNO₃/206 · 3Q MQMAS", "RbNO₃/15 · STMAS"],
        ncol=2,
        method="contour",
        equal_aspect=False,
        figsize=(9, 4),
        levels=[0.10, 0.18, 0.30, 0.50, 0.70, 0.90],
        xlabel="F2 (ppm)",
        ylabel="Isotropic F1 (ppm)",
    )
else:
    print("Set SCP_RBNO3_DATA to the directory containing RbNO3/15 and RbNO3/206.")
