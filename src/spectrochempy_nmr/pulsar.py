# Copyright (C) 2006-2013 Christian Fernandez, Jean-Paul Amoureux
# Jean-Paul Amoureux: Unité de Catalyse et Chimie du Solide, Lille, France.
# Christian Fernandez: Laboratoire Catalyse et Spectrochimie, Caen, France.
# Adapted from Masaï PULSAR (main.f90, ll.f90, util.f90).
# Distributed under the CeCILL-B license. See LICENSE.

"""PULSAR central-transition powder lines under an ideal pulse.

The historical Fortran provides numerical regression references. Scientific
validation also requires independent checks of the model and its conventions.
See the adjacent README.pulsar.md for the equations and implementation notes.
"""

from functools import lru_cache
from numbers import Real
from operator import index

import numpy as np


@lru_cache(maxsize=8)
def _powder_mesh(accuracy):
    """Build the PULSAR octant mesh and its inverse-cube vertex weights."""
    size = 5 * accuracy
    i, j = np.indices((size + 1, size + 1))
    inside = i + j <= size
    # Radially project the plane x + y + z = size onto the unit octant.
    # Its solid-angle Jacobian is proportional to radius**-3.
    points = np.array([size - i[inside] - j[inside], i[inside], j[inside]]).T
    radius = np.linalg.norm(points, axis=1)
    vertices = np.zeros_like(i)
    vertices[inside] = np.arange(radius.size)
    i, j = np.where(i + j < size)
    lower = np.column_stack((vertices[i, j], vertices[i + 1, j], vertices[i, j + 1]))
    interior = i + j < size - 1
    i, j = i[interior], j[interior]
    upper = np.column_stack(
        (vertices[i + 1, j], vertices[i, j + 1], vertices[i + 1, j + 1])
    )
    return points.T / radius, radius**-3, np.concatenate((lower, upper))


def _quadrupolar_harmonics(directions, eta, angle):
    """Return the rank-two tensor products for the central transition."""
    x, y, z = directions
    sine = np.sqrt(np.maximum(0.0, 1 - z**2))
    # Compute twice the azimuth from Cartesian coordinates. At the pole,
    # choose azimuth zero to fix the phase convention for the tensor.
    cos2alpha = np.divide(x * x - y * y, sine**2, out=np.ones_like(z), where=sine > 0)
    sin2alpha = np.divide(2 * x * y, sine**2, out=np.zeros_like(z), where=sine > 0)
    minus_two = sine**2 / 2 - eta * (1 + z**2) * cos2alpha / 6
    minus_two = minus_two - 1j * eta * z * sin2alpha / 3
    minus_one = -sine * z * (1 + eta * cos2alpha / 3)
    minus_one = minus_one - 1j * eta * sine * sin2alpha / 3
    zero = (3 * z**2 - 1 - eta * sine**2 * cos2alpha) / np.sqrt(6)
    tensor = np.array([minus_two, minus_one, zero, -minus_one.conj(), minus_two.conj()])

    c, s = np.cos(angle), np.sin(angle)
    root = np.sqrt(3 / 8)
    # Columns +1 and +2 of the rank-two Wigner matrix in PULSAR's convention.
    # The matching negative columns follow from the rank-two symmetry.
    columns = (
        [
            (1 - c) * s / 2,
            (1 + c) / 2 - c * c,
            2 * root * c * s,
            c * c - (1 - c) / 2,
            -(1 + c) * s / 2,
        ],
        [
            (1 - c) ** 2 / 4,
            (1 - c) * s / 2,
            root * s * s,
            (1 + c) * s / 2,
            (1 + c) ** 2 / 4,
        ],
    )
    result = np.zeros((5, z.size), dtype=complex)
    for order, (factor, column) in enumerate(zip((6, 3), columns), start=1):
        positive = np.asarray(column)
        negative = positive[::-1] * (-1.0) ** (np.arange(-2, 3) + order)
        for harmonic in range(5):
            for m in range(harmonic - 2, 3):
                # Multiplying tensor components adds their rotor harmonics.
                other = harmonic - m
                result[harmonic] += (
                    factor
                    * positive[m + 2]
                    * negative[other + 2]
                    * tensor[m + 2]
                    * tensor[other + 2]
                )
    # Combine the conjugate harmonics into a real part, leaving DC unchanged.
    result[1:] *= 2
    return result


def _deposit_triangles(axis, frequencies, weights, triangles):
    """Integrate each triangular frequency density over the PULSAR bins."""
    step = (axis[-1] - axis[0]) / (axis.size - 1)
    low, mid, high = np.sort((frequencies[triangles] - axis[0]) / step, axis=1).T
    first, last = low.astype(np.intp), high.astype(np.intp)
    masses = weights[triangles].sum(axis=1)
    spectrum = np.zeros_like(axis)

    # Flatten the bins covered by each triangle. Batch by bin count so that
    # wide triangles do not require a dense triangles-by-frequency array.
    counts = last - first + 1
    offsets = np.r_[0, counts.cumsum()]
    start = 0
    while start < counts.size:
        end = np.searchsorted(offsets, offsets[start] + 262_144, side="right") - 1
        end = max(start + 1, end)
        owners = np.repeat(np.arange(start, end), counts[start:end])
        bins = first[owners] + np.arange(offsets[start], offsets[end]) - offsets[owners]
        at_end = bins == last[owners]
        cdf = np.ones(bins.size)

        # Linear interpolation of frequency over a triangle gives a piecewise
        # quadratic CDF. Evaluate it only at interior bin edges. The ascending and
        # descending masks exclude the zero-width half of a flat-sided triangle.
        edges = bins + 1
        rising = ~at_end & (edges <= mid[owners])
        falling = ~at_end & ~rising
        row = owners[rising]
        cdf[rising] = (edges[rising] - low[row]) ** 2 / (
            (mid[row] - low[row]) * (high[row] - low[row])
        )
        row = owners[falling]
        cdf[falling] = 1 - (high[row] - edges[falling]) ** 2 / (
            (high[row] - mid[row]) * (high[row] - low[row])
        )

        # Difference successive CDF values, starting each triangle at zero.
        previous = np.r_[0.0, cdf[:-1]]
        previous[bins == first[owners]] = 0
        spectrum += np.bincount(
            bins, weights=(cdf - previous) * masses[owners], minlength=axis.size
        )
        start = end
    return spectrum


def _central_transition_profile(
    *,
    cq_hz,
    eta,
    larmor_hz,
    spectral_width_hz,
    points=8192,
    shift_hz=0.0,
    spinning_hz=0.0,
    accuracy=16,
    sidebands=0,
    spin=1.5,
):
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    values = (cq_hz, eta, larmor_hz, spectral_width_hz, shift_hz, spinning_hz, spin)
    if not all(isinstance(value, Real) for value in values):
        raise TypeError("PULSAR parameters must be real scalars.")
    values = tuple(map(float, values))
    if not np.isfinite(values).all():
        raise ValueError("PULSAR parameters must be finite.")
    cq_hz, eta, larmor_hz, spectral_width_hz, shift_hz, spinning_hz, spin = values
    if spin < 1.5 or spin % 1 != 0.5:
        raise ValueError("spin must be a half-integer greater than or equal to 3/2.")
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie between 0 and 1.")
    if larmor_hz <= 0 or spectral_width_hz <= 0 or spinning_hz < 0:
        raise ValueError(
            "Use positive Larmor and spectral widths and a nonnegative MAS rate."
        )
    if points < 2 or not 1 <= accuracy <= 64:
        raise ValueError("Use at least two points and a powder accuracy from 1 to 64.")

    if sidebands < 0 or (sidebands and not spinning_hz):
        raise ValueError(
            "Use a nonnegative sideband count, with a positive MAS rate for sidebands."
        )

    directions, weights, triangles = _powder_mesh(accuracy)
    angle = np.deg2rad(54.736) if spinning_hz else 0
    harmonics = _quadrupolar_harmonics(directions, eta, angle)
    # For the central pair, both second-order spin factors scale together
    # relative to I=3/2. The Wigner products themselves are independent of I.
    spin_scale = 3 * (spin + 1.5) / (4 * spin**2 * (spin - 0.5))
    harmonics *= spin_scale * cq_hz**2 / (16 * larmor_hz)
    frequencies = shift_hz + harmonics[0].real
    axis = np.linspace(-spectral_width_hz / 2, spectral_width_hz / 2, points)
    band_offset = sidebands * spinning_hz
    if (
        frequencies.min() - band_offset < axis[0]
        or frequencies.max() + band_offset > axis[-1]
    ):
        raise ValueError("The spectral window must contain the requested bands.")
    if not spinning_hz:
        spectrum = _deposit_triangles(axis, frequencies, weights, triangles)
        return axis, spectrum / spectrum.sum()

    rotor_samples = spectral_width_hz / spinning_hz
    if rotor_samples >= 4095.5:
        raise ValueError("This PULSAR prototype supports up to 4096 rotor samples.")
    count = max(5 * accuracy, int(rotor_samples + 0.5) + 1, 2 * sidebands + 1)
    orders = np.arange(1, 5)
    phase = 2 * np.pi * orders[:, None] * np.arange(count) / count
    # Integrating a frequency harmonic divides by its angular order times
    # the MAS rate. The 2*pi factors cancel when converting Hz to phase.
    rotor = np.exp(-1j * phase) / (orders[:, None] * spinning_hz)
    band_count = 2 * sidebands + 1
    intensity_bytes = frequencies.size * band_count * np.dtype(np.float64).itemsize
    if intensity_bytes > 64 * 1024**2:
        raise ValueError(
            "MAS band intensities exceed the 64 MiB limit. "
            "Reduce accuracy or sidebands."
        )
    band_orders = np.arange(-sidebands, sidebands + 1)
    band_intensity = np.empty((frequencies.size, band_count), dtype=np.float64)
    for start in range(0, frequencies.size, 128):
        block = slice(start, start + 128)
        modulation = (harmonics[1:, block].T @ rotor).imag
        signal = np.exp(-1j * modulation)
        # Ideal pulse and zero delay make the two PULSAR rotor sums
        # conjugates. Each band takes the squared magnitude of its Fourier
        # coefficient. The central-only path needs just the mean.
        if sidebands:
            coefficients = np.fft.fft(signal, axis=1)[:, band_orders] / count
        else:
            coefficients = signal.mean(axis=1, keepdims=True)
        band_intensity[block] = np.abs(coefficients) ** 2

    # Deposit raw band masses on the same grid to preserve relative areas.
    spectrum = np.zeros_like(axis)
    for band_order, intensity in zip(band_orders, band_intensity.T, strict=True):
        spectrum += _deposit_triangles(
            axis, frequencies + band_order * spinning_hz, weights * intensity, triangles
        )
    return axis, spectrum / spectrum.sum()


def simulate_central_transition(
    *,
    cq_hz,
    eta,
    larmor_hz,
    spectral_width_hz,
    points=8192,
    shift_hz=0.0,
    spinning_hz=0.0,
    accuracy=16,
    sidebands=0,
    spin=1.5,
):
    """Simulate the central transition of an isolated spin as an NDDataset.

    Set the dimensionless spin quantum number with `spin` (3/2, 5/2, ...).
    The model uses second-order quadrupolar shifts and an ideal pulse with
    zero delay. Set `spinning_hz` to zero for a static powder or to a positive
    rate for MAS at 54.736 degrees. `sidebands` adds that many orders
    on each side of the central band. Intensities sum to one across all
    requested bands. The frequency coordinate labels the left edges of the
    PULSAR bins. The spectral window must contain all requested bands.
    Powder accuracy ranges from 1 to 64, and MAS quadrature is limited
    to 4096 rotor samples. The table of intensities across powder vertices
    and requested bands is limited to 64 MiB.
    """
    from spectrochempy.core.dataset.coord import Coord
    from spectrochempy.core.dataset.nddataset import NDDataset

    axis, profile = _central_transition_profile(
        cq_hz=cq_hz,
        eta=eta,
        larmor_hz=larmor_hz,
        spectral_width_hz=spectral_width_hz,
        points=points,
        shift_hz=shift_hz,
        spinning_hz=spinning_hz,
        accuracy=accuracy,
        sidebands=sidebands,
        spin=spin,
    )
    dataset = NDDataset(profile, name="PULSAR central transition")
    dataset.set_coordset(x=Coord(axis, units="Hz", title="frequency offset"))
    return dataset
