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
def _powder_mesh(accuracy, hemisphere=False):
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
    directions = points.T / radius
    weights = radius**-3
    triangles = np.concatenate((lower, upper))
    if hemisphere:
        signs = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]])
        directions = (signs[:, :, None] * directions).transpose(1, 0, 2).reshape(3, -1)
        triangles = (triangles + np.arange(4)[:, None, None] * weights.size).reshape(
            -1, 3
        )
        weights = np.tile(weights, 4)
    return directions, weights, triangles


def _rank_two_tensor(directions, eta):
    """Return the five spherical tensor components in PULSAR convention."""
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
    return np.array([minus_two, minus_one, zero, -minus_one.conj(), minus_two.conj()])


def _csa_harmonics(directions, anisotropy_hz, eta, angle):
    """Project the first-order shift tensor onto rotor harmonics."""
    tensor = _rank_two_tensor(directions, eta)
    c, s = np.cos(angle), np.sin(angle)
    # Use the m >= 0 elements of the Wigner column k=0. The factors
    # for m > 0 include the conjugate negative harmonics.
    wigner = np.array(
        [(3 * c**2 - 1) / 2, -np.sqrt(3 / 2) * c * s, np.sqrt(3 / 8) * s**2]
    )
    wigner[1:] *= 2
    result = np.zeros((5, directions.shape[1]), dtype=complex)
    result[:3] = (np.sqrt(6) * anisotropy_hz / 2) * wigner[:, None] * tensor[2:]
    return result


def _rotated_csa_harmonics(directions, anisotropy_hz, eta, angle, rotation):
    """Project the CSA tensor in the common quadrupolar rotor frame."""
    eigenvalues = anisotropy_hz * np.array([-(1 + eta) / 2, -(1 - eta) / 2, 1])
    tensor = (rotation * eigenvalues) @ rotation.T
    x, y, z = directions
    sine = np.hypot(x, y)
    cos_phi = np.divide(x, sine, out=np.ones_like(x), where=sine > 0)
    sin_phi = np.divide(y, sine, out=np.zeros_like(y), where=sine > 0)
    polar = np.array([z * cos_phi, z * sin_phi, -sine])
    azimuthal = np.array([-sin_phi, cos_phi, np.zeros_like(z)])
    c, s = np.cos(angle), np.sin(angle)
    radial_t = tensor @ directions
    polar_t = tensor @ polar
    azimuthal_t = tensor @ azimuthal
    pp = (polar * polar_t).sum(axis=0)
    aa = (azimuthal * azimuthal_t).sum(axis=0)
    result = np.zeros((5, z.size), dtype=complex)
    # Expand n(t).T D n(t) for the field moving on its rotor cone.
    result[0] = c**2 * (directions * radial_t).sum(axis=0) + s**2 * (pp + aa) / 2
    result[1] = 2 * c * s * ((polar + 1j * azimuthal) * radial_t).sum(axis=0)
    result[2] = s**2 * ((pp - aa) / 2 + 1j * (polar * azimuthal_t).sum(axis=0))
    return result


def _validate_rotation(rotation):
    matrix = np.asarray(rotation)
    if matrix.shape != (3, 3) or matrix.dtype.kind not in "fiu":
        raise ValueError("csa_rotation must be a real 3 by 3 rotation matrix.")
    matrix = matrix.astype(float)
    if not np.isfinite(matrix).all() or not (
        np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-8)
        and np.linalg.det(matrix) > 0
    ):
        raise ValueError("csa_rotation must be a proper orthogonal matrix.")
    return matrix


def _quadrupolar_harmonics(directions, eta, angle, factors=(6, 3)):
    """Return second-order rank-two tensor products for one transition."""
    tensor = _rank_two_tensor(directions, eta)
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
    result = np.zeros_like(tensor)
    for order, (factor, column) in enumerate(
        zip(factors, columns, strict=True), start=1
    ):
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


def _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands):
    if spectral_width_hz <= 0 or spinning_hz < 0:
        raise ValueError("Use a positive spectral width and a nonnegative MAS rate.")
    if points < 2 or not 1 <= accuracy <= 64:
        raise ValueError("Use at least two points and a powder accuracy from 1 to 64.")
    if sidebands < 0 or (sidebands and not spinning_hz):
        raise ValueError(
            "Use a nonnegative sideband count, with a positive MAS rate for sidebands."
        )


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
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
    transitions="central",
):
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    values = (
        cq_hz,
        eta,
        larmor_hz,
        spectral_width_hz,
        shift_hz,
        spinning_hz,
        spin,
        csa_anisotropy_hz,
        csa_eta,
    )
    if not all(isinstance(value, Real) for value in values):
        raise TypeError("PULSAR parameters must be real scalars.")
    values = tuple(map(float, values))
    if not np.isfinite(values).all():
        raise ValueError("PULSAR parameters must be finite.")
    (
        cq_hz,
        eta,
        larmor_hz,
        spectral_width_hz,
        shift_hz,
        spinning_hz,
        spin,
        csa_anisotropy_hz,
        csa_eta,
    ) = values
    if transitions not in ("central", "satellites", "all"):
        raise ValueError("transitions must be 'central', 'satellites', or 'all'.")
    if transitions == "central":
        if spin < 1.5 or spin % 1 != 0.5:
            raise ValueError(
                "spin must be a half-integer greater than or equal to 3/2."
            )
    elif spin < 1 or spin % 1 not in (0, 0.5):
        raise ValueError(
            "spin must be an integer or half-integer greater than or equal to 1."
        )
    if transitions != "central" and spin > 16:
        raise ValueError("Quadrupolar multiplets support spin up to 16.")
    if transitions == "satellites" and spin % 1 == 0:
        raise ValueError("For an integer spin, use transitions='all'.")
    if not 0 <= eta <= 1 or not 0 <= csa_eta <= 1:
        raise ValueError("eta and csa_eta must lie between 0 and 1.")
    if larmor_hz <= 0:
        raise ValueError("Use a positive Larmor frequency.")
    _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands)

    rotation = None if csa_rotation is None else _validate_rotation(csa_rotation)
    # Misaligned tensors break octant reflection symmetry, but their
    # even-in-field shifts retain the powder average under axis inversion.
    hemisphere = bool(csa_anisotropy_hz and rotation is not None)
    directions, weights, triangles = _powder_mesh(accuracy, hemisphere)
    angle = np.deg2rad(54.736) if spinning_hz else 0
    if hemisphere:
        csa = _rotated_csa_harmonics(
            directions, csa_anisotropy_hz, csa_eta, angle, rotation
        )
    elif csa_anisotropy_hz:
        csa = _csa_harmonics(directions, csa_anisotropy_hz, csa_eta, angle)
    else:
        csa = 0.0

    # The historical quadrupolar tensor interchanges the transverse axes.
    # Use the physical EFG frame when the CSA orientation makes them observable.
    quadrupolar_eta = -eta if csa_anisotropy_hz else eta
    grid = (
        weights,
        triangles,
        shift_hz,
        spectral_width_hz,
        points,
        spinning_hz,
        accuracy,
        sidebands,
    )
    if transitions == "central":
        harmonics = _quadrupolar_harmonics(directions, quadrupolar_eta, angle)
        # For the central pair both spin factors scale together relative to I=3/2.
        spin_scale = 3 * (spin + 1.5) / (4 * spin**2 * (spin - 0.5))
        harmonics *= spin_scale * cq_hz**2 / (16 * larmor_hz)
        axis, mass = _powder_profile(harmonics + csa, *grid)
        return axis, mass / mass.sum()

    # First-order and second-order terms differ for each m -> m-1 pair.
    # The ideal excitation and transverse detection weight each pair by
    # the squared transition matrix element, up to a common factor.
    interaction = 3 * cq_hz / (4 * spin * (2 * spin - 1))
    first_order = _csa_harmonics(directions, -interaction, quadrupolar_eta, angle)
    second_order_scale = interaction**2 / larmor_hz
    spin_squared = spin * (spin + 1)
    spectrum = np.zeros(points)
    for index_from_top in range(int(2 * spin)):
        m = spin - index_from_top
        if transitions == "satellites" and m == 0.5:
            continue
        pair = m * (m - 1)
        factors = (
            -0.5 * (24 * pair - 4 * spin_squared + 9),
            -0.25 * (12 * pair - 4 * spin_squared + 6),
        )
        harmonics = _quadrupolar_harmonics(directions, quadrupolar_eta, angle, factors)
        harmonics *= second_order_scale
        harmonics += (2 * m - 1) * first_order + csa
        axis, mass = _powder_profile(harmonics, *grid)
        spectrum += (spin_squared - pair) * mass
    return axis, spectrum / spectrum.sum()


def _powder_profile(
    harmonics,
    weights,
    triangles,
    shift_hz,
    spectral_width_hz,
    points,
    spinning_hz,
    accuracy,
    sidebands,
):
    """Integrate the selected orders for a set of vertex harmonics."""
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
        return axis, spectrum

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
    return axis, spectrum


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
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
):
    """Simulate the central transition of an isolated spin as an NDDataset.

    Set the dimensionless spin quantum number with `spin` (3/2, 5/2, ...).
    The model uses second-order quadrupolar shifts and an ideal pulse with
    zero delay. Optional first-order CSA uses a signed reduced anisotropy
    in Hz and a separate asymmetry. A proper rotation matrix maps vectors
    from the CSA principal axes to the quadrupolar principal axes. Set `spinning_hz` to zero for a static powder or to a positive
    rate for MAS at 54.736 degrees. `sidebands` adds that many orders
    on each side of the central band. Intensities sum to one across all
    requested bands. The frequency coordinate labels the left edges of the
    PULSAR bins. The spectral window must contain all requested bands.
    Powder accuracy ranges from 1 to 64, and MAS quadrature is limited
    to 4096 rotor samples. The table of intensities across powder vertices
    and requested bands is limited to 64 MiB.
    """
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
        csa_anisotropy_hz=csa_anisotropy_hz,
        csa_eta=csa_eta,
        csa_rotation=csa_rotation,
    )
    return _as_dataset(axis, profile, "PULSAR central transition")


def simulate_quadrupolar(
    *,
    cq_hz,
    eta,
    larmor_hz,
    spectral_width_hz,
    spin=1.5,
    transitions="all",
    points=8192,
    shift_hz=0.0,
    spinning_hz=0.0,
    accuracy=16,
    sidebands=0,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
):
    """Simulate a selected set of quadrupolar transitions of one spin.

    Multiplet calculations support spin up to 16. Use all transitions
    for integer spins. Half-integer spins also
    support "central" or "satellites". Satellites include first- and
    second-order shifts. Intensities are normalized once over the
    requested transitions and MAS orders. See simulate_central_transition
    for frequency, powder mesh and CSA conventions.
    """
    axis, profile = _central_transition_profile(
        cq_hz=cq_hz,
        eta=eta,
        larmor_hz=larmor_hz,
        spectral_width_hz=spectral_width_hz,
        spin=spin,
        transitions=transitions,
        points=points,
        shift_hz=shift_hz,
        spinning_hz=spinning_hz,
        accuracy=accuracy,
        sidebands=sidebands,
        csa_anisotropy_hz=csa_anisotropy_hz,
        csa_eta=csa_eta,
        csa_rotation=csa_rotation,
    )
    return _as_dataset(axis, profile, "PULSAR quadrupolar spectrum")


def simulate_csa(
    *,
    anisotropy_hz,
    eta,
    spectral_width_hz,
    points=8192,
    shift_hz=0.0,
    spinning_hz=0.0,
    accuracy=16,
    sidebands=0,
):
    """Simulate first-order chemical shift anisotropy for an isolated spin.

    The signed reduced anisotropy is delta_zz - delta_iso in Hz.
    The asymmetry is (delta_yy - delta_xx) / anisotropy, in [0, 1].
    A zero MAS rate gives a static powder pattern. Otherwise, the rotor
    angle is 54.736 degrees and sidebands selects the orders on each side.
    The spectrum is normalized over the requested bands.
    """
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    values = (anisotropy_hz, eta, spectral_width_hz, shift_hz, spinning_hz)
    if not all(isinstance(value, Real) for value in values):
        raise TypeError("PULSAR parameters must be real scalars.")
    values = tuple(map(float, values))
    if not np.isfinite(values).all():
        raise ValueError("PULSAR parameters must be finite.")
    anisotropy_hz, eta, spectral_width_hz, shift_hz, spinning_hz = values
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie between 0 and 1.")
    _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands)

    directions, weights, triangles = _powder_mesh(accuracy)
    angle = np.deg2rad(54.736) if spinning_hz else 0
    harmonics = _csa_harmonics(directions, anisotropy_hz, eta, angle)
    axis, profile = _powder_profile(
        harmonics,
        weights,
        triangles,
        shift_hz,
        spectral_width_hz,
        points,
        spinning_hz,
        accuracy,
        sidebands,
    )
    return _as_dataset(
        axis, profile / profile.sum(), "PULSAR chemical shift anisotropy"
    )


def _as_dataset(axis, profile, name):
    from spectrochempy.core.dataset.coord import Coord
    from spectrochempy.core.dataset.nddataset import NDDataset

    dataset = NDDataset(profile, name=name)
    dataset.set_coordset(x=Coord(axis, units="Hz", title="frequency offset"))
    return dataset
