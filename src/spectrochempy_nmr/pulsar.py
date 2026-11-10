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

from dataclasses import dataclass
from functools import lru_cache
from numbers import Real
from operator import index

import numpy as np


@lru_cache(maxsize=8)
def _powder_mesh(accuracy, hemisphere=False, full_sphere=False):
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
        if full_sphere:
            signs = np.concatenate((signs, -signs))
        directions = (signs[:, :, None] * directions).transpose(1, 0, 2).reshape(3, -1)
        offsets = np.arange(signs.shape[0])[:, None, None] * weights.size
        triangles = (triangles + offsets).reshape(-1, 3)
        weights = np.tile(weights, signs.shape[0])
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


def _validate_rotation(rotation, label="csa_rotation"):
    matrix = np.asarray(rotation)
    if matrix.shape != (3, 3) or matrix.dtype.kind not in "fiu":
        raise ValueError(f"{label} must be a real 3 by 3 rotation matrix.")
    matrix = matrix.astype(float)
    if not np.isfinite(matrix).all() or not (
        np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-8)
        and np.linalg.det(matrix) > 0
    ):
        raise ValueError(f"{label} must be a proper orthogonal matrix.")
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


def _real_scalars(*values):
    if not all(isinstance(value, Real) for value in values):
        raise TypeError("PULSAR parameters must be real scalars.")
    scalars = tuple(map(float, values))
    if not np.isfinite(scalars).all():
        raise ValueError("PULSAR parameters must be finite.")
    return scalars


def _rotor_angle(rotor_angle_deg, spinning_hz):
    (angle,) = _real_scalars(rotor_angle_deg)
    if not 0 <= angle <= 180:
        raise ValueError("rotor_angle_deg must be between 0 and 180.")
    return np.deg2rad(angle) if spinning_hz else 0.0


def _validate_probe_bandwidth(bandwidth):
    if bandwidth is None:
        return None
    (bandwidth,) = _real_scalars(bandwidth)
    if bandwidth <= 0:
        raise ValueError("probe_bandwidth_hz must be positive.")
    return bandwidth


def _probe_weights(frequency, weights, bandwidth):
    if bandwidth is None:
        return weights
    half = bandwidth / 2
    radius = np.hypot(frequency, half)
    ratio = np.divide(half, radius, out=np.ones_like(radius), where=radius > 0)
    return weights * ratio**2


def _normalize_profile(profile):
    total = profile.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            "The selected spectrum has no finite positive intensity to normalize."
        )
    return profile / total


def _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands):
    if spectral_width_hz <= 0 or spinning_hz < 0:
        raise ValueError("Use a positive spectral width and a nonnegative MAS rate.")
    if points < 2 or not 1 <= accuracy <= 64:
        raise ValueError("Use at least two points and a powder accuracy from 1 to 64.")
    if sidebands < 0 or (sidebands and not spinning_hz):
        raise ValueError(
            "Use a nonnegative sideband count, with a positive MAS rate for sidebands."
        )


@dataclass(frozen=True)
class PassiveJ:
    """Isotropic J coupling in Hz to an unpulsed spin."""

    spin: float
    j_hz: float


def _passive_offsets(partners):
    offsets = np.array([0.0])
    for partner in partners:
        if not isinstance(partner, PassiveJ):
            raise TypeError("j_partners must contain PassiveJ instances.")
        spin, coupling = _real_scalars(partner.spin, partner.j_hz)
        if not 0.5 <= spin <= 16 or spin % 0.5 != 0:
            raise ValueError(
                "Passive partner spin must be from 1/2 to 16 in steps of 1/2."
            )
        if not coupling:
            continue
        levels = int(2 * spin + 1)
        if offsets.size * levels > 256:
            raise ValueError("Passive J coupling supports at most 256 partner states.")
        projections = np.arange(levels) - spin
        offsets = (offsets[:, None] + coupling * projections).ravel()
    values, multiplicities = np.unique(offsets, return_counts=True)
    return tuple(zip(values, multiplicities / offsets.size, strict=True))


def _central_transition_profile(
    *,
    cq_hz,
    eta,
    larmor_hz,
    spectral_width_hz,
    points=8192,
    shift_hz=0.0,
    spinning_hz=0.0,
    rotor_angle_deg=54.736,
    accuracy=16,
    sidebands=0,
    spin=1.5,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
    transitions="central",
    j_partners=(),
    probe_bandwidth_hz=None,
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
    values = _real_scalars(*values)
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

    directions, weights, triangles, angle, csa, quadrupolar_eta = _powder_interactions(
        accuracy,
        spinning_hz,
        eta,
        csa_anisotropy_hz,
        csa_eta,
        csa_rotation,
        rotor_angle_deg=rotor_angle_deg,
    )
    grid = (
        weights,
        triangles,
        shift_hz,
        spectral_width_hz,
        points,
        spinning_hz,
        accuracy,
        sidebands,
        _passive_offsets(j_partners),
        _validate_probe_bandwidth(probe_bandwidth_hz),
    )
    if transitions == "central":
        harmonics = _quadrupolar_harmonics(directions, quadrupolar_eta, angle)
        # For the central pair both spin factors scale together relative to I=3/2.
        spin_scale = 3 * (spin + 1.5) / (4 * spin**2 * (spin - 0.5))
        harmonics *= spin_scale * cq_hz**2 / (16 * larmor_hz)
        axis, mass = _powder_profile(harmonics + csa, *grid)
        return axis, _normalize_profile(mass)

    spectrum = np.zeros(points)
    for m, strength, harmonics in _transition_harmonics(
        directions, cq_hz, quadrupolar_eta, angle, spin, larmor_hz, csa
    ):
        if transitions == "satellites" and m == 0.5:
            continue
        axis, mass = _powder_profile(harmonics, *grid)
        spectrum += strength * mass
    return axis, _normalize_profile(spectrum)


def _powder_interactions(
    accuracy,
    spinning_hz,
    eta,
    csa_anisotropy_hz,
    csa_eta,
    csa_rotation,
    *,
    force_hemisphere=False,
    rotor_angle_deg=54.736,
):
    rotation = None if csa_rotation is None else _validate_rotation(csa_rotation)
    # Misaligned tensors break octant reflection symmetry, but their
    # even-in-field shifts retain the powder average under axis inversion.
    rotated = bool(csa_anisotropy_hz and rotation is not None)
    hemisphere = force_hemisphere or rotated
    # A finite spinning pulse probes a time-ordered trajectory. With
    # rotated tensors the inverted rotor axis need not prepare the same state.
    directions, weights, triangles = _powder_mesh(
        accuracy, hemisphere, full_sphere=force_hemisphere and rotated
    )
    angle = _rotor_angle(rotor_angle_deg, spinning_hz)
    if rotated:
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
    return directions, weights, triangles, angle, csa, quadrupolar_eta


def _transition_harmonics(directions, cq_hz, eta, angle, spin, larmor_hz, csa):
    """Yield m, transverse strength and frequency harmonics in descending m."""
    if cq_hz == 0:
        harmonics = np.zeros((5, directions.shape[1]), dtype=complex) + csa
        for index_from_top in range(int(2 * spin)):
            m = spin - index_from_top
            yield m, spin * (spin + 1) - m * (m - 1), harmonics
        return

    interaction = 3 * cq_hz / (4 * spin * (2 * spin - 1))
    first_order = _csa_harmonics(directions, -interaction, eta, angle)
    second_order_scale = interaction**2 / larmor_hz
    spin_squared = spin * (spin + 1)
    for index_from_top in range(int(2 * spin)):
        m = spin - index_from_top
        pair = m * (m - 1)
        factors = (
            -0.5 * (24 * pair - 4 * spin_squared + 9),
            -0.25 * (12 * pair - 4 * spin_squared + 6),
        )
        harmonics = _quadrupolar_harmonics(directions, eta, angle, factors)
        harmonics *= second_order_scale
        harmonics += (2 * m - 1) * first_order + csa
        yield m, spin_squared - pair, harmonics


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
    passive_offsets=((0.0, 1.0),),
    probe_bandwidth_hz=None,
):
    """Integrate the selected orders for a set of vertex harmonics."""
    frequencies = shift_hz + harmonics[0].real
    axis = np.linspace(-spectral_width_hz / 2, spectral_width_hz / 2, points)
    band_offset = sidebands * spinning_hz
    if (
        frequencies.min() - band_offset + passive_offsets[0][0] < axis[0]
        or frequencies.max() + band_offset + passive_offsets[-1][0] > axis[-1]
    ):
        raise ValueError("The spectral window must contain the requested bands.")
    if not spinning_hz:
        spectrum = np.zeros_like(axis)
        for offset, population in passive_offsets:
            frequency = frequencies + offset
            detected = _probe_weights(frequency, weights, probe_bandwidth_hz)
            spectrum += population * _deposit_triangles(
                axis, frequency, detected, triangles
            )
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
        band = frequencies + band_order * spinning_hz
        for offset, population in passive_offsets:
            frequency = band + offset
            detected = _probe_weights(
                frequency, weights * intensity, probe_bandwidth_hz
            )
            spectrum += population * _deposit_triangles(
                axis, frequency, detected, triangles
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
    rotor_angle_deg=54.736,
    accuracy=16,
    sidebands=0,
    spin=1.5,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
    j_partners=(),
    probe_bandwidth_hz=None,
):
    """Simulate the central transition of an isolated spin as an NDDataset.

    Set the dimensionless spin quantum number with `spin` (3/2, 5/2, ...).
    The model uses second-order quadrupolar shifts and an ideal pulse with
    zero delay. Optional first-order CSA uses a signed reduced anisotropy
    in Hz and a separate asymmetry. A proper rotation matrix maps vectors
    from the CSA principal axes to the quadrupolar principal axes.
    A positive spinning rate uses rotor_angle_deg (54.736 degrees by
    default). sidebands adds that many orders
    on each side of the central band. Intensities sum to one across all
    requested bands. The frequency coordinate labels the left edges of the
    PULSAR bins. The spectral window must contain all requested bands.
    Powder accuracy ranges from 1 to 64, and MAS quadrature is limited
    to 4096 rotor samples. The table of intensities across powder vertices
    and requested bands is limited to 64 MiB. Optional probe_bandwidth_hz
    is the Lorentzian receiver bandwidth at half height.
    """
    axis, profile = _central_transition_profile(
        cq_hz=cq_hz,
        eta=eta,
        larmor_hz=larmor_hz,
        spectral_width_hz=spectral_width_hz,
        points=points,
        shift_hz=shift_hz,
        spinning_hz=spinning_hz,
        rotor_angle_deg=rotor_angle_deg,
        accuracy=accuracy,
        sidebands=sidebands,
        spin=spin,
        csa_anisotropy_hz=csa_anisotropy_hz,
        csa_eta=csa_eta,
        csa_rotation=csa_rotation,
        j_partners=j_partners,
        probe_bandwidth_hz=probe_bandwidth_hz,
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
    rotor_angle_deg=54.736,
    accuracy=16,
    sidebands=0,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
    j_partners=(),
    probe_bandwidth_hz=None,
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
        rotor_angle_deg=rotor_angle_deg,
        accuracy=accuracy,
        sidebands=sidebands,
        csa_anisotropy_hz=csa_anisotropy_hz,
        csa_eta=csa_eta,
        csa_rotation=csa_rotation,
        j_partners=j_partners,
        probe_bandwidth_hz=probe_bandwidth_hz,
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
    rotor_angle_deg=54.736,
    accuracy=16,
    sidebands=0,
    j_partners=(),
    probe_bandwidth_hz=None,
):
    """Simulate first-order chemical shift anisotropy for an isolated spin.

    The signed reduced anisotropy is delta_zz - delta_iso in Hz.
    The asymmetry is (delta_yy - delta_xx) / anisotropy, in [0, 1].
    A zero MAS rate gives a static powder pattern. Otherwise, the rotor
    angle defaults to 54.736 degrees and sidebands selects the orders on each side.
    The spectrum is normalized over the requested bands.
    """
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    values = (anisotropy_hz, eta, spectral_width_hz, shift_hz, spinning_hz)
    values = _real_scalars(*values)
    anisotropy_hz, eta, spectral_width_hz, shift_hz, spinning_hz = values
    if not 0 <= eta <= 1:
        raise ValueError("eta must lie between 0 and 1.")
    _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands)

    directions, weights, triangles = _powder_mesh(accuracy)
    angle = _rotor_angle(rotor_angle_deg, spinning_hz)
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
        _passive_offsets(j_partners),
        _validate_probe_bandwidth(probe_bandwidth_hz),
    )
    return _as_dataset(
        axis, _normalize_profile(profile), "PULSAR chemical shift anisotropy"
    )


def _spin_operators(spin):
    m = np.arange(spin, -spin - 1, -1)
    raising = np.diag(np.sqrt(spin * (spin + 1) - m[1:] * (m[1:] + 1)), k=1)
    return (raising + raising.T) / 2, (raising - raising.T) / (2j), np.diag(m)


def _evolve_density(density, hamiltonian_hz, duration_s):
    """Apply a constant Hamiltonian, in Hz, to a batch of density matrices."""
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_hz)
    phase = np.exp(-2j * np.pi * duration_s * eigenvalues)
    propagator = (eigenvectors * phase[..., None, :]) @ eigenvectors.conj().swapaxes(
        -1, -2
    )
    return propagator @ density @ propagator.conj().swapaxes(-1, -2)


def _evolve_pulse_step(density, rf, offsets, duration_s):
    # Adjacent differences of the diagonal must equal the offsets.
    levels = np.concatenate(
        (np.zeros_like(offsets[..., :1]), offsets.cumsum(axis=-1)), axis=-1
    )
    size = rf.shape[0]
    rows = np.arange(size)
    hamiltonian = np.broadcast_to(rf, (*levels.shape, size)).copy()
    hamiltonian[..., rows, rows] += levels
    return _evolve_density(density, hamiltonian, duration_s)


@dataclass(frozen=True)
class Pulse:
    """Rectangular RF pulse in seconds, Hz and degrees."""

    duration_s: float
    rf_hz: float
    phase_deg: float = 0
    offset_hz: float = 0
    partner_rf_hz: float = 0
    partner_phase_deg: float = 0
    partner_offset_hz: float = 0


@dataclass(frozen=True)
class IdealPulse:
    """Instantaneous rotation with flip angle and phase in degrees."""

    flip_deg: float
    phase_deg: float = 0
    partner_flip_deg: float = 0
    partner_phase_deg: float = 0


@dataclass(frozen=True)
class Delay:
    """Free evolution in seconds, optionally with J decoupling."""

    duration_s: float
    decouple_j: bool = False


@dataclass(frozen=True)
class Coherence:
    """Keep the matrix elements whose row minus column spin projection is order."""

    order: int


def _rf_axis(ix, iy, phase_deg):
    phase = np.deg2rad(phase_deg)
    return np.cos(phase) * ix + np.sin(phase) * iy


def _prepare_events(
    events,
    ix,
    iy,
    iz,
    spinning_hz,
    max_rf_step_s,
    t2_s,
    *,
    partner=None,
    partner_t2_s=None,
    coupling_offsets=None,
):
    projections = np.diag(iz)
    coherence_order = projections[:, None] - projections[None, :]
    if partner is None:
        partner_order = 0
    else:
        px, py, pz = partner
        partner_projections = np.diag(pz)
        partner_order = partner_projections[:, None] - partner_projections[None, :]
    prepared = []
    for event in events:
        if isinstance(event, Coherence):
            order = index(event.order)
            if abs(order) > projections.max() - projections.min():
                raise ValueError("Coherence order is outside the spin manifold.")
            mask = (coherence_order == order) & (partner_order == 0)
            if partner is None:
                prepared.append(("mask", mask))
            else:
                n_observed = int(projections.max() - projections.min() + 1)
                prepared.append(("partner_coherence", mask, n_observed))
            continue
        if isinstance(event, IdealPulse):
            flip, phase, partner_flip, partner_phase = _real_scalars(
                event.flip_deg,
                event.phase_deg,
                event.partner_flip_deg,
                event.partner_phase_deg,
            )
            axis = flip * _rf_axis(ix, iy, phase)
            if partner is not None:
                axis += partner_flip * _rf_axis(px, py, partner_phase)
            elif partner_flip:
                raise ValueError("A partner pulse requires a coupled spin.")
            prepared.append(("ideal", axis, 1 / 360))
            continue
        if isinstance(event, Delay):
            (duration,) = _real_scalars(event.duration_s)
            if duration < 0:
                raise ValueError("Delay duration must be nonnegative.")
            if event.decouple_j and coupling_offsets is None:
                raise ValueError("J decoupling requires an active coupled spin.")
            prepared.append(
                ("delay", duration, coupling_offsets if event.decouple_j else 0)
            )
        elif isinstance(event, Pulse):
            (
                duration,
                power,
                phase,
                offset,
                partner_power,
                partner_phase,
                partner_offset,
            ) = _real_scalars(
                event.duration_s,
                event.rf_hz,
                event.phase_deg,
                event.offset_hz,
                event.partner_rf_hz,
                event.partner_phase_deg,
                event.partner_offset_hz,
            )
            if duration <= 0 or power < 0 or partner_power < 0:
                raise ValueError("Use positive pulse duration and nonnegative RF.")
            rf = power * _rf_axis(ix, iy, phase)
            frequency_shift = offset * projections
            if partner is not None:
                rf += partner_power * _rf_axis(px, py, partner_phase)
                frequency_shift += partner_offset * partner_projections
            elif partner_power or partner_offset:
                raise ValueError("A partner pulse requires a coupled spin.")
            count = 1
            if spinning_hz:
                needed = duration / min(max_rf_step_s, 1 / (32 * spinning_hz))
                if needed > 4096:
                    raise ValueError(
                        "A MAS pulse supports up to 4096 propagation steps."
                    )
                count = max(1, int(np.ceil(needed)))
            frame = np.exp(2j * np.pi * duration * frequency_shift)
            frame = frame[:, None] * frame[None, :].conj()
            prepared.append(
                ("pulse", duration, rf, -np.diff(frequency_shift), count, frame)
            )
        else:
            raise TypeError(
                "Sequence events must be Pulse, IdealPulse, Delay or Coherence."
            )
        if t2_s is not None or partner_t2_s is not None:
            decay_a = 1.0 if t2_s is None else np.exp(-duration / t2_s)
            decay_b = 1.0 if partner_t2_s is None else np.exp(-duration / partner_t2_s)
            decay = np.where(
                coherence_order != 0, decay_a, np.where(partner_order != 0, decay_b, 1)
            )
            if partner is not None:
                decay = np.where(
                    (coherence_order != 0) & (partner_order != 0),
                    (decay_a + decay_b) / 2,
                    decay,
                )
            prepared.append(("mask", decay))
    if not prepared:
        raise ValueError("Supply at least one sequence event.")
    return prepared


def _phase_integral(terms, gamma, spinning_hz):
    orders = np.arange(1, 5)
    rotor = np.exp(-1j * orders[:, None] * gamma) / (orders[:, None] * spinning_hz)
    return -(terms @ rotor).imag.swapaxes(-1, -2)


def _propagate_sequence(offsets, terms, initial, events, spinning_hz, gamma):
    density = initial
    for kind, *args in events:
        if kind == "mask":
            density = density * args[0]
        elif kind == "partner_coherence":
            mask, n_observed = args
            n_partner = density.shape[-1] // n_observed
            blocks = density.reshape(
                *density.shape[:-2], n_observed, n_partner, n_observed, n_partner
            )
            marginal = np.trace(blocks, axis1=-4, axis2=-2) / n_observed
            preserved = np.einsum("ac,...bd->...abcd", np.eye(n_observed), marginal)
            density = np.where(mask, density, preserved.reshape(density.shape))
        elif kind == "ideal":
            axis, duration = args
            density = _evolve_density(density, axis, duration)
        elif kind == "delay":
            duration, decoupled = args
            free_offsets = offsets - decoupled
            if spinning_hz:
                next_gamma = gamma + 2 * np.pi * spinning_hz * duration
                phase = (
                    2 * np.pi * free_offsets[:, None, :] * duration
                    + _phase_integral(terms, next_gamma, spinning_hz)
                    - _phase_integral(terms, gamma, spinning_hz)
                )
                gamma = next_gamma
            else:
                phase = 2 * np.pi * free_offsets * duration
            level_phase = np.concatenate(
                (np.zeros_like(phase[..., :1]), phase.cumsum(axis=-1)), axis=-1
            )
            evolution = np.exp(-1j * level_phase)
            density = evolution[..., :, None] * density * evolution[..., None, :].conj()
        else:
            duration, rf, offset, count, frame = args
            step = duration / count
            if spinning_hz:
                order = np.arange(1, 5)
                step_phase = 2 * np.pi * spinning_hz * step
                for i in range(count):
                    midpoint = gamma + (i + 0.5) * step_phase
                    rotor = np.exp(-1j * order[:, None] * midpoint)
                    instantaneous = offsets[:, None, :] + (terms @ rotor).real.swapaxes(
                        -1, -2
                    )
                    density = _evolve_pulse_step(
                        density, rf, instantaneous - offset, step
                    )
                gamma = gamma + count * step_phase
            else:
                density = _evolve_pulse_step(density, rf, offsets - offset, step)
            density = density * frame
    return density, gamma


def _rf_rotor_count(spectral_width_hz, spinning_hz, accuracy, sidebands, size):
    samples = spectral_width_hz / spinning_hz
    if samples >= 4095.5:
        raise ValueError("This PULSAR prototype supports up to 4096 rotor samples.")
    count = max(5 * accuracy, int(samples + 0.5) + 1, 2 * sidebands + 1)
    # The aligned-tensor hemisphere reduction maps antipodal trajectories
    # onto the sampled rotor grid using a half-cycle phase shift.
    count += count % 2
    if count * size**2 > 131072:
        raise ValueError(
            "MAS rotor phases and spin dimension exceed the matrix batch limit."
        )
    return count


def _sequence_state_blocks(offsets, terms, initial, events, spinning_hz, count):
    """Yield the final density matrices and rotor angle for each powder block."""
    size = initial.shape[0]
    block_size = min(128, 131072 // (count * size**2))
    gamma = 2 * np.pi * np.arange(count) / count if spinning_hz else None
    for start in range(0, offsets.shape[0], block_size):
        block = slice(start, start + block_size)
        block_terms = terms[block] if spinning_hz else None
        batch = (
            (offsets[block].shape[0], count)
            if spinning_hz
            else (offsets[block].shape[0],)
        )
        density, gamma_end = _propagate_sequence(
            offsets[block],
            block_terms,
            np.broadcast_to(initial, (*batch, size, size)),
            events,
            spinning_hz,
            gamma,
        )
        yield block, density, gamma_end


def _powder_density(
    offsets,
    terms,
    initial,
    events,
    spinning_hz,
    count,
    weights,
    triangles,
    passive_offsets,
    passive_shift,
):
    # Every vertex enters once for each incident triangle, matching the
    # solid-angle weights used by the spectral integrator.
    vertex_weights = weights * np.bincount(triangles.ravel(), minlength=weights.size)
    density = np.zeros_like(initial, dtype=complex)
    for offset, population in passive_offsets:
        for block, states, _ in _sequence_state_blocks(
            offsets + offset * passive_shift, terms, initial, events, spinning_hz, count
        ):
            if spinning_hz:
                states = states.mean(axis=1)
            density += population * np.einsum(
                "v,vab->ab", vertex_weights[block], states
            )
    return density / vertex_weights.sum()


def _validate_output(output, sidebands):
    if output not in ("spectrum", "density"):
        raise ValueError("output must be 'spectrum' or 'density'.")
    if output == "density" and sidebands:
        raise ValueError("Sideband selection applies only to spectral output.")


def _sequence_signals(
    offsets,
    terms,
    strengths,
    initial,
    events,
    spinning_hz,
    count,
    sidebands,
    *,
    readout=None,
    readout_terms=None,
):
    """Average complex band amplitudes over rotor phase at preparation onset."""
    band_orders = np.arange(-sidebands, sidebands + 1)
    amplitudes = np.empty(
        (offsets.shape[0], strengths.size, band_orders.size), dtype=complex
    )
    for block, density, gamma_end in _sequence_state_blocks(
        offsets, terms, initial, events, spinning_hz, count
    ):
        if readout is None:
            coherence = np.diagonal(density, offset=1, axis1=-2, axis2=-1)
        else:
            coherence = density[..., readout[0], readout[1]]
        signal = -1j * np.sqrt(strengths) * coherence
        if not spinning_hz:
            amplitudes[block, :, 0] = signal
            continue
        detected_terms = (terms if readout_terms is None else readout_terms)[block]
        wave = np.exp(
            1j * _phase_integral(detected_terms, gamma_end, spinning_hz)
        ).swapaxes(-1, -2)
        # Opposite-sign Fourier sums for detection and preparation.
        detection = np.fft.fft(wave, axis=-1)[..., band_orders] / count
        preparation = np.fft.ifft(signal.swapaxes(-1, -2) * wave.conj(), axis=-1)[
            ..., band_orders
        ]
        amplitudes[block] = detection * preparation
    return amplitudes


def simulate_rectangular_pulse(
    *,
    spin,
    larmor_hz,
    spectral_width_hz,
    rf_hz,
    duration_s,
    phase_deg=0.0,
    cq_hz=0.0,
    eta=0.0,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
    shift_hz=0.0,
    points=8192,
    accuracy=16,
    spinning_hz=0.0,
    rotor_angle_deg=54.736,
    sidebands=0,
    max_rf_step_s=2e-6,
    probe_bandwidth_hz=None,
):
    """Simulate a powder after one finite rectangular RF pulse.

    The pulse is on the simulation carrier. Frequencies are in Hz and
    duration_s is in seconds. Thermal Iz is the initial density matrix.
    The returned complex spectrum is scaled to unit total absorption for
    an ideal on-carrier 90 degree x pulse. Actual excitation losses and
    phase are retained. With MAS, midpoint propagation uses time steps
    no longer than max_rf_step_s or 1/32 of a rotor period.
    """
    return simulate_pulse_sequence(
        [Pulse(duration_s, rf_hz, phase_deg)],
        spin=spin,
        larmor_hz=larmor_hz,
        spectral_width_hz=spectral_width_hz,
        cq_hz=cq_hz,
        eta=eta,
        csa_anisotropy_hz=csa_anisotropy_hz,
        csa_eta=csa_eta,
        csa_rotation=csa_rotation,
        shift_hz=shift_hz,
        points=points,
        accuracy=accuracy,
        spinning_hz=spinning_hz,
        rotor_angle_deg=rotor_angle_deg,
        sidebands=sidebands,
        max_rf_step_s=max_rf_step_s,
        probe_bandwidth_hz=probe_bandwidth_hz,
    )


def simulate_pulse_sequence(
    events,
    *,
    spin,
    larmor_hz,
    spectral_width_hz,
    cq_hz=0.0,
    eta=0.0,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
    shift_hz=0.0,
    points=8192,
    accuracy=16,
    spinning_hz=0.0,
    rotor_angle_deg=54.736,
    sidebands=0,
    max_rf_step_s=2e-6,
    t2_s=None,
    receiver_phase_deg=0.0,
    transitions="all",
    j_partners=(),
    output="spectrum",
    probe_bandwidth_hz=None,
):
    """Simulate a sequence of pulses, delays and coherence selections.

    Each event uses the simulation carrier as its phase reference.
    Optional t2_s damps off-diagonal terms after each timed event.
    receiver_phase_deg multiplies the complex signal by exp(i*phase).
    The ideal on-carrier x90 pulse defines unit absorption for the
    selected transitions. RF always propagates the complete manifold.
    output="density" returns the powder-averaged deviation density
    matrix at the end of preparation, before receiver phase.
    """
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    (
        spin,
        larmor_hz,
        spectral_width_hz,
        cq_hz,
        eta,
        csa_anisotropy_hz,
        csa_eta,
        shift_hz,
        spinning_hz,
        max_rf_step_s,
        receiver_phase_deg,
    ) = _real_scalars(
        spin,
        larmor_hz,
        spectral_width_hz,
        cq_hz,
        eta,
        csa_anisotropy_hz,
        csa_eta,
        shift_hz,
        spinning_hz,
        max_rf_step_s,
        receiver_phase_deg,
    )
    if not 0.5 <= spin <= 16 or spin % 0.5 != 0 or (spin == 0.5 and cq_hz):
        raise ValueError(
            "Use spin from 1/2 to 16 in steps of 1/2 and Cq=0 for spin 1/2."
        )
    if larmor_hz <= 0:
        raise ValueError("Use a positive Larmor frequency.")
    if t2_s is not None:
        (t2_s,) = _real_scalars(t2_s)
        if t2_s <= 0:
            raise ValueError("t2_s must be positive.")
    if not 0 <= eta <= 1 or not 0 <= csa_eta <= 1:
        raise ValueError("eta and csa_eta must lie between 0 and 1.")
    if max_rf_step_s <= 0:
        raise ValueError("max_rf_step_s must be positive.")
    _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands)
    _validate_output(output, sidebands)
    bandwidth = _validate_probe_bandwidth(probe_bandwidth_hz)
    directions, weights, triangles, angle, csa, quadrupolar_eta = _powder_interactions(
        accuracy,
        spinning_hz,
        eta,
        csa_anisotropy_hz,
        csa_eta,
        csa_rotation,
        force_hemisphere=bool(spinning_hz),
        rotor_angle_deg=rotor_angle_deg,
    )
    if transitions not in ("central", "satellites", "all"):
        raise ValueError("transitions must be 'central', 'satellites', or 'all'.")
    if transitions != "all" and (spin < 1.5 or spin % 1 != 0.5):
        raise ValueError("Select all transitions for integer spins and spin 1/2.")

    transition_count = int(2 * spin)
    selected = np.ones(transition_count, dtype=bool)
    if transitions != "all":
        central = spin - np.arange(transition_count) == 0.5
        selected = central if transitions == "central" else ~central
    offsets = np.empty((directions.shape[1], transition_count))
    strengths = np.empty(transition_count)
    if spinning_hz:
        count = _rf_rotor_count(
            spectral_width_hz, spinning_hz, accuracy, sidebands, transition_count + 1
        )
        order_count = 2 * sidebands + 1 if output == "spectrum" else 0
        # Bound the two persistent complex arrays before allocating either.
        storage = offsets.size * (4 + order_count) * np.dtype(complex).itemsize
        if storage > 64 * 1024**2:
            raise ValueError(
                "MAS pulse work arrays exceed 64 MiB. Reduce accuracy or sidebands."
            )
        harmonics = np.empty((*offsets.shape, 4), dtype=complex)
    for i, (_, strength, terms) in enumerate(
        _transition_harmonics(
            directions, cq_hz, quadrupolar_eta, angle, spin, larmor_hz, csa
        )
    ):
        offsets[:, i] = shift_hz + terms[0].real
        strengths[i] = strength
        if spinning_hz:
            harmonics[:, i] = terms[1:].T
    ix, iy, iz = _spin_operators(spin)
    prepared = _prepare_events(events, ix, iy, iz, spinning_hz, max_rf_step_s, t2_s)
    passive_offsets = _passive_offsets(j_partners)
    if output == "density":
        return _powder_density(
            offsets,
            harmonics if spinning_hz else None,
            iz,
            prepared,
            spinning_hz,
            count if spinning_hz else 1,
            weights,
            triangles,
            passive_offsets,
            1,
        )

    axis = np.linspace(-spectral_width_hz / 2, spectral_width_hz / 2, points)
    band_offset = sidebands * spinning_hz
    if (
        offsets[:, selected].min() - band_offset + passive_offsets[0][0] < axis[0]
        or offsets[:, selected].max() + band_offset + passive_offsets[-1][0] > axis[-1]
    ):
        raise ValueError("The spectral window must contain the requested transitions.")
    receiver = np.exp(1j * np.deg2rad(receiver_phase_deg))
    profile = np.zeros(points, dtype=complex)
    for offset, population in passive_offsets:
        shifted = offsets + offset
        signals = _sequence_signals(
            shifted,
            harmonics if spinning_hz else None,
            strengths,
            iz,
            prepared,
            spinning_hz,
            count if spinning_hz else 1,
            sidebands,
        )
        signals *= receiver * population
        for i in np.flatnonzero(selected):
            for j, order in enumerate(range(-sidebands, sidebands + 1)):
                frequency = shifted[:, i] + order * spinning_hz
                signal = _probe_weights(
                    frequency, weights * signals[:, i, j], bandwidth
                )
                profile.real += _deposit_triangles(
                    axis, frequency, signal.real, triangles
                )
                profile.imag += _deposit_triangles(
                    axis, frequency, signal.imag, triangles
                )
    ideal_signal = strengths[selected].sum() / 2
    powder_mass = weights[triangles].sum()
    return _as_dataset(
        axis, profile / (ideal_signal * powder_mass), "PULSAR pulse sequence"
    )


@dataclass(frozen=True)
class SpinSite:
    """Secular single-spin parameters in Hz and principal-frame rotations."""

    spin: float
    larmor_hz: float
    shift_hz: float = 0
    cq_hz: float = 0
    eta: float = 0
    csa_anisotropy_hz: float = 0
    csa_eta: float = 0
    csa_rotation: object = None
    efg_rotation: object = None
    t2_s: float | None = None


def _site_parameters(site):
    if not isinstance(site, SpinSite):
        raise TypeError("observed and partner must be SpinSite instances.")
    spin, larmor, shift, cq, eta, csa, csa_eta = _real_scalars(
        site.spin,
        site.larmor_hz,
        site.shift_hz,
        site.cq_hz,
        site.eta,
        site.csa_anisotropy_hz,
        site.csa_eta,
    )
    if not 0.5 <= spin <= 16 or spin % 0.5 != 0 or (spin == 0.5 and cq):
        raise ValueError(
            "Use spin from 1/2 to 16 in steps of 1/2 and Cq=0 for spin 1/2."
        )
    if larmor <= 0 or not 0 <= eta <= 1 or not 0 <= csa_eta <= 1:
        raise ValueError(
            "Use positive Larmor frequency and asymmetries between 0 and 1."
        )
    t2 = None
    if site.t2_s is not None:
        (t2,) = _real_scalars(site.t2_s)
        if t2 <= 0:
            raise ValueError("Site t2_s must be positive.")
    efg_rotation = (
        None
        if site.efg_rotation is None
        else _validate_rotation(site.efg_rotation, "efg_rotation")
    )
    csa_rotation = (
        None if site.csa_rotation is None else _validate_rotation(site.csa_rotation)
    )
    return spin, larmor, shift, cq, eta, csa, csa_eta, efg_rotation, csa_rotation, t2


def _polar_frame(directions):
    x, y, z = directions
    sine = np.hypot(x, y)
    cos_phi = np.divide(x, sine, out=np.ones_like(x), where=sine > 0)
    sin_phi = np.divide(y, sine, out=np.zeros_like(y), where=sine > 0)
    return np.array([z * cos_phi, z * sin_phi, -sine]), np.array(
        [-sin_phi, cos_phi, np.zeros_like(z)]
    )


def _site_transitions(directions, angle, parameters, harmonic_count):
    spin, larmor, shift, cq, eta, csa, csa_eta, efg_rotation, csa_rotation, _ = (
        parameters
    )
    if efg_rotation is None:
        local = directions
    else:
        local = efg_rotation.T @ directions
    if csa and csa_rotation is not None:
        shielding = _rotated_csa_harmonics(local, csa, csa_eta, angle, csa_rotation)
    elif csa:
        shielding = _csa_harmonics(local, csa, csa_eta, angle)
    else:
        shielding = 0.0
    count = int(2 * spin)
    result = np.empty((directions.shape[1], count, harmonic_count), dtype=complex)
    strengths = np.empty(count)
    # Every site uses the physical EFG frame when relative rotations are meaningful.
    for i, (_, strength, terms) in enumerate(
        _transition_harmonics(local, cq, -eta, angle, spin, larmor, shielding)
    ):
        result[:, i] = terms[:harmonic_count].T
        strengths[i] = strength
    result[..., 0] += shift
    if efg_rotation is not None and harmonic_count > 1:
        # Express the local rotor phase origin in the common frame.
        polar, _ = _polar_frame(directions)
        local_polar, local_azimuthal = _polar_frame(local)
        rotated_polar = efg_rotation.T @ polar
        gauge = (rotated_polar * local_polar).sum(axis=0) - 1j * (
            rotated_polar * local_azimuthal
        ).sum(axis=0)
        result *= (gauge[:, None] ** np.arange(harmonic_count))[:, None, :]
    return result, strengths


def _level_harmonics(transitions):
    return np.concatenate(
        (np.zeros_like(transitions[:, :1]), transitions.cumsum(axis=1)), axis=1
    )


def simulate_coupled_sequence(
    events,
    *,
    observed,
    partner,
    j_hz,
    spectral_width_hz,
    points=8192,
    accuracy=8,
    spinning_hz=0.0,
    rotor_angle_deg=54.736,
    sidebands=0,
    max_rf_step_s=2e-6,
    receiver_phase_deg=0.0,
    transitions="all",
    j_partners=(),
    output="spectrum",
    probe_bandwidth_hz=None,
):
    """Simulate two RF-active spins with a secular isotropic J coupling.

    The observed spin is detected with the partner projection unchanged.
    Each site's EFG rotation maps its principal axes into the common
    sample frame. The CSA rotation maps into that site's EFG frame.
    Use the partner RF fields on Pulse and IdealPulse to address it.
    output="density" returns the final averaged product density matrix
    with partner indices varying fastest.
    """
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    j_hz, spectral_width_hz, spinning_hz, max_rf_step_s, receiver_phase_deg = (
        _real_scalars(
            j_hz, spectral_width_hz, spinning_hz, max_rf_step_s, receiver_phase_deg
        )
    )
    if max_rf_step_s <= 0:
        raise ValueError("max_rf_step_s must be positive.")
    _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands)
    _validate_output(output, sidebands)
    bandwidth = _validate_probe_bandwidth(probe_bandwidth_hz)
    site_a, site_b = _site_parameters(observed), _site_parameters(partner)
    spin_a, spin_b = site_a[0], site_b[0]
    size_a, size_b = int(2 * spin_a + 1), int(2 * spin_b + 1)
    size = size_a * size_b
    if size > 64:
        raise ValueError("The two-spin manifold supports at most 64 levels.")
    if transitions not in ("central", "satellites", "all"):
        raise ValueError("transitions must be 'central', 'satellites', or 'all'.")
    if transitions != "all" and (spin_a < 1.5 or spin_a % 1 != 0.5):
        raise ValueError("Select all transitions for integer spins and spin 1/2.")
    selected = np.arange(size_a - 1)
    if transitions != "all":
        central = spin_a - selected == 0.5
        selected = selected[central if transitions == "central" else ~central]

    rotated = any(value is not None for value in (*site_a[7:9], *site_b[7:9]))
    directions, weights, triangles = _powder_mesh(
        accuracy,
        bool(spinning_hz or rotated),
        full_sphere=bool(spinning_hz and rotated),
    )
    angle = _rotor_angle(rotor_angle_deg, spinning_hz)
    harmonic_count = 5 if spinning_hz else 1
    count = 1
    if spinning_hz:
        count = _rf_rotor_count(
            spectral_width_hz, spinning_hz, accuracy, sidebands, size
        )
    # Account for both site transition/level tables, both product tables,
    # and the readout and output arrays. Count real offsets as complex for
    # a conservative estimate of these persistent and assembly tables.
    storage = (
        weights.size
        * (
            (2 * (size + size_a + size_b) - 2) * harmonic_count
            + selected.size * size_b * (2 * sidebands + harmonic_count + 1)
            + size
        )
        * 16
    )
    if storage > 64 * 1024**2:
        raise ValueError(
            "Coupled-spin work arrays exceed 64 MiB. Reduce accuracy or sidebands."
        )
    a, strengths_a = _site_transitions(directions, angle, site_a, harmonic_count)
    b, _ = _site_transitions(directions, angle, site_b, harmonic_count)

    ops_a = tuple(np.kron(op, np.eye(size_b)) for op in _spin_operators(spin_a))
    ops_b = tuple(np.kron(np.eye(size_a), op) for op in _spin_operators(spin_b))
    initial = ops_a[2] + (site_b[1] / site_a[1]) * ops_b[2]
    coupling_levels = j_hz * np.diag(ops_a[2]) * np.diag(ops_b[2])
    level_a, level_b = _level_harmonics(a), _level_harmonics(b)
    levels = (level_a[:, :, None, :] + level_b[:, None, :, :]).reshape(
        weights.size, size, harmonic_count
    )
    levels[..., 0] += coupling_levels
    level_differences = np.diff(levels, axis=1)
    product_offsets = level_differences[..., 0].real
    product_terms = level_differences[..., 1:] if spinning_hz else None
    prepared = _prepare_events(
        events,
        *ops_a,
        spinning_hz,
        max_rf_step_s,
        site_a[-1],
        partner=ops_b,
        partner_t2_s=site_b[-1],
        coupling_offsets=np.diff(coupling_levels),
    )
    passive_offsets = _passive_offsets(j_partners)
    passive_shift = -np.diff(np.diag(ops_a[2]))
    if output == "density":
        return _powder_density(
            product_offsets,
            product_terms,
            initial,
            prepared,
            spinning_hz,
            count,
            weights,
            triangles,
            passive_offsets,
            passive_shift,
        )

    partner_projections = np.arange(spin_b, -spin_b - 1, -1)
    frequency = (
        a[:, selected, 0].real[..., None] - j_hz * partner_projections
    ).reshape(weights.size, -1)
    detected_terms = (
        np.repeat(a[:, selected, 1:], size_b, axis=1) if spinning_hz else None
    )
    rows = (selected[:, None] * size_b + np.arange(size_b)).ravel()
    readout = rows, rows + size_b
    strengths = np.repeat(strengths_a[selected], size_b)
    axis = np.linspace(-spectral_width_hz / 2, spectral_width_hz / 2, points)
    band_offset = sidebands * spinning_hz
    if (
        frequency.min() - band_offset + passive_offsets[0][0] < axis[0]
        or frequency.max() + band_offset + passive_offsets[-1][0] > axis[-1]
    ):
        raise ValueError("The spectral window must contain the requested transitions.")
    profile = np.zeros(points, dtype=complex)
    receiver = np.exp(1j * np.deg2rad(receiver_phase_deg))
    for offset, population in passive_offsets:
        signals = _sequence_signals(
            product_offsets + offset * passive_shift,
            product_terms,
            strengths,
            initial,
            prepared,
            spinning_hz,
            count,
            sidebands,
            readout=readout,
            readout_terms=detected_terms,
        )
        signals *= receiver * population
        for i in range(strengths.size):
            for j, order in enumerate(range(-sidebands, sidebands + 1)):
                shifted = frequency[:, i] + offset + order * spinning_hz
                weighted = _probe_weights(
                    shifted, weights * signals[:, i, j], bandwidth
                )
                profile.real += _deposit_triangles(
                    axis, shifted, weighted.real, triangles
                )
                profile.imag += _deposit_triangles(
                    axis, shifted, weighted.imag, triangles
                )
    norm = strengths.sum() / 2 * weights[triangles].sum()
    return _as_dataset(axis, profile / norm, "PULSAR coupled sequence")


def _as_dataset(axis, profile, name):
    from spectrochempy.core.dataset.coord import Coord
    from spectrochempy.core.dataset.nddataset import NDDataset

    dataset = NDDataset(profile, name=name)
    dataset.set_coordset(x=Coord(axis, units="Hz", title="frequency offset"))
    return dataset
