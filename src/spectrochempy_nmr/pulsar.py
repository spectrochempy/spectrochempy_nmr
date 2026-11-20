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

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from math import log
from math import sqrt
from numbers import Complex, Real
from operator import index

import numpy as np

from spectrochempy_nmr._pulse_program import available_sequences as available_sequences
from spectrochempy_nmr._pulse_program import load_sequence as load_sequence
from spectrochempy_nmr._pulse_program import PulseProgram as PulseProgram
from spectrochempy_nmr._pulse_program import SequenceError as SequenceError
from spectrochempy_nmr._pulse_program import parse_sequence as parse_sequence
from spectrochempy_nmr._pulse_program import read_sequence as read_sequence


_RF_PROPAGATOR_CACHE_BYTES = 16 * 1024**2


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
    return _tensor_harmonics(directions, tensor, angle)


def _tensor_harmonics(directions, tensor, angle):
    """Project a real symmetric Cartesian tensor onto rotor harmonics."""
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


def _resolve_deposition(name):
    if name == "numpy":
        return _deposit_triangles
    if name != "numba":
        raise ValueError("deposition must be 'numpy' or 'numba'.")

    from spectrochempy_nmr._pulsar_numba_deposition import get_numba_depositor

    return get_numba_depositor() or _deposit_triangles


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
    deposition="numpy",
    _normalize=True,
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
        _resolve_deposition(deposition),
    )
    if transitions == "central":
        harmonics = _quadrupolar_harmonics(directions, quadrupolar_eta, angle)
        # For the central pair both spin factors scale together relative to I=3/2.
        spin_scale = 3 * (spin + 1.5) / (4 * spin**2 * (spin - 0.5))
        harmonics *= spin_scale * cq_hz**2 / (16 * larmor_hz)
        axis, mass = _powder_profile(harmonics + csa, *grid)
        return axis, _normalize_profile(mass) if _normalize else mass

    spectrum = np.zeros(points)
    for m, strength, harmonics in _transition_harmonics(
        directions, cq_hz, quadrupolar_eta, angle, spin, larmor_hz, csa
    ):
        if transitions == "satellites" and m == 0.5:
            continue
        axis, mass = _powder_profile(harmonics, *grid)
        spectrum += strength * mass
    return axis, _normalize_profile(spectrum) if _normalize else spectrum


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
    passive_offsets,
    probe_bandwidth_hz,
    deposit,
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
            spectrum += population * deposit(axis, frequency, detected, triangles)
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
            spectrum += population * deposit(axis, frequency, detected, triangles)
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
    deposition="numpy",
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

    Set deposition="numba" for optional JIT deposition with NumPy fallback.
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
        deposition=deposition,
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
    deposition="numpy",
):
    """Simulate a selected set of quadrupolar transitions of one spin.

    Multiplet calculations support spin up to 16. Use all transitions
    for integer spins. Half-integer spins also
    support "central" or "satellites". Satellites include first- and
    second-order shifts. Intensities are normalized once over the
    requested transitions and MAS orders. See simulate_central_transition
    for frequency, powder mesh and CSA conventions.

    Set deposition="numba" for optional JIT deposition with NumPy fallback.
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
        deposition=deposition,
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
    deposition="numpy",
):
    """Simulate first-order chemical shift anisotropy for an isolated spin.

    The signed reduced anisotropy is delta_zz - delta_iso in Hz.
    The asymmetry is (delta_yy - delta_xx) / anisotropy, in [0, 1].
    A zero MAS rate gives a static powder pattern. Otherwise, the rotor
    angle defaults to 54.736 degrees and sidebands selects the orders on each side.
    The spectrum is normalized over the requested bands.

    Set deposition="numba" for optional JIT deposition with NumPy fallback.
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
        _resolve_deposition(deposition),
    )
    return _as_dataset(
        axis, _normalize_profile(profile), "PULSAR chemical shift anisotropy"
    )


def _spin_operators(spin):
    m = np.arange(spin, -spin - 1, -1)
    raising = np.diag(np.sqrt(spin * (spin + 1) - m[1:] * (m[1:] + 1)), k=1)
    return (raising + raising.T) / 2, (raising - raising.T) / (2j), np.diag(m)


def _propagator(hamiltonian_hz, duration_s):
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_hz)
    phase = np.exp(-2j * np.pi * duration_s * eigenvalues)
    return (eigenvectors * phase[..., None, :]) @ eigenvectors.conj().swapaxes(-1, -2)


def _apply_propagator(density, propagator):
    return propagator @ density @ propagator.conj().swapaxes(-1, -2)


def _pulse_propagator(rf, offsets, duration_s):
    # Adjacent differences of the diagonal must equal the offsets.
    levels = np.concatenate(
        (np.zeros_like(offsets[..., :1]), offsets.cumsum(axis=-1)), axis=-1
    )
    size = rf.shape[0]
    rows = np.arange(size)
    hamiltonian = np.broadcast_to(rf, (*levels.shape, size)).copy()
    hamiltonian[..., rows, rows] += levels
    return _propagator(hamiltonian, duration_s)


def _resolve_rf_solver(name):
    if name == "dense":
        return None
    if name != "tridiagonal":
        raise ValueError("rf_solver must be 'dense' or 'tridiagonal'.")

    from spectrochempy_nmr._pulsar_tridiagonal import get_tridiagonal_solver

    accelerated = get_tridiagonal_solver()
    if accelerated is None:
        return None

    def solve(rf, offsets, duration):
        propagator = accelerated(rf, offsets, duration)
        if propagator is None:
            return _pulse_propagator(rf, offsets, duration)
        return propagator

    return solve


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
    """Keep elements with row-minus-column spin projection equal to order.

    symmetric=True retains only the pair m=+order/2 and m=-order/2.
    """

    order: int
    symmetric: bool = False


@dataclass(frozen=True)
class PhaseCycle:
    """Explicit pulse or coherence pathways and their optional coefficients."""

    scans: tuple
    coefficients: tuple | None = None


@dataclass(frozen=True)
class _T1Series:
    preparation: object
    times: tuple
    conversion: tuple


def _phase_cycle_coefficients(scans, coefficients):
    if not 1 <= len(scans) <= 256:
        raise ValueError("A phase cycle must contain 1 to 256 scans.")
    if coefficients is None:
        coefficients = (1 / len(scans),) * len(scans)
    else:
        coefficients = tuple(coefficients)
        if len(coefficients) != len(scans) or not all(
            isinstance(weight, Complex) for weight in coefficients
        ):
            raise ValueError("Supply one numeric coefficient per phase-cycle scan.")
        coefficients = tuple(map(complex, coefficients))
        if not np.isfinite(coefficients).all():
            raise ValueError("Phase-cycle coefficients must be finite.")
    return coefficients


def _prepare_scans(events, *args, **kwargs):
    if not isinstance(events, PhaseCycle):
        return ((1.0, _prepare_events(events, *args, **kwargs)),)
    scans = tuple(events.scans)
    coefficients = _phase_cycle_coefficients(scans, events.coefficients)
    return tuple(
        (weight, _prepare_events(scan, *args, **kwargs))
        for weight, scan in zip(coefficients, scans, strict=True)
    )


def _rf_axis(ix, iy, phase_deg):
    phase = np.deg2rad(phase_deg)
    return np.cos(phase) * ix + np.sin(phase) * iy


def _rf_pulse_basis(ix, iy, phase_deg, *, real):
    if not real:
        return _rf_axis(ix, iy, phase_deg), None

    # In the descending m basis, e^(-i phase Iz) rotates Ix to the RF axis.
    # The internal Hamiltonian is diagonal, so this basis works for every
    # midpoint of a finite pulse even as the rotor changes the offsets.
    phase = np.angle(np.exp(1j * np.deg2rad(phase_deg)))
    if phase == 0.0:
        return ix.real, None
    levels = np.arange(ix.shape[0])
    basis = np.exp(1j * phase * (levels[:, None] - levels[None, :]))
    return ix.real, basis


def _prepare_events(
    events,
    ix,
    iy,
    iz,
    spinning_hz,
    max_rf_step_s,
    t2_s,
    *,
    allow_empty=False,
    rf_solver=None,
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
            if not isinstance(event.symmetric, (bool, np.bool_)):
                raise TypeError("Coherence symmetric must be boolean.")
            if event.symmetric:
                mask &= projections[:, None] + projections[None, :] == 0
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
            axis, basis = _rf_pulse_basis(
                ix, iy, phase, real=partner is None and bool(spinning_hz or rf_solver)
            )
            rf = power * axis
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
            if basis is not None:
                # Return from the real basis along with the carrier-frame correction.
                frame *= basis
            prepared.append(
                (
                    "pulse",
                    duration,
                    rf,
                    -np.diff(frequency_shift),
                    count,
                    frame,
                    basis,
                    rf_solver or _pulse_propagator,
                )
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
    if not prepared and not allow_empty:
        raise ValueError("Supply at least one sequence event.")
    return prepared


def _phase_integral(terms, gamma, spinning_hz):
    orders = np.arange(1, 5)
    rotor = np.exp(-1j * orders[:, None] * gamma) / (orders[:, None] * spinning_hz)
    return -(terms @ rotor).imag.swapaxes(-1, -2)


def _shared_pulse_keys(scans, spinning_hz, gamma):
    """Find matching RF pulses at the same sampled rotor trajectory."""
    if not spinning_hz or len(scans) < 2:
        return ({},) * len(scans)

    omega = 2 * np.pi * spinning_hz
    keys_by_scan = []
    counts = {}
    for _, events in scans:
        angles = gamma
        keys = {}
        for event_index, (kind, *args) in enumerate(events):
            if kind == "delay":
                angles = angles + omega * args[0]
            elif kind == "pulse":
                duration, rf, offset, count, *_ = args
                step = duration / count
                key = (
                    angles.tobytes(),
                    count,
                    step,
                    rf.tobytes(),
                    offset.tobytes(),
                )
                keys[event_index] = key
                counts[key] = counts.get(key, 0) + 1
                angles = angles + count * (omega * step)
        keys_by_scan.append(keys)
    return tuple(
        {event: key for event, key in keys.items() if counts[key] > 1}
        for keys in keys_by_scan
    )


def _mas_pulse_steps(offsets, terms, rf, offset, gamma, step, step_phase, count, solve):
    order = np.arange(1, 5)
    for i in range(count):
        midpoint = gamma + (i + 0.5) * step_phase
        rotor = np.exp(-1j * order[:, None] * midpoint)
        instantaneous = offsets[:, None, :] + (terms @ rotor).real.swapaxes(-1, -2)
        yield solve(rf, instantaneous - offset, step)


def _propagate_sequence(
    offsets, terms, initial, events, spinning_hz, gamma, keys, propagators, capacity
):
    density = initial
    for event_index, (kind, *args) in enumerate(events):
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
            density = _apply_propagator(density, _propagator(axis, duration))
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
            duration, rf, offset, count, frame, basis, solve = args
            step = duration / count
            if basis is not None:
                density = density * basis.conj()
            if spinning_hz:
                step_phase = 2 * np.pi * spinning_hz * step
                key = keys.get(event_index)
                if key in propagators:
                    density = _apply_propagator(density, propagators[key])
                else:
                    steps = _mas_pulse_steps(
                        offsets,
                        terms,
                        rf,
                        offset,
                        gamma,
                        step,
                        step_phase,
                        count,
                        solve,
                    )
                    if key is not None and len(propagators) < capacity:
                        # The product acts on any incoming density in the real basis.
                        pulse = next(steps)
                        for propagator in steps:
                            pulse = propagator @ pulse
                        propagators[key] = pulse
                        density = _apply_propagator(density, pulse)
                    else:
                        for propagator in steps:
                            density = _apply_propagator(density, propagator)
                gamma = gamma + count * step_phase
            else:
                pulse = solve(rf, offsets - offset, step)
                density = _apply_propagator(density, pulse)
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


def _sequence_state_blocks(offsets, terms, initial, scans, spinning_hz, count):
    """Yield each scan's density and rotor angle with a cache local to the block."""
    size = initial.shape[0]
    block_size = min(128, 131072 // (count * size**2))
    gamma = 2 * np.pi * np.arange(count) / count if spinning_hz else None
    keys_by_scan = _shared_pulse_keys(scans, spinning_hz, gamma)
    for start in range(0, offsets.shape[0], block_size):
        block = slice(start, start + block_size)
        block_terms = terms[block] if spinning_hz else None
        batch = (
            (offsets[block].shape[0], count)
            if spinning_hz
            else (offsets[block].shape[0],)
        )
        state = np.broadcast_to(initial, (*batch, size, size))
        capacity = _RF_PROPAGATOR_CACHE_BYTES // (
            state.size * np.dtype(complex).itemsize
        )
        propagators = {}
        for (coefficient, events), keys in zip(scans, keys_by_scan, strict=True):
            density, gamma_end = _propagate_sequence(
                offsets[block],
                block_terms,
                state,
                events,
                spinning_hz,
                gamma,
                keys,
                propagators,
                capacity,
            )
            yield block, coefficient, density, gamma_end


def _t1_state_blocks(offsets, terms, initial, scans, spinning_hz, count, tails):
    for block, coefficient, prepared, gamma_start in _sequence_state_blocks(
        offsets, terms, initial, scans, spinning_hz, count
    ):
        block_terms = None if terms is None else terms[block]
        block_offsets = offsets[block]
        for row, events in enumerate(tails):
            density, gamma_end = _propagate_sequence(
                block_offsets,
                block_terms,
                prepared,
                events,
                spinning_hz,
                gamma_start,
                {},
                {},
                0,
            )
            yield row, block, coefficient, density, gamma_end


def _t1_rotor_grid(times, spinning_hz, count):
    """Find a bounded common rotor grid without interpolating the RF dynamics."""
    from fractions import Fraction
    from math import lcm

    turns = np.asarray(times) * spinning_hz
    if np.max(turns) > 4096:
        return None
    shifts = np.remainder(turns, 1) * count
    multiple = 1
    for shift in shifts:
        fraction = Fraction(float(shift)).limit_denominator(32)
        if abs(float(fraction) - shift) > 1e-12:
            return None
        multiple = lcm(multiple, fraction.denominator)
        if multiple > 8 or multiple * count > 4096:
            return None
    grid_count = multiple * count
    indices = np.rint(shifts * multiple).astype(np.intp) % grid_count
    return grid_count, indices


def _conversion_readout(offsets, terms, events, spinning_hz, gamma, selected):
    """Pull detected coherences back through the conversion by trace duality."""
    size = offsets.shape[-1] + 1
    if (
        events
        and events[0][0] == "pulse"
        and all(event[0] == "mask" for event in events[1:])
    ):
        # A single RF conversion followed by selections/relaxation needs
        # only two rows of its complete propagator per detected transition.
        _, duration, rf, offset, count, frame, basis, solve = events[0]
        step = duration / count
        step_phase = 2 * np.pi * spinning_hz * step
        steps = _mas_pulse_steps(
            offsets, terms, rf, offset, gamma, step, step_phase, count, solve
        )
        pulse = next(steps)
        for propagator in steps:
            pulse = propagator @ pulse
        pulse *= frame[:, 0][:, None]
        if basis is not None:
            pulse *= basis[:, 0].conj()[None, :]
        observable = np.einsum(
            "vgsi,vgsj->svgij",
            pulse[..., selected + 1, :].conj(),
            pulse[..., selected, :],
        )
        for _, mask in events[1:]:
            observable *= mask[selected, selected + 1][:, None, None, None, None]
        return observable, gamma + count * step_phase
    shape = (selected.size, offsets.shape[0], gamma.size, size, size)
    observable = np.zeros(shape, dtype=complex)
    observable[np.arange(selected.size), :, :, selected + 1, selected] = 1
    omega = 2 * np.pi * spinning_hz
    angles = gamma
    starts = []
    for kind, *args in events:
        starts.append(angles)
        if kind == "delay":
            angles = angles + omega * args[0]
        elif kind == "pulse":
            duration, _, _, count, *_ = args
            angles = angles + count * (omega * (duration / count))
    gamma_end = angles
    for (kind, *args), start in zip(reversed(events), reversed(starts), strict=True):
        if kind == "mask":
            observable *= args[0].T
        elif kind == "ideal":
            axis, duration = args
            propagator = _propagator(axis, duration)
            observable = _apply_propagator(
                observable, propagator.conj().swapaxes(-1, -2)
            )
        elif kind == "delay":
            duration, decoupled = args
            end = start + omega * duration
            phase = (
                2 * np.pi * (offsets - decoupled)[:, None, :] * duration
                + _phase_integral(terms, end, spinning_hz)
                - _phase_integral(terms, start, spinning_hz)
            )
            levels = np.concatenate(
                (np.zeros_like(phase[..., :1]), phase.cumsum(axis=-1)), axis=-1
            )
            evolution = np.exp(1j * levels)
            observable *= evolution[..., :, None] * evolution[..., None, :].conj()
        else:
            duration, rf, offset, count, frame, basis, solve = args
            step = duration / count
            step_phase = omega * step
            observable *= frame.T
            for index in range(count - 1, -1, -1):
                propagator = next(
                    _mas_pulse_steps(
                        offsets,
                        terms,
                        rf,
                        offset,
                        start + index * step_phase,
                        step,
                        step_phase,
                        1,
                        solve,
                    )
                )
                observable = _apply_propagator(
                    observable, propagator.conj().swapaxes(-1, -2)
                )
            if basis is not None:
                observable *= basis.conj().T
    return observable, gamma_end


def _t1_conversion_signals(
    offsets,
    terms,
    strengths,
    initial,
    scans,
    spinning_hz,
    count,
    sidebands,
    evolutions,
    conversion,
    selected,
    grid,
):
    """Yield one bounded block of band amplitudes for all indirect times."""
    grid_count, shifts = grid
    size = initial.shape[0]
    block_size = max(1, min(64, 131072 // (grid_count * size**2 * selected.size)))
    band_orders = np.arange(-sidebands, sidebands + 1)
    base = np.arange(count) * (grid_count // count)
    take = (shifts[:, None] + base) % grid_count
    for start in range(0, offsets.shape[0], block_size):
        block = slice(start, min(start + block_size, offsets.shape[0]))
        block_offsets, block_terms = offsets[block], terms[block]
        signals = np.zeros(
            (len(evolutions), block_offsets.shape[0], selected.size, band_orders.size),
            dtype=complex,
        )
        readout_start = None
        for _, coefficient, prepared, gamma_start in _sequence_state_blocks(
            block_offsets, block_terms, initial, scans, spinning_hz, count
        ):
            if readout_start != gamma_start[0]:
                gamma = gamma_start[0] + 2 * np.pi * np.arange(grid_count) / grid_count
                observable, gamma_end = _conversion_readout(
                    block_offsets, block_terms, conversion, spinning_hz, gamma, selected
                )
                wave = np.exp(
                    1j
                    * _phase_integral(block_terms[:, selected], gamma_end, spinning_hz)
                ).swapaxes(-1, -2)
                readout_start = gamma_start[0]
            active_rows, active_cols = np.nonzero(np.any(prepared != 0, axis=(0, 1)))
            initial_coherences = prepared[..., active_rows, active_cols]
            readout = observable[..., active_cols, active_rows]
            phase_begin = _phase_integral(block_terms, gamma_start, spinning_hz)
            phase_end = _phase_integral(block_terms, gamma, spinning_hz)
            levels_begin = np.concatenate(
                (np.zeros_like(phase_begin[..., :1]), phase_begin.cumsum(axis=-1)),
                axis=-1,
            )
            levels_end = np.concatenate(
                (np.zeros_like(phase_end[..., :1]), phase_end.cumsum(axis=-1)), axis=-1
            )
            levels_iso = np.concatenate(
                (np.zeros_like(block_offsets[..., :1]), block_offsets.cumsum(axis=-1)),
                axis=-1,
            )
            initial_phase = (
                levels_begin[..., active_rows] - levels_begin[..., active_cols]
            )
            rotor_phase = levels_end[..., active_rows] - levels_end[..., active_cols]
            frequency = levels_iso[..., active_rows] - levels_iso[..., active_cols]
            for row, (events, indices) in enumerate(zip(evolutions, take, strict=True)):
                time = events[0][1]
                phase = (
                    2 * np.pi * time * frequency[:, None, :]
                    + rotor_phase[:, indices]
                    - initial_phase
                )
                coherences = initial_coherences * np.exp(-1j * phase)
                for _, mask in events[1:]:
                    coherences *= mask[active_rows, active_cols]
                coherence = np.einsum(
                    "svgn,vgn->vsg", readout[:, :, indices], coherences
                )
                signal = -1j * np.sqrt(strengths[selected])[None, :, None] * coherence
                detected = wave[..., indices]
                detection = np.fft.fft(detected, axis=-1)[..., band_orders] / count
                preparation = np.fft.ifft(signal * detected.conj(), axis=-1)[
                    ..., band_orders
                ]
                signals[row] += coefficient * detection * preparation
        yield block, signals


def _t1_spectrum(
    axis,
    offsets,
    terms,
    strengths,
    initial,
    scans,
    spinning_hz,
    count,
    sidebands,
    evolutions,
    conversion,
    selected,
    grid,
    weights,
    triangles,
    bandwidth,
    deposit,
    receiver,
    passive_offsets,
    workers,
):
    """Stream triangle groups so the full series need not retain every orientation."""
    profiles = np.zeros((len(evolutions), axis.size), dtype=complex)
    # Each triangle contributes independently, including shared vertex weights.
    # Retaining only the vertices of each group bounds the signal workspace.
    slots = len(evolutions) * selected.size * (2 * sidebands + 1)
    if 3 * slots * np.dtype(complex).itemsize > 32 * 1024**2:
        raise ValueError(
            "Series band amplitudes exceed 32 MiB per triangle. Split the times."
        )
    vertex_limit = max(3, (32 * 1024**2) // (16 * slots))
    triangle_count = max(1, min(4096, vertex_limit // 3))
    errors, callback = np.geterr(), np.geterrcall()
    for start in range(0, triangles.shape[0], triangle_count):
        vertices, remap = np.unique(
            triangles[start : start + triangle_count], return_inverse=True
        )
        local_triangles = remap.reshape(-1, 3)
        local_terms = None if terms is None else terms[vertices]
        local_offsets = offsets[vertices]
        signals = np.empty(
            (len(evolutions), vertices.size, selected.size, 2 * sidebands + 1),
            dtype=complex,
        )
        for offset, population in passive_offsets:
            shifted = local_offsets + offset
            signals.fill(0)
            if grid is None:
                tails = [events + conversion for events in evolutions]
                signals = _sequence_signals(
                    shifted,
                    local_terms,
                    strengths[selected],
                    initial,
                    scans,
                    spinning_hz,
                    count,
                    sidebands,
                    readout=(selected, selected + 1),
                    readout_terms=None if terms is None else local_terms[:, selected],
                    workers=workers,
                    tails=tails,
                    out=signals,
                )
            else:
                edges = np.linspace(
                    0, vertices.size, min(workers, vertices.size) + 1, dtype=int
                )

                def process(lo, hi, output):
                    with np.errstate(call=callback, **errors):
                        for block, values in _t1_conversion_signals(
                            shifted[lo:hi],
                            local_terms[lo:hi],
                            strengths,
                            initial,
                            scans,
                            spinning_hz,
                            count,
                            sidebands,
                            evolutions,
                            conversion,
                            selected,
                            grid,
                        ):
                            output[:, lo + block.start : lo + block.stop] = values

                if workers == 1:
                    process(0, vertices.size, signals)
                else:
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        jobs = [
                            pool.submit(process, lo, hi, signals)
                            for lo, hi in zip(edges[:-1], edges[1:], strict=True)
                        ]
                        for job in jobs:
                            job.result()
            signals *= receiver * population
            for i, transition in enumerate(selected):
                for j, order in enumerate(range(-sidebands, sidebands + 1)):
                    frequency = shifted[:, transition] + order * spinning_hz
                    for profile, row in zip(profiles, signals, strict=True):
                        signal = _probe_weights(
                            frequency, weights[vertices] * row[:, i, j], bandwidth
                        )
                        profile.real += deposit(
                            axis, frequency, signal.real, local_triangles
                        )
                        profile.imag += deposit(
                            axis, frequency, signal.imag, local_triangles
                        )
        del row, signals
    return profiles


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
        for block, coefficient, states, _ in _sequence_state_blocks(
            offsets + offset * passive_shift, terms, initial, events, spinning_hz, count
        ):
            if spinning_hz:
                states = states.mean(axis=1)
            density += (
                population
                * coefficient
                * np.einsum("v,vab->ab", vertex_weights[block], states)
            )
    return density / vertex_weights.sum()


def _validate_output(output, sidebands, workers):
    if output not in ("spectrum", "density"):
        raise ValueError("output must be 'spectrum' or 'density'.")
    if output == "density" and sidebands:
        raise ValueError("Sideband selection applies only to spectral output.")
    if workers < 1:
        raise ValueError("workers must be a positive integer.")
    if output == "density" and workers > 1:
        raise ValueError("Multiple workers apply only to spectral output.")


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
    workers=1,
    out=None,
    tails=None,
):
    """Average complex band amplitudes over rotor phase at preparation onset."""
    if out is None:
        shape = (offsets.shape[0], strengths.size, 2 * sidebands + 1)
        out = np.zeros(shape if tails is None else (len(tails), *shape), dtype=complex)
    rows = out[None] if tails is None else out
    workers = min(workers, offsets.shape[0])
    if workers > 1:
        # Each slice processes the full cycle and writes to its own output range.
        edges = np.linspace(0, offsets.shape[0], workers + 1, dtype=int)
        chunks = (slice(lo, hi) for lo, hi in zip(edges[:-1], edges[1:], strict=True))
        errors, callback = np.geterr(), np.geterrcall()

        def process(chunk):
            with np.errstate(call=callback, **errors):
                _sequence_signals(
                    offsets[chunk],
                    None if terms is None else terms[chunk],
                    strengths,
                    initial,
                    events,
                    spinning_hz,
                    count,
                    sidebands,
                    readout=readout,
                    readout_terms=None
                    if readout_terms is None
                    else readout_terms[chunk],
                    out=rows[0, chunk] if tails is None else rows[:, chunk],
                    tails=tails,
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in executor.map(process, chunks):
                pass
        return out

    band_orders = np.arange(-sidebands, sidebands + 1)
    if tails is None:
        states = (
            (0, *state)
            for state in _sequence_state_blocks(
                offsets, terms, initial, events, spinning_hz, count
            )
        )
    else:
        states = _t1_state_blocks(
            offsets, terms, initial, events, spinning_hz, count, tails
        )
    for row, block, coefficient, density, gamma_end in states:
        if readout is None:
            coherence = np.diagonal(density, offset=1, axis1=-2, axis2=-1)
        else:
            coherence = density[..., readout[0], readout[1]]
        signal = -1j * np.sqrt(strengths) * coherence
        if not spinning_hz:
            rows[row, block, :, 0] += coefficient * signal
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
        rows[row, block] += coefficient * detection * preparation
    return out


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
    rf_solver="dense",
    deposition="numpy",
    workers=1,
):
    """Simulate a powder after one finite rectangular RF pulse.

    The pulse is on the simulation carrier. Frequencies are in Hz and
    duration_s is in seconds. Thermal Iz is the initial density matrix.
    The returned complex spectrum is scaled to unit total absorption for
    an ideal on-carrier 90 degree x pulse. Actual excitation losses and
    phase are retained. With MAS, midpoint propagation uses time steps
    no longer than max_rf_step_s or 1/32 of a rotor period.
    Set rf_solver="tridiagonal" to use the optional Numba/LAPACK solver.

    Set deposition="numba" for optional JIT deposition with NumPy fallback.
    Set workers above one to compute groups of powder orientations in threads.
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
        deposition=deposition,
        rf_solver=rf_solver,
        workers=workers,
    )


def _mqmas_rf_profiles(preparation, conversion, coherence_order, **settings):
    """Compute both States quadratures with one excitation/readout per block."""
    spin = settings["spin"]
    spinning = settings["spinning_hz"]
    points, accuracy, sidebands, workers = (
        index(settings[name]) for name in ("points", "accuracy", "sidebands", "workers")
    )
    width, max_step, receiver_phase = _real_scalars(
        settings["spectral_width_hz"],
        settings["max_rf_step_s"],
        settings["receiver_phase_deg"],
    )
    if max_step <= 0:
        raise ValueError("max_rf_step_s must be positive.")
    t2 = settings["t2_s"]
    if t2 is not None:
        (t2,) = _real_scalars(t2)
        if t2 <= 0:
            raise ValueError("t2_s must be positive.")
    _validate_grid(width, spinning, points, accuracy, sidebands)
    _validate_output("spectrum", sidebands, workers)
    bandwidth = _validate_probe_bandwidth(settings["probe_bandwidth_hz"])
    count = _rf_rotor_count(width, spinning, accuracy, sidebands, int(2 * spin + 1))
    directions, weights, triangles = _powder_mesh(accuracy, hemisphere=True)
    angle = np.arccos(1 / np.sqrt(3))
    slots = directions.shape[1] * (int(2 * spin) * 4 + 2 * (2 * sidebands + 1))
    if slots * np.dtype(complex).itemsize > 64 * 1024**2:
        raise ValueError(
            "MQMAS RF work arrays exceed 64 MiB. Reduce accuracy or sidebands."
        )
    table = np.stack(
        [
            terms.T
            for _, _, terms in _transition_harmonics(
                directions,
                settings["cq_hz"],
                settings["eta"],
                angle,
                spin,
                settings["larmor_hz"],
                0,
            )
        ],
        axis=1,
    )
    offsets = table[..., 0].real + settings["shift_hz"]
    terms = np.ascontiguousarray(table[..., 1:])
    del table
    ct = int(spin - 0.5)
    strength = (spin + 0.5) ** 2
    axis = np.linspace(-width / 2, width / 2, points)
    if (
        offsets[:, ct].min() - sidebands * spinning < axis[0]
        or offsets[:, ct].max() + sidebands * spinning > axis[-1]
    ):
        raise ValueError("The spectral window must contain the requested transitions.")
    ix, iy, iz = _spin_operators(spin)
    pulse_args = ix, iy, iz, spinning, max_step, t2
    solver = _resolve_rf_solver(settings["rf_solver"])
    scans = _prepare_scans(preparation, *pulse_args, rf_solver=solver, allow_empty=True)
    conversion = _prepare_events(
        conversion, *pulse_args, rf_solver=solver, allow_empty=True
    )
    upper, lower = int(spin - coherence_order / 2), int(spin + coherence_order / 2)
    orders = np.arange(-sidebands, sidebands + 1)
    signals = np.zeros((2, offsets.shape[0], orders.size), dtype=complex)
    errors, callback = np.geterr(), np.geterrcall()

    def process(lo, hi):
        with np.errstate(call=callback, **errors):
            for block, coefficient, density, gamma_start in _sequence_state_blocks(
                offsets[lo:hi], terms[lo:hi], iz, scans, spinning, count
            ):
                observed, gamma_end = _conversion_readout(
                    offsets[lo:hi][block],
                    terms[lo:hi][block],
                    conversion,
                    spinning,
                    gamma_start,
                    np.array([ct]),
                )
                positive = density[..., upper, lower] * observed[0, ..., lower, upper]
                negative = density[..., lower, upper] * observed[0, ..., upper, lower]
                quadratures = (
                    -1j
                    * np.sqrt(strength)
                    * np.stack(
                        ((positive + negative) / 2, (positive - negative) / (2j))
                    )
                )
                wave = np.exp(
                    1j
                    * _phase_integral(
                        terms[lo:hi][block, ct : ct + 1], gamma_end, spinning
                    )[..., 0]
                )
                detection = np.fft.fft(wave, axis=-1)[..., orders] / count
                prepared = np.fft.ifft(quadratures * wave.conj(), axis=-1)[..., orders]
                signals[:, lo + block.start : lo + min(block.stop, hi - lo)] += (
                    coefficient * detection * prepared
                )

    workers = min(workers, offsets.shape[0])
    edges = np.linspace(0, offsets.shape[0], workers + 1, dtype=int)
    if workers == 1:
        process(0, offsets.shape[0])
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = [
                pool.submit(process, lo, hi)
                for lo, hi in zip(edges[:-1], edges[1:], strict=True)
            ]
            for job in jobs:
                job.result()
    signals *= np.exp(1j * np.deg2rad(receiver_phase))
    deposit = _resolve_deposition(settings["deposition"])
    profiles = np.zeros((2, points), dtype=complex)
    for i, order in enumerate(orders):
        frequency = offsets[:, ct] + order * spinning
        for profile, quadrature in zip(profiles, signals, strict=True):
            mass = _probe_weights(frequency, weights * quadrature[:, i], bandwidth)
            profile.real += deposit(axis, frequency, mass.real, triangles)
            profile.imag += deposit(axis, frequency, mass.imag, triangles)
    profiles /= strength / 2 * weights[triangles].sum()
    return [_as_dataset(axis, profile, "MQMAS RF transfer") for profile in profiles]


class MQMASModel:
    """Reusable RF-weighted F2 profiles for a direct, sheared MQMAS model.

    Preparation and conversion use the same event objects as
    simulate_pulse_sequence. The transfer is calculated at t1=0 for the
    symmetric +/-q/2 pair, with both States quadratures. Rebuild the model
    when changing the spin system, carrier offset or pulse sequence.
    spectrum() reuses these profiles for F1 grids and F1/F2 widths.

    The fixed-transfer approximation assumes rotor-synchronized MQ
    evolution. Conversion-phase modulation and rotor micromotion along t1
    require comparison with explicit sequence calculations. Frequencies,
    Pq, Cq and offsets are in Hz. RF phase and absolute transfer losses
    relative to an ideal central-transition x90 are retained.
    Requested F2 sidebands share the principal F1 ridge; correlations of
    rotational sidebands between F1 and F2 are omitted.
    Finite indirect-acquisition truncation is not modeled by the Gaussian.
    """

    def __init__(
        self,
        preparation,
        conversion,
        *,
        spin,
        larmor_hz,
        spinning_hz,
        spectral_width_hz,
        cq_hz=0.0,
        pq_hz=None,
        eta=0.0,
        shift_hz=0.0,
        coherence_order=3,
        points=8192,
        accuracy=16,
        sidebands=0,
        max_rf_step_s=2e-6,
        t2_s=None,
        receiver_phase_deg=0.0,
        probe_bandwidth_hz=None,
        rf_solver="dense",
        deposition="numpy",
        workers=1,
    ):
        from .processing.mqmas import _mqmas_geometry

        coherence_order, ratio, scale = _mqmas_geometry(spin, coherence_order)
        spin, larmor_hz, spinning_hz, cq_hz, eta, shift_hz = _real_scalars(
            spin, larmor_hz, spinning_hz, cq_hz, eta, shift_hz
        )
        if not 1.5 <= spin <= 15.5 or larmor_hz <= 0 or spinning_hz <= 0:
            raise ValueError(
                "Use a half-integer spin from 3/2 to 31/2 and positive Larmor and MAS frequencies."
            )
        if not 0 <= eta <= 1:
            raise ValueError("eta must lie between 0 and 1.")
        if pq_hz is not None:
            (pq_hz,) = _real_scalars(pq_hz)
            if pq_hz < 0 or cq_hz != 0:
                raise ValueError("Use a nonnegative Pq, with Cq left at zero.")
            cq_hz = pq_hz / np.sqrt(1 + eta**2 / 3)
        else:
            pq_hz = abs(cq_hz) * np.sqrt(1 + eta**2 / 3)
        quadrupolar_shift = (
            -3
            * (spin * (spin + 1) - 0.75)
            * pq_hz**2
            / (40 * spin**2 * (2 * spin - 1) ** 2 * larmor_hz)
        )
        if isinstance(preparation, PhaseCycle):
            scans = tuple(tuple(scan) for scan in preparation.scans)
            coefficients = _phase_cycle_coefficients(scans, preparation.coefficients)
        else:
            scans, coefficients = (tuple(preparation),), (1.0,)
        conversion = tuple(conversion)
        settings = dict(
            spin=spin,
            larmor_hz=larmor_hz,
            cq_hz=cq_hz,
            eta=eta,
            shift_hz=shift_hz,
            spectral_width_hz=spectral_width_hz,
            points=points,
            accuracy=accuracy,
            spinning_hz=spinning_hz,
            sidebands=sidebands,
            max_rf_step_s=max_rf_step_s,
            t2_s=t2_s,
            receiver_phase_deg=receiver_phase_deg,
            probe_bandwidth_hz=probe_bandwidth_hz,
            rf_solver=rf_solver,
            deposition=deposition,
            workers=workers,
        )
        self._profiles = _mqmas_rf_profiles(
            PhaseCycle(scans, coefficients), conversion, coherence_order, **settings
        )
        self._parameters = dict(
            approximation="fixed RF transfer with analytical F1 line",
            spin=spin,
            coherence_order=coherence_order,
            larmor_hz=larmor_hz,
            cq_hz=cq_hz,
            pq_hz=pq_hz,
            eta=eta,
            shift_hz=shift_hz,
            isotropic_f1_hz=shift_hz - 10 * quadrupolar_shift / 17,
            central_transition_qis_hz=quadrupolar_shift,
            shear_ratio=ratio,
            f1_scale=scale,
            transfer_reference_t1_s=0.0,
            spinning_hz=spinning_hz,
        )

    @property
    def f1_position_hz(self):
        """Analytical F1 position with unit chemical-shift sensitivity."""
        return self._parameters["isotropic_f1_hz"]

    def spectrum(
        self,
        *,
        f1_spectral_width_hz,
        f1_points=256,
        f1_offset_hz=0.0,
        f1_gaussian_fwhm_hz=100.0,
        f2_gaussian_fwhm_hz=0.0,
    ):
        """Construct a complex F1/F2 map using the stored RF transfer.

        F1 uses a unit-area Gaussian absorption line and its periodic
        Hilbert dispersion, weighted by the two States RF quadratures.
        F2 broadening is optional. Values are bin masses and neither
        RF profile is peak- or area-normalized. The F1 window can truncate
        the Gaussian tails. Both frequency axes are in Hz.
        """
        from scipy.signal import hilbert
        from scipy.special import ndtr

        from spectrochempy.core.dataset.coord import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        f1_points = index(f1_points)
        width, center, fwhm1, fwhm2 = _real_scalars(
            f1_spectral_width_hz,
            f1_offset_hz,
            f1_gaussian_fwhm_hz,
            f2_gaussian_fwhm_hz,
        )
        if f1_points < 2 or width <= 0 or fwhm1 <= 0:
            raise ValueError(
                "Use at least two F1 points and positive F1 width and Gaussian FWHM."
            )
        if fwhm2 < 0:
            raise ValueError("F2 Gaussian FWHM must be nonnegative.")
        step = width / f1_points
        axis = center + (np.arange(f1_points) - f1_points // 2) * step
        position = self.f1_position_hz
        if position < axis[0] - step / 2 or position > axis[-1] + step / 2:
            raise ValueError(
                "The F1 window must contain the analytical isotropic position."
            )
        sigma = fwhm1 / np.sqrt(8 * np.log(2))
        edges = np.r_[axis - step / 2, axis[-1] + step / 2]
        absorption = np.diff(ndtr((edges - position) / sigma))
        dispersion = hilbert(absorption).imag
        profiles = self._profiles
        if fwhm2:
            profiles = [
                broaden(profile, gaussian_fwhm_hz=fwhm2) for profile in profiles
            ]
        result = NDDataset(
            absorption[:, None] * profiles[0].data[None, :]
            + dispersion[:, None] * profiles[1].data[None, :],
            name="PULSAR analytical MQMAS",
        )
        result.set_coordset(
            x=profiles[0].x.copy(),
            y=Coord(
                axis,
                units="Hz",
                title="F1 isotropic frequency offset",
                rounding=False,
                sigdigits=15,
            ),
        )
        result.meta.mqmas_model = dict(
            self._parameters,
            f1_gaussian_fwhm_hz=fwhm1,
            f2_gaussian_fwhm_hz=fwhm2,
        )
        return result


def simulate_mqmas(
    preparation,
    conversion,
    *,
    f1_spectral_width_hz,
    f1_points=256,
    f1_offset_hz=0.0,
    f1_gaussian_fwhm_hz=100.0,
    f2_gaussian_fwhm_hz=0.0,
    **settings,
):
    """Build a direct MQMAS model and return its sheared F1/F2 spectrum.

    settings are those of MQMASModel. For repeated maps with different
    F1 grids or widths, construct MQMASModel once and call spectrum().
    """
    model = MQMASModel(preparation, conversion, **settings)
    return model.spectrum(
        f1_spectral_width_hz=f1_spectral_width_hz,
        f1_points=f1_points,
        f1_offset_hz=f1_offset_hz,
        f1_gaussian_fwhm_hz=f1_gaussian_fwhm_hz,
        f2_gaussian_fwhm_hz=f2_gaussian_fwhm_hz,
    )


def simulate_t1_series(preparation, t1_s, conversion, **settings):
    """Simulate complex F2 spectra at the given indirect times, in seconds.

    Supply a preparation sequence or PhaseCycle, then conversion events
    common to the scans. Each row is equivalent to inserting Delay(t1)
    between preparation and conversion in simulate_pulse_sequence. The
    preparation is propagated once per scan and powder block. The result
    is an NDDataset with indirect time on y and frequency offset on x.

    Settings are those of simulate_pulse_sequence for spectral output.
    For States, call once per quadrature branch and retain the two datasets
    for reconstruction. Keep the original sampling schedule for nonuniform
    reconstruction, since core dataset copies can linearize nearly uniform
    coordinates.

    Powder integration streams groups of triangles with bounded band-amplitude
    storage. For indirect times that share a bounded rotor grid, conversion
    readouts are reused across the series without phase interpolation.
    Other schedules use the ordinary density propagation for each time.
    """
    times = _real_scalars(*t1_s)
    if not times or min(times) < 0:
        raise ValueError("Supply one or more nonnegative indirect times.")
    return simulate_pulse_sequence(
        _T1Series(preparation, times, tuple(conversion)), **settings
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
    rf_solver="dense",
    deposition="numpy",
    workers=1,
):
    """Simulate a sequence of pulses, delays and coherence selections.

    Each event uses the simulation carrier as its phase reference.
    Optional t2_s damps off-diagonal terms after each timed event.
    receiver_phase_deg multiplies the complex signal by exp(i*phase).
    The ideal on-carrier x90 pulse defines unit absorption for the
    selected transitions. RF always propagates the complete manifold.
    output="density" returns the powder-averaged deviation density
    matrix at the end of preparation, before receiver phase.
    For finite pulses, rf_solver="tridiagonal" uses the optional
    Numba/LAPACK backend when available and falls back to the dense solver.

    Set deposition="numba" for optional JIT deposition with NumPy fallback.
    For spectral output, set workers above one to compute groups of powder
    orientations in threads. Scans share propagators within each block.
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
    workers = index(workers)
    _validate_output(output, sidebands, workers)
    series = events if isinstance(events, _T1Series) else None
    if series is not None:
        if output != "spectrum":
            raise ValueError("simulate_t1_series requires spectral output.")
        events = series.preparation
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
    row_count = 1 if series is None else len(series.times)
    order_count = 2 * sidebands + 1 if output == "spectrum" else 0
    if spinning_hz or series is not None:
        # Bound the persistent harmonics and signal arrays before allocating either.
        slots = (4 if spinning_hz else 0) + (0 if series is not None else order_count)
        if offsets.size * slots * np.dtype(complex).itemsize > 64 * 1024**2:
            raise ValueError(
                "Pulse work arrays exceed 64 MiB. Reduce accuracy, sidebands "
                "or the number of indirect times per call."
            )
    if spinning_hz:
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
    pulse_args = ix, iy, iz, spinning_hz, max_rf_step_s, t2_s
    solver = _resolve_rf_solver(rf_solver)
    prepared = _prepare_scans(
        events, *pulse_args, rf_solver=solver, allow_empty=series is not None
    )
    tails = None
    if series is not None:
        conversion = _prepare_events(
            series.conversion, *pulse_args, rf_solver=solver, allow_empty=True
        )
        evolutions = [
            _prepare_events((Delay(time),), *pulse_args, rf_solver=solver)
            for time in series.times
        ]
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
    deposit = _resolve_deposition(deposition)
    if series is not None:
        grid = _t1_rotor_grid(series.times, spinning_hz, count) if spinning_hz else None
        if grid is not None and grid[0] * iz.size * selected.sum() > 131072:
            grid = None
        profiles = _t1_spectrum(
            axis,
            offsets,
            harmonics if spinning_hz else None,
            strengths,
            iz,
            prepared,
            spinning_hz,
            count if spinning_hz else 1,
            sidebands,
            evolutions,
            conversion,
            np.flatnonzero(selected),
            grid,
            weights,
            triangles,
            bandwidth,
            deposit,
            receiver,
            passive_offsets,
            workers,
        )
        profiles /= strengths[selected].sum() / 2 * weights[triangles].sum()
        return _as_dataset(axis, profiles, "PULSAR t1 series", times=series.times)
    profiles = np.zeros((row_count, points), dtype=complex)
    signals = None
    for offset, population in passive_offsets:
        shifted = offsets + offset
        if signals is not None:
            signals.fill(0)
        signals = _sequence_signals(
            shifted,
            harmonics if spinning_hz else None,
            strengths,
            iz,
            prepared,
            spinning_hz,
            count if spinning_hz else 1,
            sidebands,
            workers=workers,
            out=signals,
            tails=tails,
        )
        rows = signals[None] if series is None else signals
        rows *= receiver * population
        for i in np.flatnonzero(selected):
            for j, order in enumerate(range(-sidebands, sidebands + 1)):
                frequency = shifted[:, i] + order * spinning_hz
                for profile, row in zip(profiles, rows, strict=True):
                    signal = _probe_weights(
                        frequency, weights * row[:, i, j], bandwidth
                    )
                    profile.real += deposit(axis, frequency, signal.real, triangles)
                    profile.imag += deposit(axis, frequency, signal.imag, triangles)
    ideal_signal = strengths[selected].sum() / 2
    powder_mass = weights[triangles].sum()
    profiles /= ideal_signal * powder_mass
    if series is not None:
        return _as_dataset(axis, profiles, "PULSAR t1 series", times=series.times)
    return _as_dataset(axis, profiles[0], "PULSAR pulse sequence")


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
    (
        spin,
        larmor,
        shift,
        cq,
        eta,
        csa,
        csa_eta,
        efg_rotation,
        csa_rotation,
        _,
    ) = parameters
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
    j_hz=0.0,
    dipolar_hz=0.0,
    dipolar_axis=(0.0, 0.0, 1.0),
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
    deposition="numpy",
    workers=1,
):
    """Simulate two RF-active spins with secular J and dipolar coupling.

    dipolar_hz adds D * (3*cos(theta)**2 - 1) * Iz*Sz in Hz,
    where dipolar_axis is a real unit bond vector in the common sample
    frame and theta is its angle to the field. This signed D follows
    historical PyPulsar, not a full dipolar tensor's zz component (2*D).
    Nonzero D requires a heteronuclear spin-1/2 pair
    with well-separated Larmor frequencies. Homonuclear flip-flop terms
    are not included. Under MAS theta follows the rotor trajectory during
    RF, delays and acquisition. Select sidebands and converge accuracy
    and max_rf_step_s for the experiment. Delay(decouple_j=True) removes
    J only, leaving the dipolar interaction active.

    The observed spin is detected with the partner projection unchanged.
    Each site's EFG rotation maps its principal axes into the common
    sample frame. The CSA rotation maps into that site's EFG frame.
    Use the partner RF fields on Pulse and IdealPulse to address it.
    output="density" returns the final averaged product density matrix
    with partner indices varying fastest.

    Set deposition="numba" for optional JIT deposition with NumPy fallback.
    For spectral output, set workers above one to compute groups of powder
    orientations in threads. Scans share propagators within each block.
    """
    points, accuracy, sidebands = index(points), index(accuracy), index(sidebands)
    (
        j_hz,
        dipolar_hz,
        spectral_width_hz,
        spinning_hz,
        max_rf_step_s,
        receiver_phase_deg,
    ) = _real_scalars(
        j_hz,
        dipolar_hz,
        spectral_width_hz,
        spinning_hz,
        max_rf_step_s,
        receiver_phase_deg,
    )
    if max_rf_step_s <= 0:
        raise ValueError("max_rf_step_s must be positive.")
    _validate_grid(spectral_width_hz, spinning_hz, points, accuracy, sidebands)
    workers = index(workers)
    _validate_output(output, sidebands, workers)
    bandwidth = _validate_probe_bandwidth(probe_bandwidth_hz)
    site_a, site_b = _site_parameters(observed), _site_parameters(partner)
    spin_a, spin_b = site_a[0], site_b[0]
    bond = np.asarray(dipolar_axis)
    if (
        bond.shape != (3,)
        or bond.dtype.kind not in "fiu"
        or not np.isfinite(bond).all()
        or not np.isclose(np.linalg.norm(bond), 1.0, rtol=0, atol=1e-7)
    ):
        raise ValueError("dipolar_axis must be a real unit vector with three entries.")
    bond = bond.astype(float)
    bond /= np.linalg.norm(bond)
    if dipolar_hz:
        if spin_a != 0.5 or spin_b != 0.5:
            raise ValueError("Dipolar coupling currently requires two spin-1/2 sites.")
        if np.isclose(site_a[1], site_b[1], rtol=1e-6, atol=0):
            raise ValueError(
                "Dipolar coupling requires a heteronuclear pair with "
                "well-separated Larmor frequencies."
            )
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
    rotated = rotated or bool(dipolar_hz and np.any(bond[:2]))
    directions, weights, triangles = _powder_mesh(
        accuracy,
        bool(spinning_hz or rotated or dipolar_hz),
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
    coupling = np.zeros((weights.size, harmonic_count), dtype=complex)
    coupling[:, 0] = j_hz
    if dipolar_hz:
        tensor = dipolar_hz * (3 * np.outer(bond, bond) - np.eye(3))
        coupling += _tensor_harmonics(directions, tensor, angle)[:harmonic_count].T
    products = np.diag(ops_a[2]) * np.diag(ops_b[2])
    level_a, level_b = _level_harmonics(a), _level_harmonics(b)
    levels = (level_a[:, :, None, :] + level_b[:, None, :, :]).reshape(
        weights.size, size, harmonic_count
    )
    levels += coupling[:, None, :] * products[None, :, None]
    level_differences = np.diff(levels, axis=1)
    product_offsets = level_differences[..., 0].real
    product_terms = level_differences[..., 1:] if spinning_hz else None
    prepared = _prepare_scans(
        events,
        *ops_a,
        spinning_hz,
        max_rf_step_s,
        site_a[-1],
        partner=ops_b,
        partner_t2_s=site_b[-1],
        coupling_offsets=np.diff(j_hz * products),
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
        a[:, selected, 0].real[..., None]
        - coupling[:, 0].real[:, None, None] * partner_projections
    ).reshape(weights.size, -1)
    detected_terms = (
        (
            a[:, selected, None, 1:]
            - coupling[:, None, None, 1:] * partner_projections[None, None, :, None]
        ).reshape(weights.size, -1, harmonic_count - 1)
        if spinning_hz
        else None
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
    deposit = _resolve_deposition(deposition)
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
            workers=workers,
        )
        signals *= receiver * population
        for i in range(strengths.size):
            for j, order in enumerate(range(-sidebands, sidebands + 1)):
                shifted = frequency[:, i] + offset + order * spinning_hz
                weighted = _probe_weights(
                    shifted, weights * signals[:, i, j], bandwidth
                )
                profile.real += deposit(axis, shifted, weighted.real, triangles)
                profile.imag += deposit(axis, shifted, weighted.imag, triangles)
    norm = strengths.sum() / 2 * weights[triangles].sum()
    return _as_dataset(axis, profile / norm, "PULSAR coupled sequence")


def _as_dataset(axis, profile, name, *, times=None):
    from spectrochempy.core.dataset.coord import Coord
    from spectrochempy.core.dataset.nddataset import NDDataset

    dataset = NDDataset(profile, name=name)
    coords = {"x": Coord(axis, units="Hz", title="frequency offset")}
    if times is not None:
        times = np.asarray(times)
        coords["y"] = Coord(
            times,
            units="s",
            title="indirect evolution time",
            rounding=False,
            linearize_below=0,
            sigdigits=12,
        )
    dataset.set_coordset(**coords)
    if times is not None:
        dataset.meta.td = list(dataset.shape)
        # Keep Coord's regular-spacing classification for the FFT, but retain
        # the exact times used by the propagator if Coord rebuilt the grid.
        time_coord = dataset.y
        time_coord._rounding = False
        if not np.array_equal(time_coord.data, times):
            time_coord._set_data(times)
    return dataset


def broaden(spectrum, *, gaussian_fwhm_hz):
    """Apply Gaussian broadening to a one-dimensional spectrum.

    The width is the full width at half maximum in Hz. The spectrum needs
    a uniformly spaced frequency axis with units convertible to Hz and
    finite, unmasked real or complex data. A zero width returns a copy.

    The transform uses a periodic frequency grid. Leave enough room at
    the edges of the spectral window to contain the broadened lines.
    The returned spectrum retains the input axis, units, and total area.
    """
    from spectrochempy.core.dataset.coord import Coord
    from spectrochempy.core.dataset.nddataset import NDDataset

    (width,) = _real_scalars(gaussian_fwhm_hz)
    if width < 0:
        raise ValueError("Gaussian FWHM must be nonnegative.")
    if not isinstance(spectrum, NDDataset):
        raise TypeError("Expected an NDDataset spectrum.")
    if spectrum.ndim != 1 or spectrum.size < 2:
        raise ValueError(
            "Expected a one-dimensional spectrum with at least two points."
        )
    axis = getattr(spectrum, "x", None)
    if axis is None or axis.units is None or not axis.units.is_compatible_with("Hz"):
        raise ValueError(
            "Expected a frequency coordinate with units convertible to Hz."
        )
    steps = np.diff(axis.to("Hz").data)
    step_hz = steps.mean()
    if (
        not np.isfinite(steps).all()
        or not np.allclose(steps, step_hz, rtol=1e-3, atol=0)
        or step_hz == 0
    ):
        raise ValueError("Expected a uniformly spaced frequency coordinate.")
    if spectrum.is_masked or not np.isfinite(spectrum.data).all():
        raise ValueError("Expected finite, unmasked spectrum data.")
    result = spectrum.copy(keepname=True)
    if width == 0:
        return result

    data = spectrum.data
    if not spectrum.is_complex:
        # Conjugate the analytic spectrum for the positive-time convention.
        data = spectrum.ht().conj().data
    time_data = np.fft.ifft(np.fft.ifftshift(data))
    time = np.arange(spectrum.size) / (spectrum.size * abs(step_hz))
    time_signal = NDDataset(time_data, coordset=[Coord(time, units="s", sigdigits=12)])
    gb = width / (1.2 * sqrt(log(2)))
    if spectrum.is_complex:
        midpoint = time_signal.size // 2
        center = time[midpoint] * time_signal.x.units
        # Complex spectra have both positive and negative time lags.
        windowed = time_signal.roll(pts=midpoint).gm(gb=gb, shifted=center)
        windowed = windowed.roll(pts=-midpoint)
    else:
        windowed = time_signal.gm(gb=gb)
    # The original coordinate is already present on the output copy.
    broadened = np.fft.fftshift(np.fft.fft(windowed.data))
    result.data = broadened if spectrum.is_complex else broadened.real
    result.history = f"Applied Gaussian broadening ({width:g} Hz FWHM)"
    return result
