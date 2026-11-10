"""Fit the PULSAR central-transition model to an absorptive NMR dataset."""

from dataclasses import dataclass
from operator import index

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur

from .pulsar import _central_transition_profile, _real_scalars


@dataclass(frozen=True)
class CentralParameters:
    """Quadrupolar line parameters with frequency quantities in Hz."""

    cq_hz: float
    eta: float
    shift_hz: float
    gaussian_fwhm_hz: float

    def _values(self):
        return np.array(
            [self.cq_hz, self.eta, self.shift_hz, self.gaussian_fwhm_hz], dtype=float
        )


@dataclass(frozen=True)
class PulsarFitResult:
    """Best fit, observed-minus-fitted residuals and fit diagnostics."""

    parameters: CentralParameters
    fitted: NDDataset
    residuals: NDDataset
    rms: float
    evaluations: int


def _frequency_offsets(dataset, larmor_hz, origin_hz):
    coord = dataset.x
    if coord is None or coord.units is None:
        raise ValueError("The dataset needs a frequency coordinate in Hz or ppm.")
    axis = np.asarray(coord.data, dtype=float)
    if coord.units == ur.ppm:
        if origin_hz:
            raise ValueError("A ppm axis already defines chemical zero.")
        return axis * larmor_hz * 1e-6
    if not coord.units.is_compatible_with(ur.Hz):
        raise ValueError("The dataset needs a frequency coordinate in Hz or ppm.")
    return (axis * coord.units).to(ur.Hz).magnitude - origin_hz


def fit_central_transition(
    dataset,
    *,
    initial,
    bounds,
    larmor_hz,
    spectral_width_hz,
    origin_hz=0.0,
    window_hz=None,
    points=8192,
    accuracy=16,
    spin=1.5,
    spinning_hz=0.0,
    sidebands=0,
    rotor_angle_deg=54.736,
    csa_anisotropy_hz=0.0,
    csa_eta=0.0,
    csa_rotation=None,
):
    """Fit one central transition on a real, one-dimensional NDDataset.

    initial and the two bounds are CentralParameters. Cq, shift and
    Gaussian FWHM are in Hz. For a frequency axis, subtract origin_hz
    to set the reference. For ppm, chemical zero is already the origin
    and larmor_hz converts ppm to frequency offsets. window_hz restricts
    the residual calculation to an interval on that offset scale.
    The model width must contain all bands throughout the given bounds.
    Start with a positive Cq and bound Gaussian FWHM below by one model bin.
    The fitted profile includes a nonnegative amplitude and linear baseline.
    """
    if not isinstance(dataset, NDDataset) or dataset.ndim != 1:
        raise TypeError("Supply a one-dimensional NDDataset.")
    if np.iscomplexobj(dataset.data):
        raise ValueError("Supply a real, absorptive spectrum.")
    larmor_hz, spectral_width_hz, origin_hz = _real_scalars(
        larmor_hz, spectral_width_hz, origin_hz
    )
    if larmor_hz <= 0 or spectral_width_hz <= 0:
        raise ValueError("Use a positive Larmor frequency and spectral width.")
    points = index(points)
    if points < 2:
        raise ValueError("Use at least two simulation points.")
    frequency = _frequency_offsets(dataset, larmor_hz, origin_hz)
    observed = np.asarray(dataset.data, dtype=float)
    if frequency.shape != observed.shape or not np.isfinite(frequency).all():
        raise ValueError("Frequency coordinates must be finite and match the spectrum.")
    steps = np.diff(frequency)
    if not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("Frequency coordinates must be strictly monotonic.")
    valid = np.isfinite(observed) & ~np.asarray(dataset.mask, dtype=bool)
    if window_hz is not None:
        lo, hi = window_hz
        if not np.isfinite((lo, hi)).all() or lo >= hi:
            raise ValueError("window_hz must give finite increasing bounds.")
        valid &= (frequency >= lo) & (frequency <= hi)
    if valid.sum() < 8:
        raise ValueError("The fit requires at least eight unmasked, finite points.")
    fit_frequency, fit_values = frequency[valid], observed[valid]
    scale = np.max(np.abs(fit_values))
    if scale == 0:
        raise ValueError("The fit window has no signal.")

    lower, upper = bounds
    if not all(
        isinstance(value, CentralParameters) for value in (initial, lower, upper)
    ):
        raise TypeError("initial and bounds must use CentralParameters.")
    x0, lo, hi = (value._values() for value in (initial, lower, upper))
    if not np.isfinite((x0, lo, hi)).all() or np.any(lo >= hi):
        raise ValueError("Fit bounds must be finite and strictly increasing.")
    if lo[0] < 0 or lo[1] < 0 or hi[1] > 1 or lo[3] < 0:
        raise ValueError(
            "Constrain Cq and Gaussian FWHM to nonnegative values and eta to [0, 1]."
        )
    if x0[0] <= 0:
        raise ValueError(
            "The initial Cq must be positive to resolve quadrupolar broadening."
        )
    if lo[3] < spectral_width_hz / (points - 1):
        raise ValueError(
            "The minimum Gaussian FWHM must be at least one simulation bin."
        )
    if hi[3] > spectral_width_hz:
        raise ValueError(
            "The maximum Gaussian FWHM must fit inside the simulation width."
        )
    if np.any(x0 < lo) or np.any(x0 > hi):
        raise ValueError("Initial parameters must lie within the fit bounds.")

    parameter_scale = np.array([1e6, 1, 1e3, 100])
    baseline = np.column_stack(
        (
            np.ones(frequency.size),
            (frequency - fit_frequency.mean()) / np.ptp(fit_frequency),
        )
    )
    normalized = fit_values / scale
    calls = 0

    def design(parameters):
        cq, eta, shift, fwhm = parameters * parameter_scale
        axis, mass = _central_transition_profile(
            cq_hz=cq,
            eta=eta,
            shift_hz=shift,
            larmor_hz=larmor_hz,
            spectral_width_hz=spectral_width_hz,
            points=points,
            accuracy=accuracy,
            spin=spin,
            spinning_hz=spinning_hz,
            sidebands=sidebands,
            rotor_angle_deg=rotor_angle_deg,
            csa_anisotropy_hz=csa_anisotropy_hz,
            csa_eta=csa_eta,
            csa_rotation=csa_rotation,
        )
        step = axis[1] - axis[0]
        sigma = fwhm / (np.sqrt(8 * np.log(2)) * step)
        if sigma > 0:
            mass = gaussian_filter1d(mass, sigma, mode="constant")
        line = np.interp(frequency, axis + step / 2, mass, left=0, right=0)
        return np.column_stack((line, baseline))

    def project(matrix):
        selected = matrix[valid]
        weights = np.linalg.lstsq(selected, normalized, rcond=None)[0]
        if weights[0] < 0:
            weights[0] = 0
            weights[1:] = np.linalg.lstsq(selected[:, 1:], normalized, rcond=None)[0]
        return weights

    def residual(parameters):
        nonlocal calls
        calls += 1
        matrix = design(parameters)
        return matrix[valid] @ project(matrix) - normalized

    result = least_squares(
        residual,
        x0 / parameter_scale,
        bounds=(lo / parameter_scale, hi / parameter_scale),
        diff_step=2e-4,
    )
    if not result.success:
        raise RuntimeError(f"PULSAR fit failed: {result.message}")

    matrix = design(result.x)
    coefficients = project(matrix)
    if coefficients[0] <= 0:
        raise ValueError("The fit did not identify a positive PULSAR component.")
    fitted_values = scale * (matrix @ coefficients)
    fitted = dataset.copy()
    fitted.data = fitted_values
    fitted.name = "PULSAR fit"
    residuals = dataset.copy()
    residuals.data = observed - fitted_values
    residuals.name = "PULSAR residuals"
    rms = float(np.sqrt(np.mean(residuals.data[valid] ** 2)))
    return PulsarFitResult(
        CentralParameters(*(result.x * parameter_scale)), fitted, residuals, rms, calls
    )
