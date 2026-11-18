"""Nonnegative inversion of phased MQMAS spectra on a shift and Pq grid."""

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.optimize import nnls

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset


from ._mqmas_response import mas_response as mas_response
from ._mqmas_response import z_filtered_3qmas_response as z_filtered_3qmas_response


def _axis(values, name, *, grid=False):
    vector = np.asarray(values)
    if np.iscomplexobj(vector):
        raise ValueError(f"{name} must be real.")
    vector = np.array(vector, dtype=float, copy=True)
    minimum = 3 if grid else 2
    if vector.ndim != 1 or vector.size < minimum or not np.isfinite(vector).all():
        raise ValueError(
            f"{name} must be a finite vector with at least {minimum} points."
        )
    step = np.diff(vector)
    if grid:
        if not (step > 0).all() or not np.allclose(
            step, step.mean(), rtol=1e-6, atol=0
        ):
            raise ValueError(f"{name} must be increasing and uniformly spaced.")
    elif not ((step > 0).all() or (step < 0).all()):
        raise ValueError(f"{name} must be strictly monotonic.")
    vector.setflags(write=False)
    return vector


def _axes(spectrum, dimension=None):
    if not isinstance(spectrum, NDDataset) or spectrum.ndim not in (1, 2):
        raise TypeError("Expected a one- or two-dimensional NDDataset.")
    if dimension is not None and spectrum.ndim != dimension:
        raise ValueError(f"Expected a {dimension}D spectrum.")
    coordinates = (spectrum.y, spectrum.x) if spectrum.ndim == 2 else (spectrum.x,)
    if any(
        coord is None
        or coord.units is None
        or not coord.units.is_compatible_with("ppm")
        for coord in coordinates
    ):
        raise ValueError("Expected calibrated spectral coordinates in ppm.")
    return tuple(
        _axis(coord.to("ppm").data, "Spectral coordinate") for coord in coordinates
    )


@dataclass(frozen=True, eq=False)
class Kernel:
    """A matrix of signed, per-population responses on a spectral and parameter grid.

    The matrix rows follow the C ordering of the spectral axes. Columns
    enumerate isotropic shift first and Pq second. Supply axes in ppm and
    the Pq grid in MHz. The parameter axes are uniformly spaced.
    """

    axes_ppm: tuple
    shift_ppm: np.ndarray
    pq_mhz: np.ndarray
    matrix: np.ndarray

    def __post_init__(self):
        axes = tuple(_axis(axis, "Spectral coordinate") for axis in self.axes_ppm)
        shift = _axis(self.shift_ppm, "Isotropic shift grid", grid=True)
        pq = _axis(self.pq_mhz, "Pq grid", grid=True)
        if len(axes) not in (1, 2) or pq[0] < 0:
            raise ValueError("Expected one or two spectral axes and nonnegative Pq.")
        raw = np.asarray(self.matrix)
        if np.iscomplexobj(raw):
            raise ValueError("Expected a real, phase-sensitive kernel matrix.")
        matrix = np.array(raw, dtype=float, copy=True, order="C")
        shape = (np.prod([len(axis) for axis in axes]), shift.size * pq.size)
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise ValueError(f"Expected a finite kernel matrix of shape {shape}.")
        matrix.setflags(write=False)
        object.__setattr__(self, "axes_ppm", axes)
        object.__setattr__(self, "shift_ppm", shift)
        object.__setattr__(self, "pq_mhz", pq)
        object.__setattr__(self, "matrix", matrix)


def build_kernel(
    spectrum, shift_ppm, pq_mhz, response, *, asymmetry="uniform", nodes=7
):
    """Average per-population responses at fixed isotropic shift and Pq.

    `response(shift_ppm, cq_hz, eta)` returns a real array with the shape
    of the supplied spectrum, on its calibrated spectral grid. Use
    `z_filtered_3qmas_response` or `mas_response` for PULSAR templates,
    or provide a response that replays the acquisition and processing.
    `asymmetry` is a fixed eta in [0, 1], "uniform", or "czjzek" for
    standard isotropic Czjzek weighting conditional on Pq.
    """
    axes = _axes(spectrum)
    shift = _axis(shift_ppm, "Isotropic shift grid", grid=True)
    pq = _axis(pq_mhz, "Pq grid", grid=True)
    if pq[0] < 0:
        raise ValueError("Pq must be nonnegative.")
    if isinstance(asymmetry, str):
        if asymmetry not in {"uniform", "czjzek"}:
            raise ValueError("Use uniform, czjzek or a fixed asymmetry in [0, 1].")
        if (
            isinstance(nodes, bool)
            or not isinstance(nodes, (int, np.integer))
            or nodes < 1
        ):
            raise ValueError("nodes must be a positive integer.")
        x, weights = np.polynomial.legendre.leggauss(nodes)
        eta = (x + 1) / 2
        if asymmetry == "czjzek":
            weights *= eta * (1 - eta**2 / 9) / (1 + eta**2 / 3) ** 2.5
        weights /= weights.sum()
    else:
        eta = np.array([float(asymmetry)])
        if not 0 <= eta[0] <= 1:
            raise ValueError("Fixed asymmetry must lie in [0, 1].")
        weights = np.ones(1)
    matrix = np.zeros((spectrum.size, shift.size * pq.size))
    cq_scale = 1e6 / np.sqrt(1 + eta**2 / 3)
    for index, (delta, coupling) in enumerate(product(shift, pq)):
        for e, scale, weight in zip(eta, cq_scale, weights, strict=True):
            values = np.asarray(
                response(float(delta), float(coupling * scale), float(e))
            )
            if (
                values.shape != spectrum.shape
                or np.iscomplexobj(values)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    "The response must be real, finite and match the spectrum."
                )
            matrix[:, index] += weight * values.ravel()
    return Kernel(axes, shift, pq, matrix)


@dataclass(frozen=True, eq=False)
class FitResult:
    """Population and amplitude maps, fitted spectra and peak-relative RMS errors.

    Both maps use the primary spectrum's peak as their intensity reference.
    The MAS gain and baseline act on a peak-normalized MAS spectrum.
    Spectral predictions and residuals are in the original input units.
    """

    population: NDDataset
    amplitude: NDDataset
    reconstructed: NDDataset
    residual: NDDataset
    rms: float
    objective: float
    mas_reconstructed: NDDataset | None = None
    mas_residual: NDDataset | None = None
    mas_rms: float | None = None
    mas_gain: float | None = None
    mas_baseline: float | None = None


@dataclass(frozen=True, eq=False)
class ProfilePoint:
    """The best fit among the supplied widths for one profile point."""

    weight: float | None
    width_hz: float
    fit: FitResult


def _observations(spectrum, kernel, dimension):
    axes = _axes(spectrum, dimension)
    if len(axes) != len(kernel.axes_ppm) or any(
        not np.array_equal(axis, saved) for axis, saved in zip(axes, kernel.axes_ppm)
    ):
        raise ValueError("Spectrum coordinates do not match the kernel.")
    data = np.asarray(spectrum.data)
    if np.iscomplexobj(data):
        raise ValueError("Pass the phased real spectrum, including signed intensities.")
    unmasked = ~np.broadcast_to(
        np.asarray(spectrum.mask, dtype=bool), data.shape
    ).ravel()
    target = np.asarray(data.ravel()[unmasked], dtype=float)
    if not target.size or not np.isfinite(target).all() or not np.any(target):
        raise ValueError("Expected finite unmasked intensities and a nonzero peak.")
    peak = np.abs(target).max()
    return kernel.matrix[unmasked], target / peak, unmasked, peak


def _curvature(shape):
    count = np.prod(shape)
    identity = np.eye(count).reshape(*shape, count)
    return np.vstack(
        [
            np.diff(identity, n=2, axis=axis).reshape(-1, count) * (size - 1) ** 2
            for axis, size in enumerate(shape)
        ]
    )


def _solve(matrix, rhs, constraint):
    diagonal = np.diag(matrix).max()
    system = matrix + 1e-12 * diagonal * np.eye(rhs.size)
    if constraint is not None:
        region, mass = constraint
        norm = np.linalg.norm(region)
        region, mass = region / norm, mass / norm
        strength = 5 * diagonal
        system += strength * np.outer(region, region)
    factor = cholesky(system, lower=True)

    def solve(linear):
        return nnls(
            factor.T,
            solve_triangular(factor, linear, lower=True),
            maxiter=10 * rhs.size,
        )[0]

    if constraint is None:
        return solve(rhs)
    target = rhs + strength * mass * region
    multiplier_scale = max(np.linalg.norm(target), diagonal * mass)

    def error(multiplier):
        weights = solve(target - multiplier * multiplier_scale * region)
        return region @ weights - mass

    start = error(0.0)
    if start == 0:
        return solve(target)
    bound = np.copysign(1.0, start)
    for _ in range(64):
        if start * error(bound) <= 0:
            break
        bound *= 2
    else:
        raise RuntimeError("Could not bracket the local population constraint.")
    low, high = sorted((0.0, bound))
    multiplier = brentq(error, low, high, xtol=1e-12, rtol=1e-12)
    return solve(target - multiplier * multiplier_scale * region)


def _selection(kernel, local_region):
    shift, pq = local_region
    a, b = sorted(shift)
    c, d = sorted(pq)
    if not np.isfinite([a, b, c, d]).all():
        raise ValueError("Local population bounds must be finite.")
    s = (kernel.shift_ppm >= a) & (kernel.shift_ppm <= b)
    q = (kernel.pq_mhz >= c) & (kernel.pq_mhz <= d)
    region = np.logical_and.outer(s, q).ravel()
    if not region.any():
        raise ValueError("Local population bounds do not include any kernel cells.")
    return region


def _spectral_result(spectrum, selected, peak, predicted):
    reconstructed = spectrum.copy()
    residual = spectrum.copy()
    input_data = np.asarray(spectrum.data).ravel()
    model = np.zeros(input_data.shape, dtype=float)
    difference = np.zeros_like(model)
    model[selected] = peak * predicted
    difference[selected] = input_data[selected] - model[selected]
    reconstructed.data = model.reshape(spectrum.shape)
    residual.data = difference.reshape(spectrum.shape)
    return reconstructed, residual


def _map(data, kernel, name):
    shape = (kernel.shift_ppm.size, kernel.pq_mhz.size)
    result = NDDataset(data.reshape(shape), name=name)
    result.set_coordset(
        y=Coord(kernel.shift_ppm, units="ppm", title="isotropic chemical shift"),
        x=Coord(kernel.pq_mhz, units="MHz", title="quadrupolar product"),
    )
    return result


def fit(
    spectrum,
    kernel,
    *,
    regularization=0.1,
    normalize_columns=False,
    minimum_column_norm=0.0,
    mas=None,
    mas_kernel=None,
    mas_weight=1.0,
    local_region=None,
    local_weight=None,
):
    """Invert a signed 2D MQMAS spectrum, optionally alongside a signed MAS spectrum.

    In the joint fit, one population map predicts both experiments. A
    positive MAS gain and an additive MAS baseline are fitted separately.
    Peak-normalized residuals are averaged over unmasked pixels in each
    experiment, with the MAS RMS weighted by `mas_weight`. Regularization
    penalizes second differences on each normalized parameter axis.

    `normalize_columns=True` is available for MQMAS alone. It regularizes
    the amplitudes on unit-norm columns and converts them back to
    populations. `minimum_column_norm` is a fraction of the largest
    unmasked column norm. Joint fits retain a cell supported by either
    spectrum. Exactly invisible columns are fixed to zero.

    To constrain a sum of populations, supply `local_region=((low_ppm,
    high_ppm), (low_mhz, high_mhz))` and a nonnegative `local_weight`.
    """
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    if not 0 <= minimum_column_norm <= 1:
        raise ValueError("minimum_column_norm must lie in [0, 1].")
    if (mas is None) != (mas_kernel is None):
        raise ValueError("Supply both mas and mas_kernel for a joint fit.")
    if mas is not None and (
        normalize_columns or not np.isfinite(mas_weight) or mas_weight <= 0
    ):
        raise ValueError(
            "Joint fits use population units and require a positive MAS weight."
        )
    if (local_region is None) != (local_weight is None):
        raise ValueError("Supply both local_region and local_weight.")
    if local_weight is not None and (not np.isfinite(local_weight) or local_weight < 0):
        raise ValueError("local_weight must be finite and nonnegative.")

    q, y, selected, peak = _observations(spectrum, kernel, 2)
    norms = np.linalg.norm(q, axis=0)
    if not np.any(norms):
        raise ValueError("The MQMAS experiment has no observable kernel columns.")
    support = (norms > 0) & (norms >= minimum_column_norm * norms.max())
    if mas is not None:
        if not (
            np.array_equal(kernel.shift_ppm, mas_kernel.shift_ppm)
            and np.array_equal(kernel.pq_mhz, mas_kernel.pq_mhz)
        ):
            raise ValueError("MQMAS and MAS kernels must share the parameter grid.")
        k, z, mas_selected, mas_peak = _observations(mas, mas_kernel, 1)
        k_mean, z_mean = k.mean(axis=0), z.mean()
        centered = k - k_mean
        zc = z - z_mean
        visible = np.linalg.norm(centered, axis=0)
        support |= (visible > 0) & (visible >= minimum_column_norm * visible.max())
    region = _selection(kernel, local_region) if local_region is not None else None
    if local_weight == 0:
        support &= ~region
    if not support.any() or (local_weight and not (region & support).any()):
        raise ValueError("The requested fit has no observable supported columns.")

    scale = norms[support] if normalize_columns else np.ones(support.sum())
    design = q[:, support] / scale
    shape = (kernel.shift_ppm.size, kernel.pq_mhz.size)
    curvature = _curvature(shape)[:, support]
    penalty = (
        np.sqrt(regularization / y.size)
        * np.linalg.norm(design, ord="fro")
        / np.linalg.norm(curvature, ord="fro")
        * curvature
    )
    gram = (design.T @ design) / y.size + penalty.T @ penalty
    rhs = design.T @ y / y.size
    constraint = None
    if local_weight:
        constraint = (region[support] / scale, local_weight)
    gain = baseline = None
    if mas is None:
        coefficients = _solve(gram, rhs, constraint)
    else:
        m = centered[:, support]
        factor = mas_weight**2 / z.size
        mgram, mrhs = factor * (m.T @ m), factor * (m.T @ zc)

        def evaluate(value):
            f = _solve(gram + value**2 * mgram, rhs + value * mrhs, constraint)
            score = (
                np.mean((design @ f - y) ** 2)
                + mas_weight**2 * np.mean((value * (m @ f) - zc) ** 2)
                + np.linalg.norm(penalty @ f) ** 2
            )
            return score, f

        seed = _solve(gram, rhs, constraint)
        projected = m @ seed
        length = projected @ projected
        if length == 0 or projected @ zc <= 0:
            initial = np.linalg.norm(zc) / max(
                np.linalg.norm(m, ord="fro"), np.finfo(float).tiny
            )
        else:
            initial = (projected @ zc) / length
        if not np.isfinite(initial) or initial <= 0 or not np.any(m):
            raise ValueError("MAS gain is unidentifiable on the supported columns.")
        center = np.log(initial)
        for span in (3.0, 6.0):
            search = minimize_scalar(
                lambda log_gain: evaluate(np.exp(log_gain))[0],
                bounds=(center - span, center + span),
                method="bounded",
                options={"xatol": 0.015},
            )
            if search.success and abs(search.x - center) < span - 0.03:
                break
        else:
            raise RuntimeError(
                "The positive MAS gain search did not reach an interior minimum."
            )
        gain = np.exp(search.x)
        _, coefficients = evaluate(gain)
        baseline = z_mean - gain * (k_mean[support] @ coefficients)

    populations = np.zeros(norms.size)
    populations[support] = coefficients / scale
    predicted = q @ populations
    reconstructed, residual = _spectral_result(spectrum, selected, peak, predicted)
    error = np.sqrt(np.mean((y - predicted) ** 2))
    objective = error**2 + np.linalg.norm(penalty @ coefficients) ** 2
    extra = {}
    if mas is not None:
        predicted_mas = gain * (k @ populations) + baseline
        mr, me = _spectral_result(mas, mas_selected, mas_peak, predicted_mas)
        mas_rms = np.sqrt(np.mean((z - predicted_mas) ** 2))
        objective += mas_weight**2 * mas_rms**2
        extra = dict(
            mas_reconstructed=mr,
            mas_residual=me,
            mas_rms=mas_rms,
            mas_gain=gain,
            mas_baseline=baseline,
        )
    return FitResult(
        population=_map(populations, kernel, "MQMAS population"),
        amplitude=_map(populations * norms, kernel, "MQMAS amplitude"),
        reconstructed=reconstructed,
        residual=residual,
        rms=error,
        objective=objective,
        **extra,
    )


def profile(spectrum, kernel, mas, mas_kernels, *, local_region, weights, **options):
    """Re-fit at each imposed weight and select the best supplied MAS width.

    `mas_kernels` maps candidate Gaussian FWHM values in Hz to kernels
    for the same spectrum and parameter grid. Results retain both residuals
    and the chosen width. On each trial, the gain and remaining population
    variables are reoptimized. Include None in weights for the free fit.
    """
    candidates = tuple((float(width), model) for width, model in mas_kernels.items())
    if not candidates or any(not np.isfinite(w) or w < 0 for w, _ in candidates):
        raise ValueError("Supply at least one finite, nonnegative MAS width.")
    points = []
    for weight in weights:
        fits = (
            (
                width,
                fit(
                    spectrum,
                    kernel,
                    mas=mas,
                    mas_kernel=model,
                    local_region=local_region if weight is not None else None,
                    local_weight=weight,
                    **options,
                ),
            )
            for width, model in candidates
        )
        width, result = min(fits, key=lambda pair: pair[1].objective)
        points.append(
            ProfilePoint(None if weight is None else float(weight), width, result)
        )
    return tuple(points)
