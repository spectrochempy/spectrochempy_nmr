"""Fit a finite RF Czjzek model, optionally starting with a spectral library."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize

from .czjzek import simulate_czjzek
from .czjzek_library import CzjzekRFLibrary
from .pulsar import PhaseCycle


@dataclass(frozen=True)
class CzjzekFitStage:
    """SciPy optimization result, elapsed seconds and exact RF propagation count."""

    optimization: OptimizeResult
    seconds: float
    propagations: int


@dataclass(frozen=True)
class CzjzekFitResult:
    """Fit stages. The final parameter estimates are in `direct.optimization.x`."""

    direct: CzjzekFitStage
    interpolated: CzjzekFitStage | None = None


def fit_czjzek(
    objective,
    x0,
    bounds,
    *,
    events,
    library=None,
    samples=128,
    seed=0,
    method="Powell",
    options=None,
    simulate_direct=simulate_czjzek,
    **settings,
):
    """Minimize a scalar objective with optional library initialization.

    `objective(parameters, simulate)` receives a callable that maps one
    Czjzek distribution to its spectrum. Use it to apply processing or
    include other fit parameters in a single objective for both stages.
    `bounds`, `method` and `options` are shared by both SciPy optimizations.

    Supply the finite RF sequence and simulation settings as for
    :func:`simulate_czjzek`. A supplied library must match those settings
    and sequence. Loading and building the library are timed by the caller.
    This includes ``probe_bandwidth_hz``: a receiver-weighted library
    requires that same bandwidth in the direct refinement settings.
    `simulate_direct` may implement the same API with a different execution
    strategy that calculates one exact response per sampled ensemble member.
    Stage counts include exact fallbacks and exclude separately prepared
    spectra. Refinement starts from the finite interpolated estimate, or
    from the original initialization if the interpolated estimate is not
    finite. The final optimizer status belongs to the direct stage.
    """
    if events is None:
        raise ValueError("A finite RF sequence is required for a Czjzek fit.")
    if isinstance(events, PhaseCycle):
        events = PhaseCycle(
            tuple(tuple(scan) for scan in events.scans),
            None if events.coefficients is None else tuple(events.coefficients),
        )
    else:
        events = tuple(events)
    if bounds is None or method.lower() not in {
        "powell",
        "nelder-mead",
        "cobyqa",
        "cobyla",
        "l-bfgs-b",
        "tnc",
        "slsqp",
        "trust-constr",
    }:
        raise ValueError(
            "Supply bounds and a SciPy minimization method that supports them."
        )
    if library is not None:
        if not isinstance(library, CzjzekRFLibrary):
            raise TypeError("library must be a CzjzekRFLibrary.")
        library._validate_fit_model(events, settings)

    def run(initial, interpolate):
        propagations = 0

        def simulate(distribution):
            nonlocal propagations
            if interpolate:
                spectrum, count = library._simulate(
                    distribution, samples=samples, seed=seed
                )
            else:
                spectrum = simulate_direct(
                    distribution, events=events, samples=samples, seed=seed, **settings
                )
                count = (
                    samples
                    if distribution.sigma_hz or distribution.shift_sigma_hz
                    else 1
                )
            propagations += count
            return spectrum

        start = perf_counter()
        result = minimize(
            objective,
            initial,
            args=(simulate,),
            bounds=bounds,
            method=method,
            options=options,
        )
        return CzjzekFitStage(result, perf_counter() - start, propagations)

    interpolated = None
    initial = x0
    if library is not None:
        interpolated = run(x0, True)
        candidate = interpolated.optimization
        if np.isfinite(candidate.fun) and np.all(np.isfinite(candidate.x)):
            initial = candidate.x
    return CzjzekFitResult(run(initial, False), interpolated)
