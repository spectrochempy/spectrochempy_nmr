"""Isotropic Gaussian EFG disorder and extended Czjzek powder spectra.

Tensors are expressed in coupling-frequency units. ``sigma_hz`` is the
standard deviation of each coordinate in a Frobenius-orthonormal basis
of symmetric traceless tensors. This fixes the scale independently of
other programs' conventions for their Czjzek width parameters.
"""

from dataclasses import dataclass
from functools import lru_cache
from operator import index

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from .pulsar import PhaseCycle
from .pulsar import _as_dataset
from .pulsar import _central_transition_profile
from .pulsar import _normalize_profile
from .pulsar import _real_scalars
from .pulsar import simulate_pulse_sequence

# Unit Frobenius norms make independent equal-variance coefficients an
# isotropic Gaussian in the five-dimensional space of EFG tensors.
_BASIS = np.array(
    [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[-1, 0, 0], [0, -1, 0], [0, 0, 2]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    ],
    dtype=float,
)
_BASIS /= np.sqrt([2, 6, 2, 2, 2])[:, None, None]


@lru_cache(maxsize=8)
def _normal_nodes(size, seed):
    # Scrambling avoids the endpoints of the inverse-normal transform.
    # Reusing these standardized nodes keeps fitting evaluations deterministic.
    uniforms = qmc.Sobol(6, scramble=True, seed=seed).random_base2(
        size.bit_length() - 1
    )
    nodes = ndtri(np.clip(uniforms, np.finfo(float).eps, 1 - np.finfo(float).eps))
    nodes.setflags(write=False)
    return nodes


def _tensor_parameters(tensors):
    """Return signed Cq and eta from symmetric traceless coupling tensors."""
    values = np.linalg.eigvalsh(tensors)
    ordered = np.take_along_axis(values, np.argsort(np.abs(values), axis=-1), axis=-1)
    cq = ordered[..., 2]
    eta = np.divide(
        ordered[..., 0] - ordered[..., 1],
        cq,
        out=np.zeros_like(cq),
        where=cq != 0,
    )
    return cq, np.clip(eta, 0, 1)


@dataclass(frozen=True)
class CzjzekSamples:
    """Sampled couplings, asymmetries, shifts and equal population weights.

    Array values returned by :meth:`CzjzekDistribution.sample` are read-only.
    Frequencies are in Hz and ``eta`` lies between zero and one.
    """

    cq_hz: np.ndarray
    eta: np.ndarray
    shift_hz: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class CzjzekDistribution:
    """Standard or extended Czjzek EFG distribution with Gaussian shifts.

    ``sigma_hz`` is the Gaussian standard deviation of each orthonormal
    tensor coordinate. ``cq_hz`` and ``eta`` define a fixed local tensor
    added to that noise. A zero local tensor gives the standard model.
    The independent chemical shift has mean ``shift_hz`` and standard
    deviation ``shift_sigma_hz``. With zero noise both tensors and shifts
    reduce to the supplied local values.

    For the standard model, ``Pq = abs(Cq)*sqrt(1+eta**2/3)`` has a chi
    distribution with five degrees of freedom and scale sqrt(2/3)*sigma_hz.
    """

    sigma_hz: float
    cq_hz: float = 0.0
    eta: float = 0.0
    shift_hz: float = 0.0
    shift_sigma_hz: float = 0.0

    def __post_init__(self):
        values = _real_scalars(
            self.sigma_hz, self.cq_hz, self.eta, self.shift_hz, self.shift_sigma_hz
        )
        if values[0] < 0 or values[4] < 0:
            raise ValueError("Distribution standard deviations must be nonnegative.")
        if not 0 <= values[2] <= 1:
            raise ValueError("eta must lie between zero and one.")

    def sample(self, size=256, seed=0):
        """Return deterministic, equally weighted Sobol Gaussian samples.

        ``size`` must be a power of two between 1 and 65536. ``seed`` is
        an integer from 0 through 2**32-1. A fully deterministic model
        returns a single member regardless of the requested sample count.
        """
        size, seed = index(size), index(seed)
        if size < 1 or size > 65536 or size & (size - 1):
            raise ValueError("size must be a power of two between 1 and 65536.")
        if not 0 <= seed < 2**32:
            raise ValueError("seed must be between 0 and 2**32-1.")
        if self.sigma_hz == 0 and self.shift_sigma_hz == 0:
            arrays = [
                np.array([v], dtype=float)
                for v in (
                    self.cq_hz,
                    self.eta if self.cq_hz else 0.0,
                    self.shift_hz,
                    1.0,
                )
            ]
        else:
            nodes = _normal_nodes(size, seed)
            local = np.diag(
                [
                    self.cq_hz * (self.eta - 1) / 2,
                    -self.cq_hz * (self.eta + 1) / 2,
                    self.cq_hz,
                ]
            )
            tensors = local + self.sigma_hz * np.einsum(
                "ni,ijk->njk", nodes[:, :5], _BASIS
            )
            if not np.isfinite(tensors).all():
                raise ValueError(
                    "The tensor distribution exceeds floating point range."
                )
            if self.sigma_hz == 0:
                cq = np.full(size, self.cq_hz, dtype=float)
                eta = np.full(size, self.eta if self.cq_hz else 0.0, dtype=float)
            else:
                cq, eta = _tensor_parameters(tensors)
            shifts = self.shift_hz + self.shift_sigma_hz * nodes[:, 5]
            if not np.isfinite(shifts).all():
                raise ValueError("The shift distribution exceeds floating point range.")
            arrays = [cq, eta, shifts, np.full(size, 1 / size)]
        for array in arrays:
            array.setflags(write=False)
        return CzjzekSamples(*arrays)


def simulate_czjzek(
    distribution,
    *,
    spin,
    larmor_hz,
    spectral_width_hz,
    samples=128,
    seed=0,
    points=8192,
    accuracy=8,
    spinning_hz=0.0,
    rotor_angle_deg=54.736,
    sidebands=0,
    transitions="central",
    events=None,
    max_rf_step_s=2e-6,
):
    """Integrate a Czjzek ensemble into a one-dimensional powder NDDataset.

    Ideal spectra have unit total area over the selected transitions and
    MAS orders after integrating the ensemble. ``events`` optionally
    supplies a PULSAR pulse sequence. Finite-sequence spectra retain the
    simulator's ideal-reference excitation scale, including losses and
    phase. The frequency origin and bin conventions follow PULSAR.
    Increasing ``samples`` refines the distribution independently of the
    powder orientation ``accuracy``. Work is streamed one tensor at a time.
    """
    if not isinstance(distribution, CzjzekDistribution):
        raise TypeError("distribution must be a CzjzekDistribution.")
    members = distribution.sample(samples, seed)
    if isinstance(events, PhaseCycle):
        sequence = PhaseCycle(
            tuple(tuple(scan) for scan in events.scans),
            None if events.coefficients is None else tuple(events.coefficients),
        )
    else:
        sequence = None if events is None else tuple(events)
    settings = dict(
        spin=spin,
        larmor_hz=larmor_hz,
        spectral_width_hz=spectral_width_hz,
        points=points,
        accuracy=accuracy,
        spinning_hz=spinning_hz,
        rotor_angle_deg=rotor_angle_deg,
        sidebands=sidebands,
        transitions=transitions,
    )
    total = None
    for cq, eta, shift, weight in zip(
        members.cq_hz, members.eta, members.shift_hz, members.weights, strict=True
    ):
        parameters = dict(cq_hz=cq, eta=eta, shift_hz=shift, **settings)
        if sequence is None:
            axis, profile = _central_transition_profile(**parameters, _normalize=False)
        else:
            result = simulate_pulse_sequence(
                sequence, **parameters, max_rf_step_s=max_rf_step_s
            )
            # The dataset coordinates are rounded for display. Recover the
            # uniform simulator grid before constructing the final dataset.
            axis = np.linspace(
                -spectral_width_hz / 2, spectral_width_hz / 2, index(points)
            )
            profile = result.data
        if total is None:
            total = weight * profile
        else:
            total += weight * profile
    if sequence is None:
        total = _normalize_profile(total)
    return _as_dataset(axis, total, "Czjzek powder spectrum")
