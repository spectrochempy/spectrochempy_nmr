"""Experimental interpolation library for finite-RF Czjzek spectra."""

from dataclasses import fields
from hashlib import sha256
import json
from operator import index
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy

from . import pulsar
from .czjzek import CzjzekDistribution

_DEFAULTS = dict(
    points=8192,
    accuracy=8,
    spinning_hz=0.0,
    rotor_angle_deg=54.736,
    sidebands=0,
    transitions="central",
    max_rf_step_s=2e-6,
)
_REQUIRED = {"spin", "larmor_hz", "spectral_width_hz"}
_INTEGER_SETTINGS = {"points", "accuracy", "sidebands"}
_FORMAT = 1


class CzjzekRFLibrary:
    """Opt-in interpolator for one finite RF sequence and spectral grid.

    Construct with :meth:`build` or :meth:`load`. The library approximates
    the complex excitation and line shape inside its grids. A fit may use
    :meth:`simulate` for initialization and the direct Czjzek calculation
    for final parameter estimates.
    """

    def __init__(self, events, cq_grid_hz, eta_grid, shift_hz, shift_bins, settings):
        settings = {**_DEFAULTS, **settings}
        if not _REQUIRED <= settings.keys() <= (_REQUIRED | _DEFAULTS.keys()):
            raise ValueError(
                "Supply the RF Czjzek spectral settings, including spin, Larmor frequency and spectral width."
            )
        self._settings = {}
        for key, value in settings.items():
            if key in _INTEGER_SETTINGS:
                value = index(value)
            elif key != "transitions":
                (value,) = pulsar._real_scalars(value)
            self._settings[key] = value
        points = self._settings["points"]
        width = self._settings["spectral_width_hz"]
        (shift_hz,) = pulsar._real_scalars(shift_hz)
        bins = index(shift_bins)
        if points < 2 or width <= 0 or not 0 < bins < points:
            raise ValueError(
                "Use a positive spectral width and 0 < shift_bins < points."
            )
        self._step = width / (points - 1)
        self._shift_hz = shift_hz
        self._bins = bins
        self._axis = np.linspace(-width / 2, width / 2, points)
        self._cq = self._grid(cq_grid_hz)
        self._eta = self._grid(eta_grid)
        if self._eta[0] < 0 or self._eta[-1] > 1:
            raise ValueError("The eta grid must lie within [0, 1].")
        if isinstance(events, pulsar.PhaseCycle):
            self._events = pulsar.PhaseCycle(
                tuple(tuple(scan) for scan in events.scans),
                None if events.coefficients is None else tuple(events.coefficients),
            )
        else:
            self._events = tuple(events)
        self._spectra = None

    @staticmethod
    def _grid(values):
        entries = np.asarray(values, dtype=object)
        message = "Each library grid must have at least two finite, strictly increasing values."
        if entries.ndim != 1 or entries.size < 2:
            raise ValueError(message)
        grid = np.asarray(pulsar._real_scalars(*entries))
        if not np.all(grid[1:] > grid[:-1]):
            raise ValueError(message)
        grid.setflags(write=False)
        return grid

    @property
    def _shape(self):
        return (2, self._cq.size, self._eta.size, self._axis.size)

    def _direct(self, cq, eta, shift):
        return pulsar.simulate_pulse_sequence(
            self._events, cq_hz=cq, eta=eta, shift_hz=shift, **self._settings
        ).data

    @classmethod
    def build(cls, events, *, cq_grid_hz, eta_grid, shift_hz, shift_bins, **settings):
        """Simulate signed-Cq and eta grids at shift_hz plus/minus shift_bins bins."""
        library = cls(events, cq_grid_hz, eta_grid, shift_hz, shift_bins, settings)
        spectra = np.empty(library._shape, dtype=np.complex128)
        for plane, offset in enumerate((-library._bins, library._bins)):
            for i, cq in enumerate(library._cq):
                for j, eta in enumerate(library._eta):
                    profile = library._direct(
                        cq, eta, library._shift_hz + offset * library._step
                    )
                    clipped = profile[:offset] if offset > 0 else profile[offset:]
                    if np.any(clipped):
                        raise ValueError(
                            "The library needs a wider spectral window for shift alignment."
                        )
                    spectra[plane, i, j] = np.roll(profile, -offset)
        spectra.setflags(write=False)
        library._spectra = spectra
        return library

    def _check_spectral_window(self, couplings, asymmetries, shifts):
        settings = self._settings
        spinning = settings["spinning_hz"]
        directions, _, _ = pulsar._powder_mesh(settings["accuracy"], bool(spinning))
        angle = pulsar._rotor_angle(settings["rotor_angle_deg"], spinning)
        extent = settings["sidebands"] * spinning
        transitions = settings["transitions"]
        for cq, eta, shift in zip(couplings, asymmetries, shifts, strict=True):
            for m, _, terms in pulsar._transition_harmonics(
                directions, cq, eta, angle, settings["spin"], settings["larmor_hz"], 0.0
            ):
                if transitions == "central" and m != 0.5:
                    continue
                if transitions == "satellites" and m == 0.5:
                    continue
                offsets = shift + terms[0].real
                if (
                    offsets.min() - extent < self._axis[0]
                    or offsets.max() + extent > self._axis[-1]
                ):
                    raise ValueError(
                        "The spectral window must contain the requested transitions."
                    )

    def simulate(self, distribution, *, samples=128, seed=0):
        """Evaluate the distribution, calculating uncovered members directly."""
        return self._simulate(distribution, samples=samples, seed=seed)[0]

    def _simulate(self, distribution, *, samples, seed):
        if not isinstance(distribution, CzjzekDistribution):
            raise TypeError("distribution must be a CzjzekDistribution.")
        members = distribution.sample(samples, seed)
        cq, eta = members.cq_hz, members.eta
        displacement = (members.shift_hz - self._shift_hz) / self._step
        inside = (
            (self._cq[0] <= cq)
            & (cq <= self._cq[-1])
            & (self._eta[0] <= eta)
            & (eta <= self._eta[-1])
            & (np.abs(displacement) <= self._bins + 1e-12)
        )

        profile = np.zeros(self._axis.size, dtype=complex)
        if np.any(inside):
            cq, eta = cq[inside], eta[inside]
            self._check_spectral_window(cq, eta, members.shift_hz[inside])
            profile += self._interpolate(
                cq, eta, displacement[inside], members.weights[inside]
            )

        uncovered = np.flatnonzero(~inside)
        for k in uncovered:
            profile += members.weights[k] * self._direct(
                members.cq_hz[k], members.eta[k], members.shift_hz[k]
            )
        spectrum = pulsar._as_dataset(self._axis, profile, "Czjzek RF library spectrum")
        return spectrum, uncovered.size

    def _interpolate(self, cq, eta, displacements, populations):
        i = np.searchsorted(self._cq, cq, side="right").clip(1, self._cq.size - 1) - 1
        j = (
            np.searchsorted(self._eta, eta, side="right").clip(1, self._eta.size - 1)
            - 1
        )
        u = (cq - self._cq[i]) / (self._cq[i + 1] - self._cq[i])
        v = (eta - self._eta[j]) / (self._eta[j + 1] - self._eta[j])
        columns = self._eta.size
        origin = i * columns + j
        nodes = origin[:, None] + np.array([0, 1, columns, columns + 1])
        weights = populations[:, None] * np.column_stack(
            ((1 - u) * (1 - v), (1 - u) * v, u * (1 - v), u * v)
        )
        size = self._axis.size
        planes = self._spectra.reshape(2, -1, size)
        frequencies = -2j * np.pi * np.fft.fftfreq(3 * size)
        summed = np.zeros(3 * size, dtype=complex)

        # Combine members at the same displacement before translation. With
        # shift disorder each group usually needs just the four adjacent nodes.
        order = np.argsort(displacements, kind="stable")
        shifts, counts = np.unique(displacements[order], return_counts=True)
        starts = np.r_[0, counts.cumsum()]
        for shift, start, end in zip(shifts, starts[:-1], starts[1:], strict=True):
            selected = order[start:end]
            if shifts.size == 1:
                mass = np.bincount(
                    nodes.ravel(), weights=weights.ravel(), minlength=planes.shape[1]
                )
                lower, upper = mass @ planes
            else:
                indices, inverse = np.unique(nodes[selected], return_inverse=True)
                mass = np.bincount(inverse.ravel(), weights=weights[selected].ravel())
                lower, upper = mass @ planes[:, indices]
            blend = np.clip((shift / self._bins + 1) / 2, 0, 1)
            aligned = (1 - blend) * lower + blend * upper
            summed += np.fft.fft(np.pad(aligned, size)) * np.exp(frequencies * shift)
        return np.fft.ifft(summed)[size : 2 * size]

    @staticmethod
    def _scan_metadata(scan):
        result = []
        for event in scan:
            if isinstance(event, pulsar.Coherence):
                if not isinstance(event.symmetric, (bool, np.bool_)):
                    raise TypeError("Coherence symmetric must be boolean.")
                encoded = ["coherence", index(event.order)]
                if event.symmetric:
                    encoded.append(True)
                result.append(encoded)
            elif isinstance(event, pulsar.Delay):
                (duration,) = pulsar._real_scalars(event.duration_s)
                result.append(["delay", duration, bool(event.decouple_j)])
            else:
                for event_type in (pulsar.Pulse, pulsar.IdealPulse):
                    if isinstance(event, event_type):
                        values = pulsar._real_scalars(
                            *(
                                getattr(event, field.name)
                                for field in fields(event_type)
                            )
                        )
                        result.append([event_type.__name__, *values])
                        break
                else:
                    raise TypeError("Unsupported RF library sequence event.")
        return result

    def _event_metadata(self):
        if not isinstance(self._events, pulsar.PhaseCycle):
            return [[self._scan_metadata(self._events), [1.0, 0.0]]]
        scans = self._events.scans
        coefficients = pulsar._phase_cycle_coefficients(
            scans, self._events.coefficients
        )
        return [
            [self._scan_metadata(scan), [weight.real, weight.imag]]
            for scan, weight in zip(scans, coefficients, strict=True)
        ]

    def _validate_fit_model(self, events, settings):
        requested = CzjzekRFLibrary(
            events, self._cq, self._eta, self._shift_hz, self._bins, settings
        )
        if (
            requested._settings != self._settings
            or requested._event_metadata() != self._event_metadata()
        ):
            raise ValueError(
                "The RF library does not match the fit simulation settings or sequence."
            )

    def _metadata(self):
        return dict(
            format=_FORMAT,
            pulsar=sha256(Path(pulsar.__file__).read_bytes()).hexdigest(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            events=self._event_metadata(),
            settings=self._settings,
            cq=self._cq.tolist(),
            eta=self._eta.tolist(),
            shift_hz=self._shift_hz,
            shift_bins=self._bins,
        )

    def save(self, path):
        """Save the complex, aligned profiles and their simulation settings."""
        path = Path(path)
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary_path = Path(file.name)
            try:
                np.savez(
                    file,
                    metadata=json.dumps(self._metadata(), sort_keys=True),
                    spectra=self._spectra,
                )
                file.close()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(
        cls, path, events, *, cq_grid_hz, eta_grid, shift_hz, shift_bins, **settings
    ):
        """Load a file only if its metadata match the requested model."""
        library = cls(events, cq_grid_hz, eta_grid, shift_hz, shift_bins, settings)
        with np.load(path, allow_pickle=False) as data:
            if json.loads(data["metadata"].item()) != library._metadata():
                raise ValueError(
                    "The library does not match the requested RF/MAS model."
                )
            spectra = data["spectra"]
        if (
            spectra.shape != library._shape
            or spectra.dtype != np.complex128
            or not np.isfinite(spectra).all()
        ):
            raise ValueError("Invalid Czjzek RF library spectra.")
        spectra.setflags(write=False)
        library._spectra = spectra
        return library
