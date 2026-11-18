"""PULSAR responses sampled on calibrated spectral axes."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from . import pulsar
from .processing.mqmas import _mqmas_geometry


@dataclass(frozen=True)
class _MQMASResponse:
    f1_hz: np.ndarray
    f2_hz: np.ndarray
    model: object
    f1_gaussian_fwhm_hz: float
    f2_gaussian_fwhm_hz: float

    def __post_init__(self):
        widths = pulsar._real_scalars(
            self.f1_gaussian_fwhm_hz, self.f2_gaussian_fwhm_hz
        )
        if widths[0] <= 0 or widths[1] < 0:
            raise ValueError("Use positive F1 and nonnegative F2 Gaussian widths.")

    def __call__(self, shift_ppm, cq_hz, eta):
        model = self.model(*pulsar._real_scalars(shift_ppm, cq_hz, eta))
        f1 = np.sort(self.f1_hz)
        step = np.diff(f1).mean()
        spectrum = model.spectrum(
            f1_spectral_width_hz=step * f1.size,
            f1_points=f1.size,
            f1_offset_hz=f1[f1.size // 2],
            f1_gaussian_fwhm_hz=self.f1_gaussian_fwhm_hz,
            f2_gaussian_fwhm_hz=self.f2_gaussian_fwhm_hz,
        )
        axis = spectrum.x.to("Hz").data
        f2_step = np.diff(axis).mean()
        density = spectrum.data.real / (step * f2_step)
        sampled = np.array(
            [
                np.interp(self.f2_hz, axis + f2_step / 2, row, left=0, right=0)
                for row in density
            ]
        )
        return sampled if self.f1_hz[0] < self.f1_hz[-1] else sampled[::-1]

    def with_broadening(self, *, f1_gaussian_fwhm_hz=None, f2_gaussian_fwhm_hz=None):
        """Return a response with new widths sharing the same cached RF profiles."""
        return replace(
            self,
            f1_gaussian_fwhm_hz=(
                self.f1_gaussian_fwhm_hz
                if f1_gaussian_fwhm_hz is None
                else f1_gaussian_fwhm_hz
            ),
            f2_gaussian_fwhm_hz=(
                self.f2_gaussian_fwhm_hz
                if f2_gaussian_fwhm_hz is None
                else f2_gaussian_fwhm_hz
            ),
        )

    def cache_info(self):
        """Return RF-cache hit, miss and size information."""
        return self.model.cache_info()

    def cache_clear(self):
        """Release RF profiles, including those shared by other width views."""
        self.model.cache_clear()


def mqmas_response(
    f1_ppm,
    f2_ppm,
    preparation,
    conversion,
    *,
    spin,
    larmor_hz,
    spinning_hz,
    spectral_width_hz,
    f1_gaussian_fwhm_hz,
    f2_gaussian_fwhm_hz=0,
    carrier_ppm=0,
    f1_offset_ppm=0,
    **options,
):
    """Return a signed fast MQMAS response for build_kernel on calibrated ppm axes.

    The callable accepts (shift_ppm, cq_hz, eta), retaining finite-RF
    excitation/conversion losses and the phased real part of both States
    quadratures. F1 must be uniform; either axis may increase or decrease.
    The analytical F1 position must lie within the F1 bin edges. F2 values
    outside the simulated window are zero. Frequency-density values are
    per Hz squared: F1 is averaged over each bin and F2 is interpolated
    at PULSAR bin centers. No column or peak normalization is applied.

    Options go to MQMASModel; receiver_phase_deg phases the response.
    Carrier is common to both axes; f1_offset_ppm is an extra F1 reference
    shift. Other acquisition/processing conventions must be supplied by
    the caller. Fixed-transfer approximation limits are those of MQMASModel.

    Each distinct shift/Cq/eta caches two RF profiles. with_broadening()
    returns a width view sharing that cache. cache_info() reports its size,
    and cache_clear() releases it for all views. Kernels remain independent
    immutable matrices and can be reused by fit() after cache clearing.
    """
    from .mqmas_inversion import _axis

    larmor_hz, carrier_ppm, f1_offset_ppm = pulsar._real_scalars(
        larmor_hz, carrier_ppm, f1_offset_ppm
    )
    if larmor_hz <= 0:
        raise ValueError("larmor_hz must be positive.")
    if {"cq_hz", "pq_hz", "eta", "shift_hz"} & options.keys():
        raise ValueError("Shift, Cq and eta are supplied by the parameter grid.")
    f1 = (
        (_axis(f1_ppm, "F1 coordinate") - carrier_ppm - f1_offset_ppm)
        * larmor_hz
        * 1e-6
    )
    f2 = (_axis(f2_ppm, "F2 coordinate") - carrier_ppm) * larmor_hz * 1e-6
    steps = np.diff(f1)
    if not np.allclose(steps, steps.mean(), rtol=1e-6, atol=0):
        raise ValueError("The fast MQMAS response requires a uniform F1 grid.")
    f1.setflags(write=False)
    f2.setflags(write=False)
    if isinstance(preparation, pulsar.PhaseCycle):
        preparation = pulsar.PhaseCycle(
            tuple(tuple(scan) for scan in preparation.scans),
            None
            if preparation.coefficients is None
            else tuple(preparation.coefficients),
        )
    else:
        preparation = tuple(preparation)
    conversion = tuple(conversion)

    @lru_cache(maxsize=None)
    def model(shift_ppm, cq_hz, eta):
        return pulsar.MQMASModel(
            preparation,
            conversion,
            spin=spin,
            larmor_hz=larmor_hz,
            spinning_hz=spinning_hz,
            spectral_width_hz=spectral_width_hz,
            shift_hz=(shift_ppm - carrier_ppm) * larmor_hz * 1e-6,
            cq_hz=cq_hz,
            eta=eta,
            **options,
        )

    return _MQMASResponse(f1, f2, model, f1_gaussian_fwhm_hz, f2_gaussian_fwhm_hz)


def mas_response(
    axis_ppm,
    events,
    *,
    spin,
    larmor_hz,
    spinning_hz,
    spectral_width_hz,
    gaussian_fwhm_hz=0,
    carrier_ppm=0,
    **options,
):
    """Return a signed MAS template generator for a frequency axis in ppm.

    Pass a PULSAR pulse sequence and its experimental carrier. The
    simulation options are forwarded to `pulsar.simulate_pulse_sequence`.
    """
    events = tuple(events)
    frequency = (np.asarray(axis_ppm, dtype=float) - carrier_ppm) * larmor_hz * 1e-6

    def response(shift_ppm, cq_hz, eta):
        spectrum = pulsar.simulate_pulse_sequence(
            events,
            spin=spin,
            larmor_hz=larmor_hz,
            spinning_hz=spinning_hz,
            spectral_width_hz=spectral_width_hz,
            shift_hz=(shift_ppm - carrier_ppm) * larmor_hz * 1e-6,
            cq_hz=cq_hz,
            eta=eta,
            output="spectrum",
            **options,
        )
        spectrum = pulsar.broaden(spectrum, gaussian_fwhm_hz=gaussian_fwhm_hz)
        hz = spectrum.x.to("Hz").data
        spacing = hz[1] - hz[0]
        return np.interp(
            frequency,
            hz + spacing / 2,
            spectrum.data.real / spacing,
            left=0.0,
            right=0.0,
        )

    return response


def _powder_nodes(order):
    z, weights = np.polynomial.legendre.leggauss(order)
    z = (z + 1) / 2
    phi = np.arange(2 * order) * np.pi / order
    radius = np.sqrt(1 - z**2)
    directions = np.array(
        [
            np.outer(radius, np.cos(phi)),
            np.outer(radius, np.sin(phi)),
            np.broadcast_to(z[:, None], (order, phi.size)),
        ]
    ).reshape(3, -1)
    return directions, np.repeat(weights / (2 * phi.size), phi.size)


def _pulse(offsets, terms, phase, rf, duration, spinning_hz, max_step):
    count = int(np.ceil(duration / max_step))
    step = duration / count
    advance = 2 * np.pi * spinning_hz * step
    result = np.eye(rf.shape[0])
    orders = np.arange(1, 5)
    for index in range(count):
        midpoint = phase + advance * (index + 0.5)
        rotation = np.exp(-1j * orders[:, None] * midpoint)
        differences = offsets[:, None, :] + (terms @ rotation).real.swapaxes(-1, -2)
        result = pulsar._pulse_propagator(rf, differences, step) @ result
    return result, phase + count * advance


def z_filtered_3qmas_response(
    f1_ppm,
    f2_ppm,
    pulses,
    *,
    spin,
    larmor_hz,
    spinning_hz,
    dwell_s,
    increments,
    f2_sigma_ppm,
    carrier_ppm=0,
    conversion_delay_s=0,
    z_filter_s=0,
    f1_lb_hz=0,
    first_point_scale=0.5,
    f1_offset_ppm=0,
    receiver_phase_deg=0,
    powder_order=5,
    rotor_count=12,
    max_rf_step_s=0.25e-6,
):
    """Return a template generator for sheared inner-3QMAS with a z filter.

    Supply the excitation, conversion and readout as three PULSAR
    rectangular pulses. F1 uses the isotropic powder-mean frequency, the
    finite time grid and an exponential window. F2 models the centerband
    CT powder pattern with Gaussian broadening and orientation-dependent
    transfer. Dwell and delays are in seconds, and the spectral axes are
    in ppm with F1 on the isotropic scale.
    """
    order, ratio, scale = _mqmas_geometry(spin, 3)
    f1, f2 = np.asarray(f1_ppm), np.asarray(f2_ppm)
    positive = (larmor_hz, spinning_hz, dwell_s, f2_sigma_ppm, max_rf_step_s)
    nonnegative = (conversion_delay_s, z_filter_s, f1_lb_hz, first_point_scale)
    if not np.isfinite(
        (*positive, *nonnegative, carrier_ppm, f1_offset_ppm, receiver_phase_deg)
    ).all():
        raise ValueError("The 3QMAS acquisition parameters must be finite.")
    if min(positive) <= 0 or min(nonnegative) < 0:
        raise ValueError("Invalid timing, frequency or broadening for 3QMAS.")
    for count in (increments, powder_order, rotor_count):
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 2
        ):
            raise ValueError(
                "Increments, powder order and rotor count must be integers of at least two."
            )
    pulses = tuple(pulses)
    if len(pulses) != 3 or any(not isinstance(p, pulsar.Pulse) for p in pulses):
        raise ValueError("Supply three rectangular PULSAR Pulse events.")
    if any(
        not np.isfinite((p.duration_s, p.rf_hz, p.phase_deg, p.offset_hz)).all()
        or p.duration_s <= 0
        or p.partner_rf_hz
        or p.partner_offset_hz
        for p in pulses
    ):
        raise ValueError("Expected finite single-spin pulses of positive duration.")

    directions, powder_weights = _powder_nodes(powder_order)
    phases = np.arange(rotor_count) * (2 * np.pi / rotor_count)
    rotor_angle = np.arccos(1 / np.sqrt(3))
    ix, iy, iz = pulsar._spin_operators(spin)
    rf = []
    for pulse in pulses:
        phase = np.deg2rad(pulse.phase_deg)
        rf.append(pulse.rf_hz * (np.cos(phase) * ix + np.sin(phase) * iy))
    left = (iz.shape[0] - order - 1) // 2
    right = left + order
    inner = slice(left, right)
    ct = iz.shape[0] // 2 - 1
    time = np.arange(increments) * dwell_s
    window = np.exp(-np.pi * f1_lb_hz * time)
    window[0] *= first_point_scale
    if not window.sum():
        raise ValueError("The indirect acquisition window has zero weight.")
    window /= window.sum()
    receiver = np.exp(1j * np.deg2rad(receiver_phase_deg))
    hz_per_ppm = larmor_hz * 1e-6

    def response(shift_ppm, cq_hz, eta):
        harmonics = np.stack(
            [
                values.T
                for _, _, values in pulsar._transition_harmonics(
                    directions, cq_hz, eta, rotor_angle, spin, larmor_hz, 0.0
                )
            ],
            axis=1,
        )
        offsets, terms = harmonics[..., 0].real, harmonics[..., 1:]
        detuning = offsets + (shift_ppm - carrier_ppm) * hz_per_ppm

        def propagate(index, phase):
            pulse = pulses[index]
            return _pulse(
                detuning - pulse.offset_hz,
                terms,
                phase,
                rf[index],
                pulse.duration_s,
                spinning_hz,
                max_rf_step_s,
            )

        excite, phase = propagate(0, phases)
        rho = pulsar._apply_propagator(iz, excite)
        end = phase + 2 * np.pi * spinning_hz * conversion_delay_s
        evolution = (
            2 * np.pi * detuning[:, None, inner].sum(axis=-1) * conversion_delay_s
        )
        evolution = evolution + (
            pulsar._phase_integral(terms[:, inner], end, spinning_hz)
            - pulsar._phase_integral(terms[:, inner], phase, spinning_hz)
        ).sum(axis=-1)
        coherence = rho[..., right, left] * np.exp(-1j * evolution)
        convert, phase = propagate(1, end)
        populations = (
            2
            * (
                convert[..., :, right]
                * coherence[..., None]
                * convert[..., :, left].conj()
            ).real
        )
        read, phase = propagate(2, phase + 2 * np.pi * spinning_hz * z_filter_s)
        signal = (
            -2j
            * ix[ct, ct + 1]
            * np.sum(
                read[..., ct, :] * populations * read[..., ct + 1, :].conj(), axis=-1
            )
        )
        wave = np.exp(
            1j
            * pulsar._phase_integral(terms[:, ct : ct + 1], phase, spinning_hz)[..., 0]
        )
        centerband = (signal * wave.conj()).mean(axis=-1) * wave.mean(axis=-1)
        center = (
            shift_ppm
            + f1_offset_ppm
            + (
                (offsets[:, inner].sum(axis=-1) - ratio * offsets[:, ct])
                @ powder_weights
                / (scale * hz_per_ppm)
            )
        )
        indirect = (
            np.cos(2 * np.pi * scale * hz_per_ppm * np.outer(f1 - center, time))
            @ window
        )
        direct = (f2[:, None] - shift_ppm - offsets[:, ct] / hz_per_ppm) / f2_sigma_ppm
        direct = np.exp(-0.5 * direct**2) @ (
            powder_weights * (receiver * centerband).real
        )
        return np.outer(indirect, direct / (np.sqrt(2 * np.pi) * f2_sigma_ppm))

    return response
