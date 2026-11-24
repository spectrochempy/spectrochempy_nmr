"""A small, unit-aware PFG workflow for prepared spectra and Bruker diffSte."""

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

import spectrochempy as scp
from spectrochempy.core.units import DimensionalityError


def _b_coord(values, count):
    if isinstance(values, scp.Coord):
        coord = scp.Coord(values.data, units=values.units, rounding=False, sigdigits=15)
        coord.mask = values.mask
    elif hasattr(values, "units") and hasattr(values, "magnitude"):
        coord = scp.Coord(
            values.magnitude, units=values.units, rounding=False, sigdigits=15
        )
    else:
        raise ValueError("Supply b as a Coord or quantity with units of s/m^2")
    try:
        coord = coord.to("s/m^2")
    except (ValueError, TypeError, DimensionalityError) as exc:
        raise ValueError(
            "Diffusion weights b must have units compatible with s/m^2"
        ) from exc
    if np.iscomplexobj(coord.data):
        raise ValueError("Diffusion weights b must be real")
    data = np.asarray(coord.data, dtype=float)
    if (
        np.any(coord.mask)
        or data.ndim != 1
        or data.size != count
        or not np.isfinite(data).all()
        or np.any(data < 0)
        or np.unique(data).size < 4
    ):
        raise ValueError(
            "Supply one finite nonnegative b per spectrum, "
            "with at least four distinct values"
        )
    coord.title = "Diffusion weighting"
    coord.meta["units_source"] = "explicit s/m^2 conversion"
    return coord


def _exported_b(path):
    try:
        root = ElementTree.parse(path / "diff.xml").getroot()
    except (FileNotFoundError, ElementTree.ParseError) as exc:
        raise ValueError(
            "Keep diff.xml with the diffSte acquisition, "
            "or supply calibrated b with units"
        ) from exc
    if root.findtext("pulprog") != "diffSte":
        raise ValueError(
            "Automatic b reading supports Bruker diffSte only; "
            "supply calibrated b with units"
        )
    rows = root.findall("./XValues/X")
    try:
        indices = [int(row.attrib["i"]) for row in rows]
        values = [float(row.attrib["B"]) for row in rows]
        count = int(root.findtext("nXValues"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("diff.xml must contain indexed diffusion weights B") from exc
    if indices != list(range(count)) or len(values) != count:
        raise ValueError("diff.xml row indices must match the acquired spectrum order")
    return scp.Coord(values, units="s/m^2", rounding=False, sigdigits=15)


def _validate_spectra(spectra):
    if spectra.ndim != 2 or spectra.shape[0] < 4 or spectra.shape[1] < 2:
        raise ValueError(
            "Diffusion requires a two-dimensional series of at least four spectra"
        )
    if np.iscomplexobj(spectra.data):
        raise ValueError(
            "Supply phased, real spectra; inspect the phase before fitting"
        )
    if np.any(spectra.mask) or not np.isfinite(spectra.data).all():
        raise ValueError("Diffusion spectra must contain finite, unmasked values")
    if spectra.x is None or spectra.x.units != scp.ur.ppm:
        raise ValueError("The direct spectral axis must be in ppm")
    x = np.asarray(spectra.x.data, dtype=float)
    steps = np.diff(x)
    if not np.isfinite(x).all() or not (np.all(steps > 0) or np.all(steps < 0)):
        raise ValueError("The ppm axis must be finite and strictly monotonic")
    if not np.allclose(steps, steps[0], rtol=1e-5, atol=0):
        raise ValueError("PFG area summation requires a uniformly spaced ppm axis")


def _decay(x, amplitude, rate):
    return amplitude * np.exp(-x * rate)


@dataclass(frozen=True)
class DiffusionFit:
    """A single-band monoexponential fit, I(b) = I0 exp(-b D).

    Attributes
    ----------
    D : Quantity
        Nonnegative apparent diffusion coefficient in m²/s.
    I0 : Quantity
        Fitted area at zero diffusion weighting, in the measured area units.
    areas, fitted, residual : NDDataset
        Measured areas, reconstructed areas, and measured minus reconstructed
        areas, on the same b axis in s/m². The fit is unweighted, without offset.
    rmse : Quantity
        Root mean squared residual, in the area units.
    provenance : dict
        Source of the b axis and the processing/integration choices.
    diagnostics : dict
        Optimizer diagnostics, including success and a boundary flag. No
        uncertainty on D or independent gradient calibration is inferred.
    """

    D: object
    I0: object
    areas: object
    fitted: object
    residual: object
    rmse: object
    provenance: dict
    diagnostics: dict

    def plot(self, **kwargs):
        """Display measured/fitted areas and the residual in two panels."""
        order = np.argsort(self.areas.x.data)
        axis = self.areas.x[order]
        curves = scp.NDDataset(
            np.vstack([self.areas.data[order], self.fitted.data[order]]),
            coordset=[scp.Coord(labels=["Measured", "Fit"]), axis],
            units=self.areas.units,
            title="Integrated area",
        )
        residual = scp.NDDataset(
            self.residual.data[order][None, :],
            coordset=[scp.Coord(labels=["Residual"]), axis.copy()],
            units=self.residual.units,
            title="Area residual",
        )
        kwargs.setdefault("ylabel", f"Area / {self.areas.units:~P}")
        kwargs.setdefault("figsize", (8, 6))
        return scp.multiplot(
            [curves, residual],
            nrow=2,
            ncol=1,
            labels=["Measured (blue) and fitted (orange) areas", "Residual"],
            **kwargs,
        )


class Diffusion:
    """Measure an apparent diffusion coefficient from a PFG series.

    Parameters
    ----------
    source : path-like or NDDataset
        A Bruker diffSte acquisition folder (or its ``ser`` file), or a series
        of already phased, real spectra with a uniformly spaced ppm axis.
        Rows correspond to diffusion weights; no Fourier transform is applied
        to that series axis.
    b : Coord or Quantity, optional
        One calibrated diffusion weight per row, convertible to s/m². Bare
        numbers are refused. For a diffSte folder, the default reads the B
        table in ``diff.xml`` in acquired row order. These are exported values,
        not a new calibration of the gradients. Prepared spectra require b.

    Notes
    -----
    This is a single-band monoexponential measurement, not a DOSY inversion.
    Automatic raw processing is currently limited to Bruker diffSte QF series
    with DQD direct acquisition, known digital delay and saved phase.
    """

    def __init__(self, source, *, b=None):
        self._raw = None
        self._spectra = None
        self._processing = {}
        if isinstance(source, scp.NDDataset):
            _validate_spectra(source)
            self._spectra = source.copy()
            self._b = _b_coord(b, source.shape[0])
            self._source = {"b_source": "caller", "spectra_source": "prepared spectra"}
        else:
            from . import read

            path = Path(source).expanduser()
            path = path.parent if path.name == "ser" else path
            raw = read(path / "ser", remove_digital_filter=False)
            if raw is None or raw.ndim != 2 or not np.iscomplexobj(raw.data):
                raise ValueError(
                    "Expected one complex Bruker SER trace per diffusion weight"
                )
            acquisition = raw.meta.get("source_metadata", {}).get("acqus", {})
            indirect = raw.meta.get("source_metadata", {}).get("acqu2s", {})
            if (
                acquisition.get("PULPROG") != "diffSte"
                or acquisition.get("AQ_mod") != 3
                or indirect.get("FnMODE") != 1
            ):
                raise ValueError(
                    "Raw diffusion processing supports Bruker diffSte with QF "
                    "rows and DQD F2; supply prepared spectra and calibrated b "
                    "for other sequences"
                )
            self._b = _b_coord(_exported_b(path) if b is None else b, raw.shape[0])
            self._raw = raw
            self._source = {
                "b_source": str(path / "diff.xml") if b is None else "caller",
                "spectra_source": str(path / "ser"),
                "sequence": "diffSte",
            }
        self._set_b()

    def _set_b(self):
        if self._spectra is not None:
            self._spectra.y = self._b.copy()
            readonly = self._spectra.meta.readonly
            self._spectra.meta.readonly = False
            self._spectra.meta["diffusion"] = {
                **self._source,
                "processing": dict(self._processing),
            }
            self._spectra.meta.readonly = readonly

    @property
    def b(self):
        """Diffusion weights in s/m², returned as a copy."""
        return self._b.copy()

    def process(self, *, size=65536, lb=0.0, phc0=None, phc1=None):
        """Transform only F2 and apply one phase to all rows.

        ``size`` is the FFT size, without truncating acquired samples. ``lb``
        is exponential broadening in Hz (a number or quantity). Phase values
        are in degrees and default to the saved phase. The integer digital
        delay is removed in time; its fractional remainder is corrected on
        the final FFT grid. Repeated calls start from the original SER.
        Prepared spectra are returned unchanged; processing options apply
        only to a raw acquisition.
        """
        if self._raw is None:
            return self._spectra.copy()
        from .processing.bruker_filter import resolve_bruker_delay
        from .processing.digital_filter import correct_delay_time

        if (
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size < self._raw.shape[-1]
        ):
            raise ValueError(
                "size must be an integer at least as large as the acquired F2 trace"
            )
        delay, delay_source = resolve_bruker_delay(self._raw, None)
        if not np.isfinite(delay) or delay < 0 or delay + 2 >= self._raw.shape[-1]:
            raise ValueError("Invalid digital group delay for this F2 trace")
        broadening = (
            scp.ur.Quantity(lb, "Hz") if not hasattr(lb, "units") else lb.to("Hz")
        )
        if not np.isfinite(broadening.magnitude) or broadening.magnitude < 0:
            raise ValueError("lb must be a finite nonnegative broadening in Hz")
        phases = []
        for name, override in (("phc0", phc0), ("phc1", phc1)):
            stored = self._raw.meta.get(name)
            value = (
                override if override is not None else (stored[-1] if stored else None)
            )
            if value is None:
                raise ValueError(f"No saved {name}; supply a common phase in degrees")
            value = value.to("degree").magnitude if hasattr(value, "units") else value
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            phases.append(float(value))
        corrected = correct_delay_time(self._raw.data, delay)
        fid = self._raw[:, : corrected.shape[-1]].copy()
        fid.data = corrected
        if broadening.magnitude:
            fid = fid.em(lb=broadening, dim="x")
        fft = fid.fft(dim="x", size=int(size))
        fraction = delay - np.floor(delay)
        spectra = fft.pk(
            dim="x",
            phc0=(phases[0] - 180 * fraction) * scp.ur.degree,
            phc1=(phases[1] + 360 * fraction) * scp.ur.degree,
        ).real
        _validate_spectra(spectra)
        self._processing = {
            "fft_size": int(size),
            "lb_hz": float(broadening.magnitude),
            "phc0_deg": phases[0],
            "phc1_deg": phases[1],
            "group_delay_points": float(delay),
            "delay_source": delay_source,
        }
        readonly = spectra.meta.readonly
        spectra.meta.readonly = False
        acquisition = dict(spectra.meta.get("acquisition", {}))
        acquisition["digital_filter"] = {
            "applied": True,
            "group_delay_points": float(delay),
            "delay_source": delay_source,
            "method": "integer time correction and fractional F2 phase",
        }
        spectra.meta["acquisition"] = acquisition
        spectra.meta.readonly = readonly
        self._spectra = spectra
        self._set_b()
        return self._spectra.copy()

    def fit(self, *, region):
        """Integrate a ppm band and fit I(b) = I0 exp(-b D).

        ``region=(low, high)`` gives two ppm limits in either order. The
        nearest bins are included at both ends. Areas use equal bin weights
        times the positive ppm spacing. No baseline correction or per-row
        normalization is applied. Inspect the spectra before fitting.
        Returns a :class:`DiffusionFit` with D, measured/fitted areas and
        residual. If needed, call ``process`` first with your phase settings.
        """
        if self._spectra is None:
            self.process()
        spectra = self._spectra
        bounds = np.asarray(region, dtype=float)
        if (
            bounds.shape != (2,)
            or not np.isfinite(bounds).all()
            or bounds[0] == bounds[1]
        ):
            raise ValueError("region must contain two different finite ppm limits")
        x = np.asarray(spectra.x.data, dtype=float)
        low, high = sorted(bounds)
        if low < x.min() or high > x.max():
            raise ValueError("region lies outside the measured ppm axis")
        start, stop = sorted(int(np.argmin(np.abs(x - value))) for value in bounds)
        if stop <= start:
            raise ValueError("region must span at least two spectral points")
        values = spectra.data[:, start : stop + 1].sum(axis=1) * abs(x[1] - x[0])
        scale = float(np.max(np.abs(values)))
        if scale == 0 or values[np.argmin(self._b.data)] <= 0:
            raise ValueError(
                "The selected band must have a positive area at the lowest b; "
                "check the phase and baseline"
            )
        bscale = float(np.max(self._b.data))
        data = scp.NDDataset(
            values / scale,
            coordset=[scp.Coord(self._b.data / bscale, rounding=False, sigdigits=15)],
        )
        optimizer = scp.Optimize(usermodels={"pfg_decay": _decay})
        optimizer.script = """
MODEL: diffusion
shape: pfg_decay
$ amplitude: 1, 0, none
$ rate: 1, 0, none
"""
        result = optimizer.fit(data).result
        diagnostics = dict(result.diagnostics)
        if not diagnostics["success"]:
            raise RuntimeError(f"Diffusion fit failed: {diagnostics['message']}")
        rate = float(optimizer.fp["rate_diffusion"])
        amplitude = float(optimizer.fp["amplitude_diffusion"]) * scale
        predicted = amplitude * np.exp(-self._b.data / bscale * rate)
        units = spectra.units * scp.ur.ppm if spectra.units is not None else scp.ur.ppm
        areas = scp.NDDataset(
            values, coordset=[self._b.copy()], units=units, title="Integrated area"
        )
        fitted = scp.NDDataset(
            predicted, coordset=[self._b.copy()], units=units, title="Fitted area"
        )
        residual = areas - fitted
        residual.title = "Area residual"
        provenance = {
            **self._source,
            "processing": dict(self._processing),
            "region_ppm": (float(low), float(high)),
            "integration": "inclusive nearest bins, equal weights times ppm spacing",
            "model": "I0 * exp(-b * D), unweighted, no offset",
        }
        for curve in (areas, fitted, residual):
            curve.meta["diffusion"] = provenance.copy()
        diagnostics["at_lower_bound"] = rate < 1e-6
        return DiffusionFit(
            D=rate / bscale * scp.ur("m^2/s"),
            I0=amplitude * units,
            areas=areas,
            fitted=fitted,
            residual=residual,
            rmse=float(np.sqrt(np.mean((values - predicted) ** 2))) * units,
            provenance=provenance,
            diagnostics=diagnostics,
        )
