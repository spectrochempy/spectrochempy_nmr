"""Conversion from the neutral normalized acquisition contract."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_OPERATION_KEYS = {
    "name",
    "stage",
    "input_dimensions",
    "output_dimensions",
    "axis",
    "scale",
    "conjugation",
    "sign",
    "permutation",
    "evidence",
}


def _plain(value):
    """Convert neutral read-only mappings/diagnostics to copyable metadata."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if all(hasattr(value, field) for field in ("code", "severity", "message", "context")):
        return {
            "code": value.code,
            "severity": value.severity,
            "message": value.message,
            "context": _plain(value.context),
        }
    return value


def _axis_coordinate(axis, size, dim):
    import spectrochempy as scp  # noqa: PLC0415

    if axis.values is None:
        return scp.Coord.arange(size, dims=dim, title=axis.label)
    kwargs = {"title": axis.label}
    if axis.unit is not None:
        kwargs["units"] = axis.unit
    return scp.Coord(np.asarray(axis.values), **kwargs)


def _interop_metadata(result):
    operations = []
    for operation in result.applied_operations:
        if not isinstance(operation, Mapping) or not _OPERATION_KEYS <= set(operation):
            raise ValueError("normalized operation does not satisfy its schema")
        operations.append(_plain(operation))
    axes = []
    for axis in result.axes:
        reference = getattr(axis, "reference", None)
        axes.append(
            {
                "label": axis.label,
                "unit": axis.unit,
                "domain": axis.domain,
                "direction": axis.direction,
                "reference": (
                    {
                        "observation_frequency": reference.observation_frequency,
                        "observation_frequency_unit": reference.observation_frequency_unit,
                        "offset": reference.offset,
                        "offset_unit": reference.offset_unit,
                        "source": reference.source,
                    }
                    if reference is not None
                    else None
                ),
            }
        )
    return {
        "normalization_profile": result.normalization_profile,
        "normalization_state": result.normalization_state,
        "quadrature_mode": result.quadrature_mode,
        "branch_order": tuple(result.branch_order),
        "interop_dimensions": tuple(result.dimensions),
        "interop_physical_dimensions": tuple(result.physical_dimensions),
        "interop_domains": tuple(result.domains),
        "interop_encoding": tuple(result.encoding),
        "provenance": _plain(result.provenance),
        "acquisition": _plain(result.acquisition),
        "applied_operations": tuple(operations),
        "diagnostics": _plain(result.diagnostics),
        "source_metadata": _plain(result.metadata),
        "interop_axes": tuple(axes),
    }


def from_interop(result):
    """Convert one normalized 1D acquisition into a complex NDDataset.

    The converter consumes the neutral normalized protocol by attributes and
    deliberately does not import nmrglue-ng or inspect vendor metadata. A raw
    descriptive ``ReadResult`` is rejected because it has not established the
    normalization guarantees required by this conversion.
    """
    required = (
        "data",
        "axes",
        "dimensions",
        "physical_dimensions",
        "domains",
        "quadrature_mode",
        "branch_order",
        "normalization_state",
        "normalization_profile",
        "applied_operations",
        "provenance",
        "acquisition",
        "metadata",
        "encoding",
        "diagnostics",
    )
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise TypeError(
            "from_interop requires a normalized acquisition; missing "
            + ", ".join(missing)
        )
    data = np.asarray(result.data)
    if data.ndim != 1 or not np.iscomplexobj(data):
        raise ValueError("from_interop currently supports one complex 1D acquisition")
    if tuple(result.dimensions) != ("F2",) or tuple(result.physical_dimensions) != ("F2",):
        raise ValueError("from_interop 1D input must declare the F2 dimension")
    if tuple(result.domains) != ("time",):
        raise ValueError("from_interop 1D input must be in the time domain")
    if len(result.axes) != 1:
        raise ValueError("from_interop 1D input must have one physical axis")
    if result.quadrature_mode != "complex-1d" or tuple(result.branch_order) != ("complex",):
        raise ValueError("from_interop 1D input has an unsupported quadrature contract")
    if result.normalization_state not in ("storage-normalized", "quadrature-normalized"):
        raise ValueError("from_interop requires a validated normalized acquisition")
    if not isinstance(result.provenance, Mapping) or not isinstance(result.acquisition, Mapping):
        raise TypeError("from_interop provenance and acquisition must be mappings")

    import spectrochempy as scp  # noqa: PLC0415

    axis = result.axes[0]
    dataset = scp.NDDataset(
        np.array(data, copy=True),
        coordset=[_axis_coordinate(axis, data.size, "x")],
        dims=["x"],
    )
    dataset.meta.update(_interop_metadata(result))
    dataset.meta.axis_roles = ("F2",)
    dataset.meta.data_domain = "time"
    dataset.meta.isfreq = [False]
    dataset.meta.td = [int(data.size)]
    return dataset
