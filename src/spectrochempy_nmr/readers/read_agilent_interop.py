"""Experimental Agilent 1D facade using nmrglue-ng interop."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np


def _acquisition_paths(paths, *, directory=None):
    if len(paths) == 1 and isinstance(paths[0], (list, tuple)):
        paths = tuple(paths[0])
    if not paths:
        raise ValueError("at least one Agilent acquisition path is required")
    base = Path(directory) if directory is not None else None
    found = []
    seen = set()
    for value in paths:
        path = Path(value)
        if base is not None and not path.is_absolute():
            path = base / path
        if path.is_file() and path.name == "fid":
            acquisition = path.parent
        elif path.is_dir():
            acquisition = path
        else:
            raise FileNotFoundError(f"Agilent acquisition path does not exist: {path}")
        fid = acquisition / "fid"
        procpar = acquisition / "procpar"
        if not fid.is_file() or not procpar.is_file():
            raise ValueError(
                f"not an Agilent acquisition directory (fid/procpar required): {acquisition}"
            )
        key = acquisition.resolve()
        if key not in seen:
            seen.add(key)
            found.append(key)
    return tuple(found)


def _known_equal(first, second):
    if first is None or second is None:
        return True
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        for key in set(first) & set(second):
            if not _known_equal(first[key], second[key]):
                return False
        return True
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return len(first) == len(second) and all(
            _known_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return np.array_equal(first, second)
    return first == second


def _validate_merge_compatibility(datasets):
    reference = datasets[0]
    for index, dataset in enumerate(datasets[1:], start=1):
        if dataset.shape != reference.shape:
            raise ValueError(
                f"Agilent acquisitions {index} and 0 have incompatible shapes"
            )
        if dataset.dtype != reference.dtype:
            raise ValueError("Agilent acquisitions have incompatible dtypes")
        if dataset.dims != reference.dims:
            raise ValueError("Agilent acquisitions have incompatible dimensions")
        for key in (
            "spectral_width_hz",
            "dwell_time_s",
            "observation_frequency_mhz",
            "carrier_hz",
            "carrier_ppm",
            "carrier_source",
            "original_reference",
            "nucleus",
        ):
            if not _known_equal(
                dataset.meta.acquisition.get(key), reference.meta.acquisition.get(key)
            ):
                raise ValueError(f"Agilent acquisitions have incompatible acquisition {key}")
        for key in ("interop_axes", "interop_domains", "interop_encoding"):
            if not _known_equal(
                getattr(dataset.meta, key, None), getattr(reference.meta, key, None)
            ):
                raise ValueError(f"Agilent acquisitions have incompatible {key}")
        for dim in dataset.dims:
            left = dataset.coordset[dim]
            right = reference.coordset[dim]
            if str(left.units) != str(right.units) or not np.array_equal(
                left.data, right.data
            ):
                raise ValueError("Agilent acquisitions have incompatible coordinates")
        if dataset.meta.data_domain != reference.meta.data_domain:
            raise ValueError("Agilent acquisitions have incompatible domains")
        for key in (
            "normalization_profile",
            "normalization_state",
            "quadrature_mode",
            "branch_order",
            "axis_roles",
            "applied_operations",
        ):
            if getattr(dataset.meta, key, None) != getattr(reference.meta, key, None):
                raise ValueError(f"Agilent acquisitions have incompatible {key}")


def _stack_datasets(datasets):
    import spectrochempy as scp  # noqa: PLC0415

    _validate_merge_compatibility(datasets)
    result = scp.stack(*datasets)
    reference = datasets[0]
    result.meta.series_kind = "independent_acquisitions"
    result.meta.series_axis_role = "acquisition_index"
    result.meta.series_coordinate = "index"
    result.meta.normalization_state = reference.meta.normalization_state
    result.meta.normalization_profile = reference.meta.normalization_profile
    result.meta.quadrature_mode = reference.meta.quadrature_mode
    result.meta.branch_order = reference.meta.branch_order
    result.meta.acquisition_order = tuple(
        dataset.meta.provenance.get("source") for dataset in datasets
    )
    result.meta.individual_provenance = tuple(
        dict(dataset.meta.provenance) for dataset in datasets
    )
    result.meta.individual_acquisition = tuple(
        dict(dataset.meta.acquisition) for dataset in datasets
    )
    result.meta.individual_normalization_history = tuple(
        {
            "state": dataset.meta.normalization_state,
            "profile": dataset.meta.normalization_profile,
            "quadrature_mode": dataset.meta.quadrature_mode,
            "branch_order": dataset.meta.branch_order,
            "dimensions": dataset.meta.interop_dimensions,
            "physical_dimensions": dataset.meta.interop_physical_dimensions,
            "domains": dataset.meta.interop_domains,
            "operations": dataset.meta.applied_operations,
            "diagnostics": dataset.meta.diagnostics,
            "axes": dataset.meta.interop_axes,
            "encoding": dataset.meta.interop_encoding,
        }
        for dataset in datasets
    )
    result.history = f"Stacked {len(datasets)} normalized Agilent acquisitions"
    return result


def read_agilent_interop(
    *paths,
    merge=False,
    directory=None,
    recursive=False,
):
    """Read Agilent 1D acquisitions through nmrglue-ng interop.

    This experimental facade accepts acquisition directories or direct ``fid``
    paths. It does not replace :func:`read_agilent` and does not discover
    arbitrary child directories. ``merge=True`` stacks only scientifically
    compatible independent 1D acquisitions.
    """
    if recursive:
        raise NotImplementedError("recursive Agilent discovery is not supported")
    acquisition_paths = _acquisition_paths(paths, directory=directory)
    try:
        import nmrglue as ng  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ImportError(
            "read_agilent_interop requires the optional nmrglue-ng package"
        ) from exc

    from ..interop import from_interop

    datasets = []
    for acquisition in acquisition_paths:
        raw = ng.interop.read_agilent_fid(str(acquisition))
        normalized = ng.interop.normalize_agilent_fid(raw)
        datasets.append(from_interop(normalized))
    if len(datasets) == 1:
        return datasets[0]
    if not merge:
        import spectrochempy as scp  # noqa: PLC0415

        return scp.ScpObjectList(datasets)
    return _stack_datasets(datasets)
