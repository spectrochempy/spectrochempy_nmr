"""Plugin-owned scientific constraints for automatic import assembly."""

from .isotopes import _canonical_nucleus
from .nmr_metadata import extract_nmr_metadata


def _measurement_identity(dataset):
    """Describe known physical NMR axes, without interpreting descriptive titles."""
    meta = dataset.meta
    normalized = meta.get("interop_dimensions") is not None
    origin = meta.get("origin")
    if not normalized and (
        not isinstance(origin, str)
        or origin not in {"topspin", "agilent", "jeol", "tecmag", "simpson"}
    ):
        return None

    physical = tuple(meta.get("interop_physical_dimensions", ()))
    # A series can add observation dimensions without changing its acquisition.
    single = normalized and physical == ("F2",)
    identity = extract_nmr_metadata(meta, ndim=1 if single else dataset.ndim)
    series = normalized and meta.get("quadrature_mode") == "series-stored-2d"
    roles = tuple(meta.get("axis_roles", ()))
    reason = None
    if series:
        if roles != ("trace", "F2") or dataset.dims[-1] != "x":
            reason = (
                "NMR series must have the observation axis first for automatic merging"
            )
        axes = [(roles.index("F2"), "F2")]
    elif single or identity.ndim == 1:
        axes = [(0, "F2")]
        if dataset.ndim == 2 and dataset.dims[-1] != "x":
            reason = (
                "NMR series must have the observation axis first for automatic merging"
            )
        if dataset.ndim > 2:
            reason = "NMR series with multiple observation axes cannot be automatically merged"
    elif identity.ndim > 1:
        axes = list(enumerate(physical or ("F1", "F2")))
        reason = (
            "physical multidimensional NMR acquisitions require a separate series axis"
        )
    else:
        return None

    constraints = {}
    for index, axis in axes:
        prefix = f"nmr.{axis}"
        if identity.nuclei and index < len(identity.nuclei):
            try:
                nucleus = _canonical_nucleus(identity.nuclei[index])
            except (TypeError, ValueError):
                pass  # Unrecognized declarations remain unknown.
            else:
                constraints[f"{prefix}.nucleus"] = nucleus
        domain_known = True
        if not normalized or series:
            flags = meta.get("isfreq")
            domain_known = (
                flags is not None
                and index < len(flags)
                and isinstance(flags[index], bool)
            )
        if (
            domain_known
            and index < len(identity.domains)
            and identity.domains[index] in {"time", "frequency"}
        ):
            constraints[f"{prefix}.domain"] = identity.domains[index]
        if series or single or identity.ndim == 1:
            if getattr(dataset.dtype, "kind", None) in {"b", "i", "u", "f", "c"}:
                constraints[f"{prefix}.representation"] = (
                    "complex" if dataset.is_complex else "real"
                )
        elif identity.iscomplex and index < len(identity.iscomplex):
            state = identity.iscomplex[index]
            if isinstance(state, bool):
                constraints[f"{prefix}.representation"] = "complex" if state else "real"
        if reason and identity.encoding and index < len(identity.encoding):
            encoding = identity.encoding[index]
            if (
                encoding
                and encoding not in {"undefined", "unknown"}
                and not encoding.startswith("unknown(")
            ):
                constraints[f"{prefix}.encoding"] = encoding
    return {"constraints": constraints, "reason": reason}
