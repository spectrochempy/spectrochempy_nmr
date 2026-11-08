"""Apply a provider-declared preparation recipe to a canonical complex FID."""

from collections.abc import Mapping

import numpy as np

from spectrochempy.core.units import ur


def prepare_fid(dataset, *, remove_dc_offset, digital_filter, normalize_scan_gain):
    if not isinstance(remove_dc_offset, bool) or not isinstance(
        normalize_scan_gain, bool
    ):
        raise ValueError("DC and scan/gain options must be booleans")
    if digital_filter not in (None, "time"):
        raise ValueError("digital_filter must be None or 'time'")
    acquisition = dict(dataset.meta.get("acquisition", {}))
    recipe = acquisition.get("fid_preparation")
    if (
        not isinstance(recipe, Mapping)
        or recipe.get("schema") != "fid-preparation-v1"
        or recipe.get("complex_convention") != "centered-negative-exponent-dft"
    ):
        raise ValueError("FID preparation requires a supported provider descriptor")
    samples = np.asarray(dataset.data)
    axis = dataset.coord(0)
    width = acquisition.get("spectral_width_hz")
    if (
        samples.size < 2
        or recipe.get("sample_count") != samples.size
        or not np.iscomplexobj(samples)
        or not np.isfinite(samples).all()
        or np.any(dataset.mask)
        or axis.units is None
        or not axis.units.is_compatible_with(ur.s)
        or not isinstance(width, (int, float))
        or isinstance(width, bool)
        or not np.isfinite(width)
        or width <= 0
        or not np.allclose(
            axis.to("s").data, np.arange(samples.size) / width, rtol=1e-9, atol=1e-14
        )
    ):
        raise ValueError("FID preparation requires the original calibrated time grid")

    stages = ("dc_offset", "digital_filter", "scan_gain")
    requested = (remove_dc_offset, digital_filter == "time", normalize_scan_gain)
    applied = tuple(
        acquisition.get(stage, {}).get("applied") is True for stage in stages
    )
    filter_spec = recipe.get("digital_filter", {})
    no_filter_needed = filter_spec.get("state") == "not-required"
    filter_already_skipped = (
        no_filter_needed
        and acquisition.get("digital_filter", {}).get("requested") is True
    )
    pending = (
        requested[0] and not applied[0],
        requested[1] and not applied[1] and not filter_already_skipped,
        requested[2] and not applied[2],
    )
    for index, wanted in enumerate(pending):
        if (
            wanted
            and any(applied[index + 1 :])
            and not (index == 1 and no_filter_needed)
        ):
            raise ValueError(
                "FID preparation order is DC, digital filter, then scan/gain"
            )
    if not any(pending):
        return dataset.copy()

    work = samples.copy()
    completed = {}
    scale = None
    if pending[0]:
        work -= work.mean()
        completed["dc_offset"] = {"applied": True, "input_points": int(work.size)}
    if pending[1]:
        state = filter_spec.get("state")
        if (
            state == "available"
            and filter_spec.get("method") == "integer-shift-fold-trim"
        ):
            shift = filter_spec.get("left_shift_points")
            fold = filter_spec.get("fold_points")
            discard = filter_spec.get("discard_points")
            if (
                any(
                    not isinstance(count, int) or isinstance(count, bool)
                    for count in (shift, fold, discard)
                )
                or not 0 <= shift < work.size
                or not 0 <= fold <= discard < work.size - 1
            ):
                raise ValueError("invalid FID filter geometry")
            work = np.roll(work, -shift)
            if fold:
                work[:fold] += work[: -fold - 1 : -1].copy()
            work = work[: work.size - discard]
            completed["digital_filter"] = {
                **filter_spec,
                "requested": True,
                "applied": True,
                "removed_points": discard,
            }
        elif state == "not-required":
            completed["digital_filter"] = {
                "requested": True,
                "applied": False,
                "reason": "no-decimation",
            }
        else:
            raise ValueError("FID digital-filter correction is unavailable")
    if pending[2]:
        gain = recipe.get("scan_gain", {})
        divisor = gain.get("divisor")
        if (
            gain.get("state") != "available"
            or not isinstance(divisor, (int, float))
            or isinstance(divisor, bool)
            or not np.isfinite(divisor)
            or divisor <= 0
        ):
            raise ValueError("FID scan/gain normalization is unavailable")
        scale = 1.0 / divisor
        if not np.isfinite(scale):
            raise ValueError("invalid FID scan/gain scale")
        work /= divisor
        completed["scan_gain"] = {**gain, "applied": True}
        acquisition["normalization"] = "scan-gain"
    if not np.isfinite(work).all():
        raise ValueError("FID preparation produced nonfinite samples")

    result = dataset[: work.size].copy()
    result.data = work
    result.meta.readonly = False
    acquisition.update(completed)
    acquisition["returned_points"] = int(work.size)
    acquisition["fid_preparation"] = {**recipe, "sample_count": int(work.size)}
    result.meta.acquisition = acquisition
    result.meta.td = [int(work.size)]
    processing = dict(result.meta.get("nmr_processing", {}))
    processing["fid_preparation"] = {
        **processing.get("fid_preparation", {}),
        **completed,
    }
    result.meta.nmr_processing = processing
    result.meta.applied_operations = (
        *result.meta.get("applied_operations", ()),
        {
            "name": "prepare-fid",
            "stage": "acquisition",
            "input_dimensions": ("F2",),
            "output_dimensions": ("F2",),
            "axis": "F2",
            "scale": scale,
            "conjugation": "none",
            "sign": "none",
            "permutation": None,
            "evidence": recipe["schema"],
            **completed,
        },
    )
    result.meta.readonly = dataset.meta.readonly
    return result
