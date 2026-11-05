"""Common processing for normalized complex-channel NDDatasets."""

from __future__ import annotations

import numpy as np

from .quadrature import STATES_TPPI
from .quadrature import STATES_TPPI_F1COEF
from .quadrature import STATES_TPPI_F2_CONVENTION
from .quadrature import reconstruct_f1_interferograms


def _meta_value(meta, key, default=None):
    try:
        return meta.get(key, default)
    except AttributeError:
        return default


def _has_mask(dataset):
    return bool(np.any(np.asarray(dataset.mask, dtype=bool)))


def _validate_complex_channels(dataset, reconstruction):
    """Validate the normalized acquisition-channel contract."""
    if reconstruction != STATES_TPPI:
        raise ValueError(f"unsupported complex-channel reconstruction: {reconstruction}")
    if tuple(dataset.dims) != ("c", "y", "x"):
        raise ValueError("complex-channel input must have dimensions ('c', 'y', 'x')")
    data = np.asarray(dataset.data)
    if data.ndim != 3 or data.shape[0] != 2 or not np.iscomplexobj(data):
        raise ValueError("complex-channel input must have shape (2, nF1, nF2)")
    if _meta_value(dataset.meta, "component_axis") != "c":
        raise ValueError("complex-channel input must declare component_axis='c'")
    axis_roles = _meta_value(dataset.meta, "axis_roles")
    if (tuple(axis_roles) if axis_roles is not None else ()) != (
        "component",
        "F1",
        "F2",
    ):
        raise ValueError("complex-channel input must declare component, F1 and F2 axis roles")
    if _meta_value(dataset.meta, "stage") != "acquisition":
        raise ValueError("complex-channel input must have stage='acquisition'")
    labels = _meta_value(dataset.meta, "component_labels")
    if (tuple(labels) if labels is not None else ()) != ("branch_0", "branch_1"):
        raise ValueError("complex-channel input must label branches branch_0 and branch_1")
    channel_labels = dataset.coordset["c"].labels
    if channel_labels is None or tuple(str(label) for label in channel_labels) != (
        "branch_0",
        "branch_1",
    ):
        raise ValueError("complex-channel coordinate labels must identify both branches")
    if list(_meta_value(dataset.meta, "isfreq") or ()) != [False, False, False]:
        raise ValueError("complex-channel input must be in the time domain")
    if list(_meta_value(dataset.meta, "td") or ()) != list(data.shape):
        raise ValueError("complex-channel td metadata must match the dataset shape")
    if _meta_value(dataset.meta, "encoding") not in (None, ""):
        raise ValueError("complex-channel input must not carry an active FFT encoding")
    quadrature = _meta_value(dataset.meta, "quadrature_mode")
    if quadrature != STATES_TPPI:
        raise ValueError("complex-channel input must declare quadrature_mode='states-tppi'")
    if tuple(_meta_value(dataset.meta, "branch_order", ())) != ("E1", "E2"):
        raise ValueError("complex-channel input must declare normalized E1/E2 branches")
    if tuple(_meta_value(dataset.meta, "f1coef", ())) != STATES_TPPI_F1COEF:
        raise ValueError("complex-channel input has an unsupported States-TPPI f1coef")
    if _meta_value(dataset.meta, "tppi_alternation_state") != "encoded":
        raise ValueError("complex-channel input must declare encoded TPPI alternation")
    if _meta_value(dataset.meta, "normalization_state") != "normalized":
        raise ValueError("complex-channel input must declare normalized branches")
    if _meta_value(dataset.meta, "f2_convention") != STATES_TPPI_F2_CONVENTION:
        raise ValueError("complex-channel input has an unsupported F2 convention")
    if _has_mask(dataset):
        raise ValueError("complex-channel FFT processing requires unmasked input")
    return data


def _carried_metadata(dataset):
    """Return descriptive normalized-input metadata without interpreting it."""
    carried = {}
    for key in (
        "provenance",
        "acquisition",
        "normalization",
        "sequence",
        "vendor",
        "quadrature_mode",
        "branch_order",
        "f1coef",
        "tppi_alternation_state",
        "normalization_state",
        "f2_convention",
        "applied_operations",
    ):
        value = _meta_value(dataset.meta, key)
        if value is not None:
            carried[key] = value
    return carried


def _make_channel_dataset(data, *, labels, y_coord, x_coord, stage, meta=None):
    import spectrochempy as scp  # noqa: PLC0415

    result = scp.NDDataset(
        np.asarray(data),
        coordset=[
            scp.Coord.arange(2, dims="c", labels=list(labels)),
            y_coord.copy(),
            x_coord.copy(),
        ],
        dims=["c", "y", "x"],
    )
    result.meta.update(
        {
            "component_axis": "c",
            "axis_roles": ("component", "F1", "F2"),
            "stage": stage,
            "component_labels": tuple(labels),
            "reconstruction_profile": STATES_TPPI,
        }
    )
    if meta is not None:
        result.meta.update(meta)
    return result


def process_complex_channels(dataset, *, reconstruction):
    """Process normalized channels with ordinary core FFTs on ``x`` and ``y``."""
    _validate_complex_channels(dataset, reconstruction)
    import spectrochempy as scp  # noqa: PLC0415

    working = dataset.copy()
    working.meta.td = list(working.shape)
    carried = _carried_metadata(dataset)

    # The normalized input contract uses N * fftshift(ifft). Express that
    # convention through the core's ordinary complex FFT identity.
    f2_input = working.copy()
    f2_input.data = np.conjugate(f2_input.data)
    f2 = scp.fft(f2_input, dim="x")
    f2_data = np.conjugate(np.asarray(f2.data))
    interferograms = _make_channel_dataset(
        reconstruct_f1_interferograms(
            f2_data,
            profile=reconstruction,
            meta=dataset.meta,
        ),
        labels=("C_R", "C_I"),
        y_coord=f2.coordset["y"],
        x_coord=f2.coordset["x"],
        stage="f1_interferograms",
        meta={
            "processing_profile": reconstruction,
            "fft_f2": True,
            "reconstruction_f1": True,
            **carried,
        },
    )
    interferograms.meta.isfreq = [False, False, True]
    interferograms.meta.td = list(interferograms.shape)

    spectrum = scp.fft(interferograms, dim="y")
    spectrum = _make_channel_dataset(
        np.asarray(spectrum.data),
        labels=("S_R", "S_I"),
        y_coord=spectrum.coordset["y"],
        x_coord=spectrum.coordset["x"],
        stage="spectrum",
        meta={
            "processing_profile": reconstruction,
            "fft_f2": True,
            "reconstruction_f1": True,
            "fft_f1": True,
            "mathematical_component_order": ("RR", "RI", "IR", "II"),
            "data_domain": "frequency",
            "coordinates_status": "provisional_index",
            **carried,
            "fft_dimensions": ("x", "y"),
        },
    )
    spectrum.meta.td = list(spectrum.shape)
    spectrum.meta.isfreq = [False, True, True]
    return spectrum


def extract_complex_components(dataset):
    """Extract the validated NOESY mathematical components in contract order."""
    if tuple(dataset.dims) != ("c", "y", "x"):
        raise ValueError("NOESY channel dataset must have dimensions ('c', 'y', 'x')")
    if _meta_value(dataset.meta, "stage") != "spectrum":
        raise ValueError("NOESY components require a spectrum-stage dataset")
    if tuple(_meta_value(dataset.meta, "component_labels", ())) != ("S_R", "S_I"):
        raise ValueError("NOESY spectrum channels must be labelled S_R and S_I")
    if _meta_value(dataset.meta, "quadrature_mode") != STATES_TPPI:
        raise ValueError("spectrum must declare quadrature_mode='states-tppi'")

    import spectrochempy as scp  # noqa: PLC0415

    source = np.ma.array(dataset.data, mask=np.asarray(dataset.mask, dtype=bool))
    sr, si = source[0], source[1]
    arrays = {"RR": sr.real, "RI": si.real, "IR": sr.imag, "II": si.imag}
    masks = {
        "RR": np.ma.getmaskarray(sr),
        "IR": np.ma.getmaskarray(sr),
        "RI": np.ma.getmaskarray(si),
        "II": np.ma.getmaskarray(si),
    }
    coords = [dataset.coordset["y"].copy(), dataset.coordset["x"].copy()]
    result = {}
    for name, values in arrays.items():
        item = scp.NDDataset(
            np.ma.array(
                np.asarray(values).copy(),
                mask=np.asarray(masks[name], dtype=bool).copy(),
            ),
            coordset=coords,
            dims=["y", "x"],
        )
        item.meta.update(
            {
                "stage": "spectrum_component",
                "component": name,
                "component_source": "S_R" if name in ("RR", "IR") else "S_I",
                "profile": _meta_value(dataset.meta, "processing_profile"),
            }
        )
        result[name] = item
    return result
