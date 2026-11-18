"""Compatibility import for the former TPPI implementation module."""


def process_agilent_tppi(dataset, reference):
    from ..nmr_metadata import extract_nmr_metadata
    from .tppi import process_tppi

    metadata = extract_nmr_metadata(dataset.meta, ndim=dataset.ndim)
    return process_tppi(dataset, metadata.spectral_width_hz, reference)
