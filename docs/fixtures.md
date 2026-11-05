# Gallery fixtures

The executable RTD gallery fetches a small, immutable fixture set rather than
the complete SpectroChemPy data corpus. The manifest is
`docs/gallery-fixtures.toml`; every file is verified by size and SHA-256.

The source is `spectrochempy_data` revision
`08bb9b0cbff8f4363c48b6bff7ccb743f8140e0a`. The provenance register identifies
the TopSpin 1D and relaxation measurements as LCS data released under CC BY
4.0, with attribution to Christian Fernandez, LCS. Pulse programs and generated
`__index__` files are intentionally not fetched because they are excluded from
that grant or are generated metadata.

The selected files total 417,443 bytes (approximately 408 KiB). The fetch is performed only by
the RTD pre-build job and can be run locally with::

   python docs/fetch_gallery_data.py

No fixture is copied into the NMR repository or published as a new data
distribution.
