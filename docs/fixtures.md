# Gallery fixtures

The executable RTD gallery fetches a small, immutable fixture set rather than
the complete SpectroChemPy data corpus. The manifest is
`docs/gallery-fixtures.toml`; every file is verified by size and SHA-256.

The TopSpin source is `spectrochempy_data` revision
`08bb9b0cbff8f4363c48b6bff7ccb743f8140e0a`. The provenance register identifies
the TopSpin 1D and relaxation measurements as LCS data released under CC BY
4.0, with attribution to Christian Fernandez, LCS. Pulse programs and generated
`__index__` files are intentionally not fetched because they are excluded from
that grant or are generated metadata.

The JEOL epicatechin 1D and COSY acquisitions and the processed NMRPipe
gossypol COSY plane come from [nmrXiv project P33](https://doi.org/10.57992/nmrxiv.p33)
under CC0 1.0. They are retrieved from pinned revisions of nmrglue-ng.
The NMRPipe file is a conversion of the processed Bruker RR plane from
[nmrXiv sample S213](https://nmrxiv.org/sample/S213).

The selected files total about 10.5 MiB. Run the same fetch as the RTD
pre-build job locally with:

```console
python docs/fetch_gallery_data.py
```

No fixture is copied into the NMR repository or published as a new data
distribution.
