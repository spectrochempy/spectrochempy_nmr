# Gallery fixtures

The executable RTD gallery fetches a small, immutable fixture set rather than
the complete SpectroChemPy data corpus. The manifest is
`docs/gallery-fixtures.toml`; every file is verified by size and SHA-256.

The TopSpin source is `spectrochempy_data` revision
`08bb9b0cbff8f4363c48b6bff7ccb743f8140e0a`. The provenance register identifies
the TopSpin 1D, 2D, CP and relaxation measurements as LCS data released under CC BY
4.0, with attribution to Christian Fernandez, LCS. Pulse programs and generated
`__index__` files are intentionally not fetched because they are excluded from
that grant or are generated metadata.

The JEOL epicatechin 1D acquisition and the processed NMRPipe
gossypol COSY plane come from [nmrXiv project P33](https://doi.org/10.57992/nmrxiv.p33)
under CC0 1.0. They are retrieved from pinned revisions of nmrglue-ng.
The NMRPipe file is a conversion of the processed Bruker RR plane from
[nmrXiv sample S213](https://nmrxiv.org/sample/S213).

The Bruker PST-5 aluminium-27 MQMAS acquisition (experiment 51) comes from
[spectrochempy_data revision 86f8c4b](https://github.com/spectrochempy/spectrochempy_data/tree/86f8c4b9bf6dcbd512a7d8d0c476c1b2a7e1d18d/testdata/nmrdata/bruker/tests/nmr/MQMAS_27AL_PST5/).
The dataset is licensed under CC BY 4.0 and attributed to Christian Fernandez,
LCS. The fixture contains the SER and the parameters used to read it.

The raw epicatechin COSY JDF and its processed JCAMP-DX export are fetched
from [spectrochempy_data revision 34448ef](https://github.com/spectrochempy/spectrochempy_data/tree/34448ef343f17a2ee57d2bc06661ac90e2d1ff48/testdata/nmrdata/jeol),
also under CC0 1.0. They support the common QF guide and the two COSY gallery
examples. Their hashes and source provenance are recorded in that dataset.

The selected files total about 22.8 MiB. Run the same fetch as the RTD
pre-build job locally with:

```console
python docs/fetch_gallery_data.py
```

No fixture is copied into the NMR repository or published as a new data
distribution.

## JEOL COSY magnitude qualification

The optional tests in `tests/qualification/test_jeol_cosy_magnitude.py`
use JDF and JCAMP-DX COSY pairs from nmrXiv project P33 for
[luteolin](https://doi.org/10.57992/nmrxiv.p33.s214),
[epicatechin](https://doi.org/10.57992/nmrxiv.p33.s192), and
[rutin](https://doi.org/10.57992/nmrxiv.p33.s188). The test module pins
the file hashes. Keep the original filenames in study subdirectories
and point `NMR_JEOL_COSY_DIR` at their common parent. The rejection test
also uses the raw epicatechin HSQC from the same study.

```console
NMR_JEOL_COSY_DIR=/path/to/CENAPT python -m pytest -q tests/qualification/test_jeol_cosy_magnitude.py
```

The interrupted States-TPPI guide also fetches the PST-5 HMQC
SER and acquisition/processing parameters from `topspin_2d/1` at the
same pinned revision. The stored pulse program is excluded.
