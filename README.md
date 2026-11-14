# spectrochempy-nmr

[![Tests](https://github.com/spectrochempy/spectrochempy_nmr/actions/workflows/test.yml/badge.svg)](https://github.com/spectrochempy/spectrochempy_nmr/actions/workflows/test.yml)
[![Conda](https://img.shields.io/conda/v/spectrocat/spectrochempy-nmr)](https://anaconda.org/spectrocat/spectrochempy-nmr)
[![PyPI version](https://img.shields.io/pypi/v/spectrochempy-nmr)](https://pypi.org/project/spectrochempy-nmr/)
[![Documentation](https://img.shields.io/badge/docs-online-blue.svg)](https://www.spectrochempy.fr/projects/nmr/latest/)
[![License](https://img.shields.io/badge/license-CeCILL--B-blue.svg)](LICENSE)

NMR plugin for SpectroChemPy.

> [!WARNING]
> **Experimental plugin.** The public API is still evolving. Function
> signatures, options, and behavior may change between releases, including in
> workflows that have already been validated. Check the changelog before
> upgrading. [Changelog](CHANGELOG.rst)

The plugin provides NMR-specific readers for vendor and exchange formats,
validated 1D workflows, and processing for qualified 2D acquisitions.
The 0.4 series stores paired 2D data as two ordinary complex channels.
See the [NMR documentation](https://www.spectrochempy.fr/projects/nmr/latest/)
for installation, readers, processing and supported acquisition profiles.

## Installation

Install SpectroChemPy 1.3.1 with the NMR extra.

```bash
python -m pip install "spectrochempy[nmr]==1.3.1"
```

The plugin installs `nmrglue-ng`. If you use the original `nmrglue`, install
the plugin in a separate environment because both distributions provide the
same Python package. See the
[installation guide](https://www.spectrochempy.fr/projects/nmr/latest/getting_started/installation.html)
to repair an environment containing both.

For an editable development install from this repository, see
[Development](#development).

## Usage

```python
import spectrochempy as scp

fid = scp.nmr.read("path/to/1/fid")
experiment = scp.nmr.Experiment(fid)
spectrum = experiment.process(apodization="em", lb=2.0, size=32768)

processed = scp.nmr.read("path/to/1/pdata/1/1r")
```

The [Quickstart](https://www.spectrochempy.fr/projects/nmr/latest/getting_started/quickstart.html)
shows how to apply phase correction explicitly.

Use `Experiment.process(replay="topspin")` to replay supported TopSpin
recipes explicitly. The [2D processing guide](https://www.spectrochempy.fr/projects/nmr/latest/userguide/processing_2d.html)
describes the qualified XFB workflow.

The current explicit apodization contract covers the public modes already
exposed by `Experiment.process()`: `em(lb=...)`, `gm(lb=..., gb=...)`, and
`sp(ssb=..., pow=...)`.

The result of `Experiment.process()` also records the SpectroChemPy-owned
processing trace in `result.meta.nmr_processing["scp_processing"]`.
`requested` contains only the arguments explicitly provided by the user,
whereas `applied` contains only the operations that were actually executed and
the values they really consumed. This trace is attached to the result only:
the source dataset is not mutated, the vendor `procs` profile remains purely
descriptive, and `phase="metadata"` does not replay TopSpin `PHC0`/`PHC1`.

The NMR ppm/frequency unit context is also provided by this plugin:

```python
from spectrochempy_nmr.units import set_nmr_context

set_nmr_context(104.3 * scp.ur.MHz)
```

The compatibility aliases `scp.nmr.read_topspin(...)` and
`scp.read_topspin(...)` are kept for historical usage. TopSpin is a reader, so
it is not exposed as `dataset.read_topspin(...)` or
`dataset.nmr.read_topspin(...)`.

### Experimental Agilent NOESY States-TPPI

An opt-in prototype is available for explicitly annotated raw Agilent/Varian
NOESY States data:

```python
result = scp.nmr.process_agilent_noesy_states_tppi(
    raw_dataset,
    transport="none",  # or explicitly "var2pipe"
)
```

The input must declare `seqfil='NOESY'`, `phase=(1, 2)`, `ni`, `np` and
`arraydim`, with a matching complex `(arraydim, np // 2)` matrix. The
prototype performs two ordinary complex FFTs and returns two complex channels
in the explicitly declared **mathematical** `(RR, RI, IR, II)` order. These
labels are not a claim that the physical Agilent quadrature convention has been
validated. It does not perform apodization, phasing, linear
prediction, or universal vendor interpretation. Other Agilent sequences and
encodings are rejected. This is experimental and is not part of the validated
public processing workflow. The input must also declare `ntraces=1`; returned
coordinates are provisional index coordinates and are not a calibrated physical
frequency claim. Raw acquisition values are retained under `raw_acquisition`.

The returned native channels have an explicit component view:

```python
channels = scp.nmr.process_agilent_noesy_states_tppi(raw_dataset)
components = scp.nmr.extract_noesy_components(channels)
```

`channels` is a standard complex `NDDataset` with dimensions `("c", "y",
"x")` and channel labels `S_R`, `S_I`. The `c` axis is mathematical and
non-physical. The component mapping is explicit: `RR=Re(S_R)`, `RI=Re(S_I)`,
`IR=Im(S_R)`, and `II=Im(S_I)`. This path uses ordinary core FFTs and does not
use quaternion storage. Time-domain input must be unmasked; masks may be
applied to the final spectrum and are retained by component extraction and
native NDDataset persistence. The Agilent metadata and transport operation are
an adapter boundary; the common channel engine receives only the normalized
3D dataset and an explicit reconstruction profile.

States-TPPI is the indirect-dimension acquisition mode; NOESY is only the
first experimentally validated sequence using this adapter. The common engine
does not select its calculation from the sequence or vendor identity. The
normalized contract records `quadrature_mode="states-tppi"`, explicit `E1/E2`
branch order, the validated `f1coef`, TPPI alternation state, normalization
state and F2 transform convention. Other sequences are not thereby claimed to
be experimentally validated.

The implementation is split into three layers:

- `processing/complex_channels.py` owns the normalized NDDataset contract,
  the two core FFT calls and channel/component views;
- `processing/quadrature.py` owns the single validated `states-tppi`
  reconstruction equation;
- `adapters/agilent_noesy.py` owns the provisional Agilent metadata validation,
  transport conversion and branch demultiplexing.

The Agilent adapter is transitional. The common engine does not import readers,
vendor adapters or `spectrochempy_hypercomplex`; future `nmrglue-ng`
normalization will feed it through the same normalized-channel boundary.

### Experimental neutral interop conversion

The plugin also exposes a generic converter for a validated normalized
acquisition supplied by `nmrglue.interop`:

```python
dataset = scp.nmr.from_interop(normalized_acquisition)
```

The first supported profile is a storage-normalized complex 1D FID. The
converter accepts the neutral contract rather than a vendor or sequence name,
preserves axes, provenance, acquisition metadata and applied operations, and
does not perform FFT, scaling, conjugation, phasing or merging. A descriptive
`ReadResult` is rejected until a profile-specific normalization step has
established its guarantees. Existing `read_agilent`, `read_topspin` and merge
behavior are unchanged in this experimental bridge.

### Experimental Agilent interop facade

The complete normalized 1D path can be exercised without replacing the public
reader:

```python
dataset = scp.nmr.read_agilent_interop("experiment/", merge=False)
stacked = scp.nmr.read_agilent_interop(
    "experiment_01/", "experiment_02/", merge=True
)
```

The facade accepts acquisition directories or direct `fid` paths, delegates
decoding and normalization to `nmrglue-ng`, returns a `ScpObjectList` for
multiple acquisitions with `merge=False`, and uses the existing core `stack()`
only after compatibility checks for `merge=True`. It does not replace
`read_agilent()`, discover arbitrary recursive directories, or interpret the
stacked axis as spectral F1. Its role is `acquisition_index` unless a
validated experimental parameter is supplied by a future extension.
Pseudo-2D acquisition is not implemented here. Stacked output retains the
shared normalization contract and each acquisition's individual provenance,
operation history, diagnostics, axis references and encoding.

## Development

See [CHANGELOG.rst](CHANGELOG.rst) for plugin release notes, including the
unreleased development changes. The plugin maintains its own changelog;
SpectroChemPy release notes can link to a specific published plugin version.

Install and run the standalone development hooks before submitting changes:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Hook revisions match the core's existing hygiene, spelling and Ruff hooks.
The plugin configuration does not run core requirement, lazy-stub, version or
release-note generators. Adapted third-party parsers and binary test fixtures
are excluded from these mechanical checks. Ruff starts with its default lint
rules rather than imposing the core's broader lint policy on the extracted
package.

The experimental complex-channel processing path stores two complex channels
in an ordinary `NDDataset` with shape `(2, nF1, nF2)` and dimensions
`("c", "y", "x")`. It uses the existing core complex FFTs and does not require
`spectrochempy-hypercomplex` or `numpy-quaternion`. Native channel paths require a core exposing `FFT_PREVALIDATION_API >= 1`;
they refuse to start on published cores 1.1.0/1.1.4, which retain ordinary 1D
and interop support. The read-only prevalidation hook rejects unsupported
FFT/IFFT requests before in-place changes. All readers and processing paths
use ordinary numeric arrays. The optional
`hypercomplex` extra and the legacy quaternion phasing handler have been removed.
The former `_channels` NOESY function is consolidated into
`process_agilent_noesy_states_tppi`.

CI covers ordinary numeric/1D and interop support on released cores 1.1.0
and 1.1.4 without a representation plugin. Native reader profiles, in-place
refusal regressions, channel processing and persistence also run against the
exact companion core commit with FFT prevalidation.
Synthetic numerical agreement does not validate
the physical quadrature of an uncharacterized acquisition or provide calibrated
2D frequency axes.

From the plugin directory, install the plugin against an installed compatible
core (or install the desired core wheel first):

```bash
python -m pip install -e . pytest
python -m pytest tests/ -v
```

Tests and examples can run outside the monorepo. Real-data tests still use
`scp.preferences.datadir` and its sibling `testdata-extra`; unavailable fixtures
may cause skips. A scientific baseline must record which datasets were available
and must not treat skipped cases as validated.

Build source and wheel distributions from this directory:

```bash
python -m pip install build
python -m build
```

Both distributions include `LICENSE` (CeCILL-B) and `NMRGLUE_LICENSE.rst`
(the notice for code adapted from nmrglue). The plugin uses nmrglue-ng
for vendor and exchange format decoding.
