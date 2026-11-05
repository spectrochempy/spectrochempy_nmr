# spectrochempy-nmr

NMR plugin for SpectroChemPy.

**Extraction candidate:** publication still belongs to the SpectroChemPy
monorepo. See [EXTRACTION.md](EXTRACTION.md) for the pinned source, validation
scope and remaining release/documentation handover.

This package is the home for NMR-specific readers and tools that are useful in
SpectroChemPy but should not live in the core package. It currently provides a
validated public 1D workflow for reading data and processing raw 1D FIDs
through `scp.nmr.read(...)` and `scp.nmr.Experiment(...)`. The readers also
load raw 2D time-domain data and vendor-processed 2D spectra and can represent
them as hypercomplex arrays; this reading/representation scope is separate from
processing, which remains limited to validated 1D experiments.

Future NMR readers or processing helpers can be added here without creating a
new plugin package for each vendor format. Multi-dimensional NMR processing
remains under active characterization and is not currently part of the public
supported processing workflow. A pseudo-2D series (a list of ordinary 1D
spectra sharing a secondary coordinate, such as a relaxation series) must be
distinguished from a genuine 2D experiment with two Fourier dimensions: the
former follows the validated 1D processing path per spectrum.

## Installation

This plugin requires SpectroChemPy 1.1.0 or later (and remains
restricted to SpectroChemPy versions below 2). Users who stay on
SpectroChemPy 1.0.0 should keep an earlier compatible version of this plugin.


```bash
pip install "spectrochempy>=1.1.0,<2" "spectrochempy-nmr==0.2.0"
```

For an editable development install from this repository, see
[Development](#development).

## Usage

```python
import spectrochempy as scp

dataset = scp.nmr.read("path/to/1/fid")
dataset = scp.nmr.read("path/to/1/pdata/1/1r")
dataset = scp.nmr.read("path/to/experiment", expno=1, procno=1)

experiment = scp.nmr.Experiment(dataset)
spectrum = experiment.process(apodization="em", lb=2.0, size=32768)
```

This explicit processing example does not imply that vendor processing
parameters are automatically imported and replayed from metadata. That contract
remains under active characterization.

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
prototype performs two ordinary complex FFTs and assembles quaternion storage
in the explicitly declared **mathematical** `(RR, RI, IR, II)` order. These
labels are not a claim that the physical Agilent quadrature convention has been
validated. It does not perform a quaternion FFT, apodization, phasing, linear
prediction, or universal vendor interpretation. Other Agilent sequences and
encodings are rejected. This is experimental and is not part of the validated
public processing workflow. The input must also declare `ntraces=1`; returned
coordinates are provisional index coordinates and are not a calibrated physical
frequency claim. Raw acquisition values are retained under `raw_acquisition`.

The first vendor-neutral channel representation is also available:

```python
channels = scp.nmr.process_agilent_noesy_states_tppi_channels(raw_dataset)
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

From the plugin directory, install the plugin against an installed compatible
core (or install the desired core wheel first):

```bash
python -m pip install -e ".[hypercomplex]" pytest
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
(the notice for the adapted/vendorized parsers). Repository extraction does not
change those parsers or introduce a dependency on nmrglue-ng.
