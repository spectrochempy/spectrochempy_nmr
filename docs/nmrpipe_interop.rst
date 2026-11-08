NMRPipe interop qualification (Unreleased)
==========================================

The optional external nmrglue-ng provider can import complete, already transformed
NMRPipe 1D spectra into the generic Experiment path::

   import nmrglue as ng
   from spectrochempy_nmr import from_interop
   from spectrochempy_nmr.experiment import Experiment

   raw = ng.interop.read_nmrpipe_1d("spectrum.ft1")
   source = from_interop(ng.interop.normalize_nmrpipe_1d(raw))
   spectrum = Experiment(source).process()

This requires the provider release pinned in
``tests/qualification/requirements.txt`` (or a later compatible release).
The provider is an optional integration dependency; it is not bundled or
installed by the plugin. This path is explicit and does not change
``scp.nmr.read`` dispatch. No core change is required for this 1D integration.

Supported contract
------------------

Real spectra remain real; complex spectra retain their real and imaginary
components. Experiment preserves the supplied calibrated frequency axis and
performs no second FFT or acquisition filtering. Explicit phase correction is
available only when an imaginary component exists. Changing vendor provenance
does not change processing. SCP persistence preserves samples, axes and metadata.

NMRPipe ORIG is the last-point frequency in Hz. The adapter uses
``f[i] = ORIG + (N - 1 - i) * SW / N`` and OBS in MHz for ppm conversion.
CAR is retained as source metadata, not substituted for ORIG. Calibration
already recorded by the processing software is preserved; it is not proof of
a correct chemical reference or an independently determined standard.

Qualification and limits
------------------------

* Synthetic real/complex files exercise both byte orders in the provider.
* Two existing real NMRPipe reference FIDs, of Agilent and Bruker origin, were
  transformed with external NMRPipe 13.0 Rev 2026.072.12.03. Complex outputs and
  real outputs from zero-phase PS with imaginary deletion were imported.
* Raw output words and header fields are checked separately from the interop
  reader, then samples, Hz/ppm axes, phase rules and SCP round trips are checked.
  This shares format assumptions and does not certify the earlier conversion
  of the historical reference FIDs. No new native vendor conversion is claimed.
* The installed-provider qualification is reproducible with the test below.
  External references remain local; no spectra are redistributed in this PR.
* Storage components and header fields have float32 precision. Original
  precision, acquisition filter history and original vendor metadata cannot be
  reconstructed. Existing scaling and phase choices are preserved.
* Only a single complete frequency-domain F2 spectrum is qualified. Raw NMRPipe
  FIDs, transposition and extracted grids are outside that 1D profile.
  Invalid flags, nonfinite values, missing calibration and mismatched payload
  sizes are rejected rather than guessed. Broader ``ng.pipe.read`` capability
  does not imply normalized qualification.
* NMRPipe is an intermediate format, not a promise that its converters can read
  every native processed format. Native processed reading remains scoped to
  Bruker. nmrML is a separate planned qualification.

Run the synthetic CI cases::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py -k synthetic

For the local external cases, set ``NMR_PIPE_AGILENT_SOURCE`` and
``NMR_PIPE_BRUKER_SOURCE`` to the corresponding reference FIDs, and
``NMR_PIPE_EXECUTABLE`` to the installed nmrPipe executable. Then run the whole
test file. Each source is hashed before/after; tools write only under pytest's
temporary directory. Missing external configuration produces explicit skips,
which must not be reported as validated coverage.

Import a processed real 2D plane
--------------------------------

Pass an explicitly normalized, entirely frequency-domain real plane to
``from_interop()``::

   import nmrglue as ng
   from spectrochempy_nmr import from_interop

   raw = ng.interop.read_nmrpipe_2d("cosy.ft2")
   spectrum = from_interop(ng.interop.normalize(raw))

The converter accepts one real F1/F2 plane and maps the supplied physical
axis order onto the dataset's ``(y, x)`` dimensions. It preserves the
sample dtype, orientation, calibrated axes and per-axis references.
``spectrum.meta.axis_roles`` identifies the physical dimensions in dataset
order, including for a transposed input. An axis already in ppm retains
that unit. A supplied MHz observation frequency allows conversion from
Hz to ppm using the same convention as the 1D path.

The qualification compares the 512 × 2048 real COSY plane with the raw
NMRPipe payload and checks every axis coordinate from the header. CI downloads
the checksum-pinned `provider fixture
<https://github.com/spectrochempy/nmrglue-ng/blob/14f5ee8b06592060a987cbd12f79c6e9a36c8a20/tests/fixtures/fileio/data/nmrpipe_2d_cosy_example.ft2>`_
and checks an SCP round trip. To run the test locally, set
``NMR_PIPE_COSY_REFERENCE`` to the downloaded file and run::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py::test_real_nmrpipe_cosy_plane

Import a processed 2D plane with one complex axis
-------------------------------------------------

Use the NMRPipe storage reader for a fully frequency-domain plane with
exactly one complex dimension::

   raw = ng.interop.read_nmrpipe_2d_storage("complex.ft2")
   spectrum = from_interop(ng.interop.normalize(raw))

The adapter combines ``rr`` with ``ri`` for a complex column axis, or
``rr`` with ``ir`` for a complex row axis. It keeps the stored imaginary
sign and the physical order of the axes. Float32 and float64 component
planes become complex64 and complex128, respectively. The physical complex
axis is recorded in ``spectrum.meta.interop_complex_axis``. The input
encoding and branch order remain available in the interop metadata.

CI generates small frequency-domain NMRPipe files covering F1 and F2
quadrature in both physical axis orders. For local qualification, set
``NMR_PIPE_COSY_REFERENCE`` as above and ``NMR_PIPE_EXECUTABLE`` to the
NMRPipe executable. Then run::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py::test_nmrpipe_singly_complex_cosy_reference

This test creates a temporary complex reference with NMRPipe
``-fn HT -auto`` on the pinned real COSY. It was qualified with NMRPipe
13.0 Rev 2026.072.12.03. The generated imaginary plane tests binary import
and SCP persistence. It is a Hilbert transform of the real spectrum, so it
does not establish the original acquired quadrature.

Import a processed 2D plane with four components
-------------------------------------------------

Use the same storage reader and normalization for a fully frequency-domain
NMRPipe plane with both dimensions complex. The dataset has dimensions
``(c, y, x)`` and preserves the four real planes and their floating-point
dtype. The ``c`` coordinate labels are ``rr``, ``ri``, ``ir`` and ``ii``.
For example, ``spectrum.c`` identifies the branches and ``spectrum.data[1]``
contains ``ri``. The first letter describes the row axis and the second the
column axis in the supplied physical order. Consult ``spectrum.meta.axis_roles``
to map those axes to F1 and F2. The converter keeps that order and the stored
signs. Hypercomplex is optional for this import.

CI compares both physical orders against the binary words of synthetic
NMRPipe files. To reproduce the external reference, download the pinned
`2D FID fixture
<https://github.com/spectrochempy/nmrglue-ng/blob/14f5ee8b06592060a987cbd12f79c6e9a36c8a20/tests/fixtures/fileio/data/nmrpipe_2d_time.fid>`_,
set ``NMR_PIPE_2D_FID_REFERENCE`` to its path and set
``NMR_PIPE_EXECUTABLE`` to the NMRPipe executable::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py::test_nmrpipe_four_component_reference

The test runs ``FT -auto``, ``TP``, then ``FT -auto`` and qualifies the output
before and after a final ``TP``. This sequence was qualified with NMRPipe
13.0 Rev 2026.072.12.03. The test checks all samples, both calibrated axes,
and their persistence with the component labels in an SCP round trip. The
core SCP reader can widen float32 to float64 while preserving sample values.

These paths qualify conversion to ``NDDataset``. Raw and mixed-domain 2D
inputs and 2D processing through ``Experiment`` remain outside the bridge.

Report reproducible import/processing problems in the
`spectrochempy_nmr issue tracker
<https://github.com/spectrochempy/spectrochempy_nmr/issues>`_, including versions,
format, header information and the smallest shareable example. Array-only
round trips do not establish the correctness of reference metadata.
