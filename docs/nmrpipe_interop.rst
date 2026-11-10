NMRPipe interop qualification
==========================================

The nmrglue-ng provider can import complete, already transformed
NMRPipe 1D spectra into the generic Experiment path::

   import nmrglue as ng
   from spectrochempy_nmr import from_interop
   from spectrochempy_nmr.experiment import Experiment

   raw = ng.interop.read_nmrpipe_1d("spectrum.ft1")
   source = from_interop(ng.interop.normalize_nmrpipe_1d(raw))
   spectrum = Experiment(source).process()

This requires the provider release pinned in
``tests/qualification/requirements.txt`` (or a later compatible release).
The plugin installs nmrglue-ng as a dependency. This path is explicit and does not change
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

Import a normalized raw 2D QF FID
---------------------------------

This bridge requires a provider with NMRPipe raw QF normalization support,
which follows the nmrglue-ng 1.2.0 release. The NMRPipe provider qualifies
non-transposed QF files. For a known complete QF grid::

   stored = ng.interop.read("qf.fid")
   fid = from_interop(ng.interop.normalize(stored, confirm_time_grid=True))

The provider explicitly identifies QF and supplies the original time grids
in seconds. F1 is sampled as real, and F2 is complex. The resulting
``NDDataset`` preserves the file's physical axis order, samples and imaginary
sign. ``fid.meta.interop_complex_axis`` identifies F2. Both axes are temporal.
TPPI and mixed-domain inputs require a separate contract.

Set ``NMR_PIPE_QF_2D_REFERENCE`` to a complete, non-transposed NMRPipe QF FID
to compare the stored samples and both time grids and check SCP persistence::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py::test_nmrpipe_raw_qf_2d_reference

Process the normalized QF FID
------------------------------

Use ``Experiment(fid).process()`` for the FFT-only QF path::

   from spectrochempy_nmr.experiment import Experiment

   spectrum = Experiment(fid).process()

The transform runs along physical F2 first, then F1, in either physical axis
order supplied at import. Keep the imported dataset dimension order when
calling ``process()``. The direct step uses the centered, unnormalized
positive-sign DFT corresponding to NMRPipe ``FT -nodf``. The indirect
step uses the conjugate of the centered negative-sign DFT corresponding
to ``FT -neg``. The result keeps complex values and physical axis order,
with descending carrier-relative Hz axes. The supplied spectral widths
must agree with the uniform time grids. The original FID is preserved.

This entry point uses the input grid size. Apodization, zero-filling,
phase, magnitude and explicit referencing are outside this FFT-only path.
The returned axes describe relative frequencies and carry no implicit
chemical-shift calibration.

CI compares both complex FFT stages directly with checksum-pinned NMRPipe
checkpoints. The checkpoint data include zero-filling, so the test supplies
an already padded input with the original dwell times. With
``NMR_PIPE_QF_2D_REFERENCE`` configured as above and ``NMR_PIPE_EXECUTABLE``
pointing to NMRPipe, the external test also compares both stages and the
frequency grids from the resulting NMRPipe headers::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py::test_nmrpipe_raw_qf_experiment_reference

Import a States-TPPI acquisition
--------------------------------

For a known States-TPPI acquisition with adjacent E1/E2 traces and the
acquired alternation intact::

   stored = ng.interop.read("states-tppi.fid")
   normalized = ng.interop.normalize(stored, confirm_time_grid=True)
   fid = from_interop(normalized, profile="states-tppi")

The explicit profile asserts the acquisition convention. An NMRPipe States
storage header alone does not identify States-TPPI. The importer expects the
normalized ``rr, ri, ir, ii`` planes in F1/F2 order. It packs them into two
complex planes labelled E1 and E2 on ``(c, y, x)``, preserves the signs and
time grids, and records that alternation is still encoded. Import leaves reconstruction, alternation correction and FFT to
processing.

CI uses a synthetic NMRPipe file with distinct real and imaginary values in
both branches. The opt-in Bruker proton-exchange reference uses the original
SER and acquisition files with pinned checksums. Its 800 acquired words per
trace are packed into a temporary NMRPipe file, omitting storage padding.
Set ``NMR_BRUKER_STATES_TPPI_REFERENCE`` to the original acquisition
directory to run the real case::

   python -I -m pytest tests/qualification/test_nmrpipe_interop.py -k states_tppi_import

Process the imported States-TPPI FID
------------------------------------

Use ``Experiment(fid).process()`` on the imported E1/E2 dataset. The FFT-only
path follows the NMRPipe ``FT -nodf``, ``TP``, and ``FT -alt`` conventions.
It transforms F2, forms cross-pairs from the real and imaginary parts of
both branches, and applies the alternating sign once before the F1 FFT.
The result stores four real planes labelled rr, ri, ir and ii, with
descending Hz grids centred on zero at the central bin. The grid uses
the NMRPipe bin spacing. Absolute offsets and chemical referencing are
outside this processing path.

CI compares each complex checkpoint with the existing pinned NMRPipe
oracle. Its zero-filled grids are supplied as already padded inputs.
The real Bruker qualification additionally runs NMRPipe when
``NMR_PIPE_EXECUTABLE`` is set and compares both FFT stages, the cross-pairs,
and the centred frequency grids. Keep the imported dataset dimension
order when processing.

Process a confirmed pure States acquisition
-------------------------------------------

For a complete NMRPipe time-domain file whose branch order and signs
are known to follow the NMRPipe States convention, use::

   fid = scp.nmr.read(
       "states.fid",
       protocol="interop",
       normalize_options={"confirm_time_grid": True},
       conversion_profile="states",
   )
   spectrum = Experiment(fid).process()

This FFT-only path uses ``FT -nodf`` in both dimensions, with F2 before F1
and no alternating sign in F1. It returns the four real component planes
and centered Hz axes. Use ``Experiment.process()`` to enforce the transform
order. The pure States profile requires SpectroChemPy 1.2.0 or later.

The Agilent qualification uses the historical ``agilent_2d`` acquisition
from the nmrglue v0.5 corpus and explicit ``var2pipe`` States conversion.
That conversion conjugates the direct traces and flips the sign of the
second branch. Native ``read_agilent()`` returns the stored branches.
The :doc:`userguide/reading_data` guide explains how to select States
processing for a native acquisition.

Set ``NMR_AGILENT_STATES_REFERENCE`` to the acquisition directory and put
``nmrPipe`` and ``var2pipe`` on PATH to run the opt-in comparison::

   python -I -m pytest tests/qualification/test_agilent_states_fft.py

Report reproducible import/processing problems in the
`spectrochempy_nmr issue tracker
<https://github.com/spectrochempy/spectrochempy_nmr/issues>`_, including versions,
format, header information and the smallest shareable example. Array-only
round trips do not establish the correctness of reference metadata.
