NMRPipe 1D interop qualification (Unreleased)
=============================================

The optional external nmrglue-ng provider can import complete, already transformed
NMRPipe 1D spectra into the generic Experiment path::

   import nmrglue as ng
   from spectrochempy_nmr import from_interop
   from spectrochempy_nmr.experiment import Experiment

   raw = ng.interop.read_nmrpipe_1d("spectrum.ft1")
   source = from_interop(ng.interop.normalize_nmrpipe_1d(raw))
   spectrum = Experiment(source).process()

This requires the provider revision pinned in
``tests/qualification/requirements.txt`` (or a later compatible revision).
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
  FIDs, 2D/3D/4D, transposition and extracted grids are rejected by this adapter.
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

Report reproducible import/processing problems in the
`spectrochempy_nmr issue tracker
<https://github.com/spectrochempy/spectrochempy_nmr/issues>`_, including versions,
format, header information and the smallest shareable example. Array-only
round trips do not establish the correctness of reference metadata.
