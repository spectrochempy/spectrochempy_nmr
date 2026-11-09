Changelog
=========

This changelog covers the independently versioned ``spectrochempy-nmr``
distribution. Entries under Unreleased describe development after the latest
release, not capabilities available in the published 0.2.0 package.


Unreleased
----------

Added
~~~~~

* Prepare normalized complex 1D FIDs explicitly with
  ``Experiment.prepare_fid()`` before FFT, using provider-declared corrections.

Changed
~~~~~~~

* Return the normalized interop dataset for qualified Bruker 2D pdata,
  preserving both complex channels, receiver gain and both axis options.

* Use the normalized interop dataset for qualified Bruker echo/anti-echo SER
  data while preserving the public two-stage FFT convention.

* Use the normalized interop dataset for qualified Bruker States-TPPI SER
  data with calibrated public FFT presentation.


* Use the normalized interop dataset for complete Bruker QF SER data when
  the provider supplies the public FFT presentation contract. Preserve the
  existing two-stage FFT display and correction options.


* Return the normalized Bruker 1D FID dataset from ``read_topspin()`` when
  the provider supports the FID contract. Use canonical complex samples and
  preserve the default public FFT presentation. Keep the previous reader
  path for older providers and unsupported acquisitions.

* Preserve the normalized dataset in the public Bruker 1D pdata reader
  when the provider supplies the requested axis and receiver-gain correction.
  Distinguish the processing reference from the acquisition frequency.

* Delegate compatible Bruker 2D SER decoding to nmrglue-ng and preserve
  the TopSpin reader's recovery path for incomplete acquisitions.

* Delegate Bruker 1D FID decoding to nmrglue-ng. The public TopSpin reader
  keeps its digital-filter and DC-offset options, gain normalization and axes.

* Delegate qualified Bruker 1D ``pdata`` to nmrglue-ng while preserving
  TopSpin reader data, gain normalization, metadata, and axis options.

* Depend on nmrglue-ng and delegate qualified Bruker 2D ``pdata`` decoding
  to it while preserving the public TopSpin reader's channels and axes.

* Offer phase- and time-domain Bruker F2 group-delay corrections for imported
  2D echo/anti-echo FIDs. Resolve the delay from acquisition metadata when
  available and qualify both methods on the official HSQC.

* Process imported 2D echo/anti-echo FIDs with explicit Bruker Rance–Kay
  reconstruction in ``Experiment.process()``. Qualify the reconstruction
  and both FFTs against NMRPipe on the official HSQC.

* Import normalized raw 2D echo/anti-echo acquisitions through
  ``from_interop()``, preserving both complex branches and time axes.

* Process imported States-TPPI acquisitions with ``Experiment.process()``.
  Qualify both FFTs and the cross-pairs against NMRPipe and return four
  labelled real spectral planes with centred Hz axes.

* Import explicitly declared States-TPPI acquisitions as two labelled
  complex channels, preserving the acquired alternation and time axes.

* Process explicitly normalized raw 2D QF acquisitions with
  ``Experiment.process()``. Qualify the complex F2 and F1 FFTs against
  NMRPipe and return carrier-relative Hz axes.

* Import explicitly normalized raw 2D QF NMRPipe acquisitions through
  ``from_interop()``, preserving both time axes and the complex F2 samples.

* Qualify complete Bruker processed 2D spectra through ``from_interop()``.
  Preserve the four original plane names, calibrated axes and selected
  storage scale.

* Import fully processed four-component 2D spectra through ``from_interop()``
  as labelled real planes, preserving the input dtype at import. Branch order
  and axis calibration survive SCP persistence without requiring Hypercomplex.

* Import fully processed 2D spectra with one complex axis through
  ``from_interop()``. Preserve the real and imaginary values, physical axis
  order and complex-axis identity, including on SCP round trips.

* Accept normalized real 2D frequency spectra through ``from_interop()``.
  Preserve physical axis order, sample dtype and calibration. Qualify the
  conversion and SCP persistence with a real NMRPipe COSY spectrum.

* Qualify explicit nmrML 1D raw/processed interop through the generic Experiment
  consumer, including dtype/count/sign/axis conventions and SCP persistence.
  Add no runtime provider dependency or automatic reader dispatch. (#34)

* Qualify explicit optional NMRPipe processed 1D interop with Experiment on real
  Agilent/Bruker reference inputs transformed by NMRPipe; preserve real/complex
  samples, last-point references and SCP persistence. Document unsupported cases.
  (#33)

* Accept normalized real or complex processed 1D spectra, preserving their
  supplied frequency axes and processing references without another FFT or
  implicit recalibration. Reject phase requests when imaginary data are absent.
  Qualify complete Bruker pdata spectra on two real acquisitions.


* Align commit-message and pull-request title prefixes with the SpectroChemPy
  convention and add CI validation for pull-request titles.

* Qualify Bruker QSIM/DQD 1D via the same normalized Experiment workflow as
  Agilent, with explicit filter/canonical-conjugation provenance, source
  independence and SCP persistence. Keep vendor interpretation in nmrglue-ng;
  add no vendor-specific processing branch to the plugin.

* Integrate normalized complex 1D acquisitions with Experiment, independent of
  origin, profile labels and raw vendor metadata. Keep default FFT axes in Hz
  and select ppm referencing explicitly from the producer's numeric descriptor.
  Validate the processing chain and persistence on both real Agilent acquisitions.
  The experimental ``reference_agilent_1d`` helper now delegates to the generic
  ``reference_interop_1d`` helper and requires the normalized frequency descriptor.

* Reject unsupported channel FFT/IFFT requests before any in-place preparation,
  preserving samples, axes, coordinates, masks and metadata on refusal. Native
  channels and guarded indirect profiles require a core exposing
  ``FFT_PREVALIDATION_API >= 1``; older cores retain the ordinary 1D workflow.

* Remove quaternion storage, the Hypercomplex extra and the quaternion phasing
  handler. Experimental 2D branches now use ordinary complex datasets with a
  labelled component axis. Bruker pdata preserves all four components.
* Consolidate the experimental NOESY channel function into
  ``process_agilent_noesy_states_tppi``; remove its legacy quaternion and
  inplace variants and the redundant ``_channels`` entry point.
* Keep Agilent and JEOL storage decoding distinct from quadrature normalization;
  reject unvalidated indirect transforms instead of guessing their interpretation.
* Require F2 before F1 for Bruker paired acquisitions and reject Fourier
  transforms of the component axis. Inverse channel reconstruction is unsupported.
* Preserve real QF FID transforms and reject partially transformed input to
  the experimental NOESY adapter. Resolve its physical axes by name and verify
  the branch coordinate order before decoding native channels.

Added
~~~~~

* Experimental explicit Agilent 1D frequency referencing after interop FFT,
  with a required choice of reference model, preserved disagreement evidence,
  calibrated Hz/ppm coordinates and SCP persistence. Qualify the two real
  acquisitions against an optional external NMRPipe oracle, keeping samples
  unchanged and chemical-reference ambiguity explicit.

* Real-provider Agilent 1D qualification with pinned nmrglue-ng, checksummed
  external acquisitions, independent binary sample checks and wheel-based CI.
  The pinned provider includes the observed-nucleus correction from
  nmrglue-ng PR #153, covered by a passing regression for acquisition metadata
  and axis labels. Public reader dispatch remains unchanged.
* Qualify sourced Agilent scan counts from nmrglue-ng PR #154 through the
  experimental facade, independent-acquisition stacking and SCP persistence.
  Counts of 160 and 512 remain metadata; samples are not scan-normalized.

* Declarative TopSpin 1D processing plans with explicit validation and
  unsupported-parameter diagnostics. These plans do not execute vendor recipes.
* Experimental conversion of descriptive Bruker processed 1D results, and a
  separate converter for normalized complex 1D acquisitions, to ``NDDataset``.
* Experimental Agilent 1D interop reading with optional stacking of compatible
  independent acquisitions and preservation of their individual provenance.
* Experimental 2D States-TPPI processing using two ordinary complex channels
  and core FFTs, with an explicitly scoped Agilent NOESY adapter. Physical
  quadrature validation and calibrated 2D axes remain separate requirements.
* Standalone Sphinx documentation and Read the Docs build configuration.

Fixed
~~~~~

* Keep Agilent interop time/frequency metadata coherent through FFT and restore
  the time coordinate after inverse FFT of an explicitly referenced spectrum.

* Preserve accessible normalized 1D time coordinates instead of collapsing small
  dwell intervals through default coordinate rounding.

* Reject incompatible Agilent interop references across all acquisitions when
  stacking, even when the first acquisition has unknown reference metadata.
* Reject masked interop samples and invalid time coordinates instead of losing
  masks or accepting inconsistent axes. Copy array-valued metadata so edits to
  converted datasets cannot modify the normalized source.
* Pair Agilent 2D States phase branches along successive indirect traces,
  preserving the full direct complex axis.

Development
~~~~~~~~~~~

* Validate installed plugin wheels on released cores 1.1.0 and 1.1.3 without
  Hypercomplex, numpy-quaternion or nmrglue-ng, including numerical reference,
  component extraction, persistence and normalized interop tests.
* Add standalone pre-commit hygiene, spelling, Ruff lint and formatting hooks
  without version, changelog or release-generation scripts.

0.2.0 — 2026-10-26
------------------

* First release from the standalone plugin repository after publication cutover.
* Preserve the existing core compatibility bounds (``>=1.1.0,<2``).
* Conda and Zenodo publication are not included in this release.

See the `0.2.0 release
<https://github.com/spectrochempy/spectrochempy_nmr/releases/tag/spectrochempy-nmr-v0.2.0>`_
and `changes since 0.2.0
<https://github.com/spectrochempy/spectrochempy_nmr/compare/spectrochempy-nmr-v0.2.0...master>`_.

Release-note ownership
----------------------

Update Unreleased for user-visible plugin changes. At release time, assign those
entries to the plugin version being published. The core changelog should record
its own integration or compatibility changes and link to the relevant plugin
release rather than duplicate this changelog.

Future automatic aggregation into core release notes should consume an explicit
published plugin tag or version and preserve its source link. It must not copy
the mutable Unreleased section or imply that a plugin change ships in the core.
No cross-repository aggregation or release automation is enabled here.
