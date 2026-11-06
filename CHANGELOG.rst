Changelog
=========

This changelog covers the independently versioned ``spectrochempy-nmr``
distribution. Entries under Unreleased describe development after the latest
release, not capabilities available in the published 0.2.0 package.

Unreleased
----------

Changed
~~~~~~~

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
