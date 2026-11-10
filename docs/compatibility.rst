Compatibility and scope
=======================

NMR 0.3.0 requires ``spectrochempy>=1.1.0,<2`` and
``nmrglue-ng>=1.2.4,<2``. Guarded native-channel workflows require the core
FFT prevalidation hook (``FFT_PREVALIDATION_API >= 1``), available in
SpectroChemPy 1.2.0. The plugin stores supported paired 2D data as two
ordinary complex channels. See :doc:`userguide/processing_2d` for the
processing scope and :doc:`concepts/hypercomplex` for the storage contract.

``docs/compatibility.toml``, ``docs/build-manifest.txt`` and
``docs/phase2b-validation.md`` are historical phase-2b build records. They do
not describe the current development dependencies. The active RTD requirements
are in ``docs/requirements-rtd.txt``. No RTD parent/subproject relationship or
DNS setting is changed by this migration.
