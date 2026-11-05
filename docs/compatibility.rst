Compatibility and scope
=======================

This is **Unreleased** documentation after NMR 0.2.0. The plugin requires
``spectrochempy>=1.1.0,<2``. CI tests its installed wheel against released
cores 1.1.0 and 1.1.4 and the pinned core revision after quaternion retirement.
No core source change or additional representation plugin is required.

The public processing workflow remains validated for 1D. The experimental 2D
storage and processing contracts are described in :doc:`concepts/hypercomplex`.
Storage decoding alone does not validate a physical quadrature convention.

``docs/compatibility.toml``, ``docs/build-manifest.txt`` and
``docs/phase2b-validation.md`` are historical phase-2b build records. They do
not describe the current development dependencies. The active RTD requirements
are in ``docs/requirements-rtd.txt``. No RTD parent/subproject relationship or
DNS setting is changed by this migration.
