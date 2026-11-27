Compatibility and scope
=======================

The NMR 0.4 series requires SpectroChemPy 1.1.0 or later and
nmrglue-ng 1.3.0 or later, within their respective 1.x series.
Use SpectroChemPy 1.2.0 or later for workflows with paired 2D data or
undetermined indirect quadrature. SpectroChemPy 1.2.1 separates the FFT from
phase correction. In a step-by-step workflow, apply the phase with ``pk()``
after the FFT.

The development documentation also covers features listed under
:doc:`changelog` in the Unreleased section. Use the documentation version
that matches your installed plugin.

See :doc:`userguide/reading_data` for supported file formats and
:doc:`userguide/processing_2d` for supported acquisition and processing profiles.

Automatic import assembly
-------------------------

The development Core's ``importer.measurement_identity`` handler is required
for NMR merge checks. Older Core releases do not apply this new import policy,
even with the updated plugin. The plugin's other existing workflows remain
available with the versions listed above.

With both updates, automatic assembly (including ``merge=True``) checks known
nuclear species, time/frequency domains and decoded real/complex representation.
Nucleus spellings such as ``H1`` and ``1H`` identify the same isotope. Vendor
acquisition codes such as QSIM and DQD do not establish a contradiction once
they describe the same decoded 1D complex representation. For mergeable 1D
traces and observation series, representation follows the current data:
``spectrum.real`` is real even if acquisition metadata still says complex.
Missing or
unrecognized facts remain unknown; these checks do not certify complete
scientific comparability.

Compatible 1D spectra and observation-first series can be assembled when their
retained coordinates also agree. Physical multidimensional NMR acquisitions
stay separate: concatenating a physical axis would not create a series axis.
Slicing can retain the original acquisition axis declarations. When these no
longer describe the current dimensions, inputs stay separate instead of
aborting the import or guessing their physical identity.
A warning explains why inputs could not be combined. Use ``merge=False`` to
keep all inputs separate. Explicit ``stack`` and ``concatenate`` retain their
existing behavior and require the caller to choose scientifically valid axes.

The checks apply to supported reader metadata and normalized interop metadata,
also when restored from native ``.scp`` files. Descriptive titles and vendor
names alone do not establish a measurement identity. Native imports require
the NMR plugin to be loaded for these plugin-owned checks.
