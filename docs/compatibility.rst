Compatibility and scope
=======================

The NMR 0.3 series requires SpectroChemPy 1.1.0 or later and
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
