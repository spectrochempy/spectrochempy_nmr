SpectroChemPy NMR documentation
================================

.. warning::

   **Experimental plugin** for :external+core:doc:`SpectroChemPy <index>`.
   The public API is still evolving. Function signatures, options, and
   behavior may change between releases, including in validated workflows.
   Check the :doc:`changelog` before upgrading.

SpectroChemPy NMR is developed and released independently, alongside
SpectroChemPy. This gives NMR-specific features such as acquisition
quadrature, chemical-shift referencing, and the processing of complex signals
their own development cycle.

Working with SpectroChemPy
--------------------------

Import a free induction decay (FID) or a processed spectrum with
``scp.nmr.read()``. The plugin returns a SpectroChemPy
:external+core:py:class:`~spectrochempy.NDDataset` with coordinates, units, and
acquisition metadata. You can slice and plot the dataset and apply the
core's general-purpose processing and analysis tools directly. For
NMR-specific processing, ``scp.nmr.Experiment(dataset)`` uses the acquisition
metadata to select and validate the workflow.

See :external+core:ref:`the NMR plugin overview in SpectroChemPy <nmr-plugin>`
for installation and API integration.

Formats and reference workflows
-------------------------------

The plugin reads Bruker TopSpin, Agilent/Varian, JEOL, and SIMPSON data. It
also reads supported profiles of the NMRPipe, nmrML, and NMR JCAMP-DX exchange
formats. See :doc:`userguide/reading_data` for supported inputs and options.

Processing has been qualified primarily against
`Bruker TopSpin <https://www.bruker.com/protected/en/services/software-downloads/nmr.html>`_
and `NMRPipe <https://www.ibbr.umd.edu/nmrpipe/install>`_. The plugin supports
validated 1D workflows and processing of qualified 2D acquisitions. The
:doc:`2D processing guide <userguide/processing_2d>` explains when to process
the FID directly and when to replay a recorded TopSpin recipe.

Contents
--------

.. toctree::
   :maxdepth: 2

   getting_started/index
   userguide/index
   reference/index
   examples_overview
   about
