SpectroChemPy NMR
=================

SpectroChemPy NMR provides tools for reading, processing, analyzing and
simulating nuclear magnetic resonance data within
:external+core:doc:`SpectroChemPy <index>`. Work with acquired signals,
processed spectra or calculated responses using the same datasets,
calibrated coordinates, physical units and plotting tools.

Read acquisitions and exchange spectra
--------------------------------------

Import raw acquisitions and processed spectra from supported Bruker TopSpin,
Agilent/Varian, JEOL and SIMPSON files, or use supported profiles of NMRPipe,
nmrML and NMR JCAMP-DX. Save datasets in SpectroChemPy's native format to
retain their metadata and processing history, or export supported spectra
through JCAMP-DX and CSDF, including exchanges with Dmfit.

Explore the :doc:`reading guide <userguide/reading_data>`, the
:doc:`spectrum exchange guide <userguide/exchanging_spectra>` and the
:doc:`import/export examples <examples/gallery/importer/index>`.

Process one- and two-dimensional data
-------------------------------------

Build a workflow around ``scp.nmr.read()`` and ``scp.nmr.Experiment``:
prepare FIDs, apply apodization and zero filling, Fourier transform the
acquired dimensions, correct phases and reference the frequency axes.
Readers arrange the complex and hypercomplex components so that common
processing operates independently of the source instrument.

Supported workflows include conventional 1D and 2D processing, magnitude
spectra and symmetrization, as well as MQMAS, STMAS and Multiplex SPAM
reconstruction. Recorded TopSpin processing settings can also be replayed
for supported acquisitions.

Start with the :doc:`1D guide <userguide/processing_1d>`, the
:doc:`2D guide <userguide/processing_2d>`, the
:doc:`quadrupolar NMR guides <userguide/quadrupolar>` or the
:doc:`processing examples <examples/gallery/nmr/index>`.

Analyze spectra and experimental series
---------------------------------------

Combine NMR-specific operations with SpectroChemPy's analysis tools to
correct baselines, detect peaks, integrate 1D and 2D regions, and fit
overlapping signals. Follow spectral intensities across experimental
series to analyze relaxation and cross-polarization dynamics.

See the :doc:`analysis guide <userguide/analysis>`, the
:doc:`baseline guide <userguide/baseline>` and the
:doc:`analysis examples <examples/gallery/analysis/index>`.

Simulate spectra and fit physical models
----------------------------------------

Use PULSAR to calculate solid-state powder spectra under static conditions,
magic-angle spinning or variable-angle spinning. Model quadrupolar
interactions and chemical-shift anisotropy, explore finite RF pulses and
supported coupled-spin sequences, and reuse pulse programs while varying
experimental parameters.

Fit calculated line shapes to observations, describe quadrupolar disorder
with Czjzek distributions, or estimate a shared population distribution
from MQMAS and MAS spectra.

Explore the :doc:`PULSAR guide <userguide/pulsar>`, the
:doc:`pulse-program guide <userguide/pulse_programs>`, the
:doc:`Czjzek guide <userguide/czjzek>`, the
:doc:`MQMAS/MAS inversion guide <userguide/mqmas_inversion_synthetic>` and
the :doc:`simulation examples <examples/gallery/simulation/index>`.

Choose your starting point
--------------------------

Follow :doc:`Getting started <getting_started/index>` for installation and
a first workflow. Use the :doc:`guides <userguide/index>` to understand
methods and processing choices, the :doc:`gallery <examples_overview>` for
short runnable workflows, and the :doc:`API reference <reference/index>`
for individual functions and options.

.. warning::

   **Experimental plugin.** SpectroChemPy NMR is developed and released
   independently, alongside SpectroChemPy. Support depends on the input
   format, acquisition mode and physical model; each guide describes the
   supported scope and relevant qualifications.

   The public API is still evolving. Function signatures, options, and
   behavior may change between releases, including in supported workflows.
   Check the :doc:`changelog` before upgrading.

Contents
--------

.. toctree::
   :maxdepth: 2

   getting_started/index
   userguide/index
   reference/index
   examples_overview
   about
