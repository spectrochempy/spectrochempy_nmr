SpectroChemPy NMR
=================

Open, process, measure and simulate NMR spectra with SpectroChemPy.
All the first recipes use one import: ``import spectrochempy as scp``.

Install the NMR plugin
----------------------

Start with :doc:`Install the NMR plugin <getting_started/installation>`.
The page links to the Core installation guide for Python and SpectroChemPy,
then shows how to add the NMR plugin and open a notebook.
Then follow :doc:`Your first spectrum <getting_started/quickstart>`.
Each recipe gives complete code, an example figure and the settings you can change.

What would you like to do?
--------------------------

* **Display a spectrum:** :doc:`open and zoom an example <getting_started/quickstart>`,
  then :doc:`read your own files <userguide/reading_data>`.
* **Process an acquisition:** turn a recorded signal into a
  :doc:`1D spectrum <userguide/processing_1d>` or a
  :doc:`2D map <userguide/processing_2d>`. For solid-state experiments,
  continue to :doc:`MQMAS, STMAS and Multiplex SPAM <userguide/quadrupolar>`.
* **Measure spectra:** :doc:`correct a baseline <userguide/baseline>`,
  :doc:`find peaks, integrate and fit lines <userguide/analysis>`.
* **Analyze a series:** follow
  :doc:`signal recovery <userguide/relaxation_t1>` or
  :doc:`cross-polarization buildup <userguide/cp_buildup>`.
* **Simulate and fit spectra:** start with :doc:`PULSAR <userguide/pulsar>`;
  explore :doc:`disordered sites <userguide/czjzek>` or
  :doc:`joint MQMAS/MAS fitting <userguide/mqmas_inversion_synthetic>`.
* **Save and exchange spectra:** save your work in the
  :doc:`first recipe <getting_started/quickstart>`, or
  :doc:`export spectra for other software <userguide/exchanging_spectra>`.

The plugin is experimental. Use the documentation version matching your
installation; see :doc:`about` for compatibility and release notes.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting_started/index
   userguide/index
   examples_overview
   reference/index
   about
