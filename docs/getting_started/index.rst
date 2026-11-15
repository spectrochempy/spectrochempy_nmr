Getting started
===============

SpectroChemPy NMR adds the tools needed to turn an NMR acquisition into a
spectrum you can analyze with SpectroChemPy. For example, processing a 2D
free induction decay (FID) requires knowing how the spectrometer encoded
quadrature in the indirect dimension and how to reference each frequency
axis. The plugin uses acquisition metadata to handle these NMR-specific
steps, while the core library provides the shared dataset and general-purpose
methods.

* **Read data.** Import raw acquisitions and processed spectra from
  spectrometer formats such as Bruker TopSpin, Agilent/Varian and JEOL, or
  from supported exchange formats such as NMRPipe and nmrML. See
  :doc:`/userguide/reading_data`.
* **Process raw acquisitions.** Use ``scp.nmr.Experiment`` to apply steps
  such as apodization, zero filling, Fourier transformation and phase
  correction, with the acquisition conventions for each dimension. See
  :doc:`/userguide/processing_1d` and :doc:`/userguide/processing_2d`.
* **Work with spectra.** Whether you import a processed spectrum or
  transform an FID, the result is an ``NDDataset``. Use core SpectroChemPy
  methods to correct the baseline, find peaks, integrate spectral regions
  and fit overlapping lines. See :doc:`/userguide/analysis`.
* **Simulate and fit solid-state spectra.** PULSAR models 1D powder spectra,
  including line shapes from quadrupolar nuclei and chemical shift
  anisotropy. See :doc:`/userguide/pulsar` and
  :doc:`/userguide/quadrupolar`.

Support varies by format and acquisition sequence. If you encounter an
unsupported dataset or have a workflow you would like to see covered, tell
us about it in the
`NMR plugin issue tracker <https://github.com/spectrochempy/spectrochempy_nmr/issues>`_.
Include the format, the experiment and the processing or analysis step you
need. These details help guide future development.

.. toctree::
   :maxdepth: 2

   installation
   quickstart

See also
--------

* :external+core:doc:`SpectroChemPy documentation <index>`
