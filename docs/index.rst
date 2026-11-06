.. SpectroChemPy NMR documentation master file.

SpectroChemPy NMR documentation
================================

This is the development documentation for the independent
``spectrochempy-nmr`` plugin. Changes marked **Unreleased** are not included
in the published 0.2.0 package.

The plugin provides validated **1D** reading and processing. Its experimental
2D paths store associated branches as two ordinary complex channels. General
2D processing remains under characterization.

Start with one of the following paths:

* :doc:`getting_started/installation` and :doc:`getting_started/quickstart`;
* :doc:`userguide/reading_data` and :doc:`userguide/processing_1d`;
* :doc:`concepts/hypercomplex` for native complex channels;
* :doc:`examples/gallery/nmr/index` for executable NMR examples;
* :doc:`reference/nmr_api` for APIs.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   userguide/reading_data
   userguide/processing_1d

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/quadrature
   concepts/hypercomplex

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/nmr_api
   reference/hypercomplex_api

.. toctree::
   :maxdepth: 2
   :caption: Examples and compatibility

   agilent_qualification
   processed_1d_interop
   nmrpipe_interop
   nmrml_interop
   bruker_normalized_qualification
   agilent_frequency_reference

   examples/gallery/importer/index
   examples/gallery/apodization/index
   examples/gallery/nmr/index
   compatibility
   changelog

* `Main SpectroChemPy documentation <https://www.spectrochempy.fr/>`_
* `NMR plugin source <https://github.com/spectrochempy/spectrochempy_nmr>`_
