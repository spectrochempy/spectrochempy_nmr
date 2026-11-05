.. SpectroChemPy NMR documentation master file.

SpectroChemPy NMR documentation
================================

This is the SpectroChemPy NMR documentation for the independent
``spectrochempy-nmr`` and ``spectrochempy-hypercomplex`` plugins. The Python
distributions remain independent; this site groups their scientific
documentation under the SpectroChemPy identity.

The NMR plugin currently provides validated **1D** reading and processing. It
also reads 2D data and can represent phase-sensitive quadrature as
hypercomplex data, but general 2D processing is not yet a supported workflow.

Start with one of the following paths:

* :doc:`getting_started/installation` and :doc:`getting_started/quickstart`;
* :doc:`userguide/reading_data` and :doc:`userguide/processing_1d`;
* :doc:`concepts/hypercomplex` for quaternion data;
* :doc:`examples/gallery/nmr/index` for executable NMR examples;
* :doc:`reference/nmr_api` and :doc:`reference/hypercomplex_api` for APIs.

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

   examples/gallery/importer/index
   examples/gallery/apodization/index
   examples/gallery/nmr/index
   compatibility
   changelog

* `Main SpectroChemPy documentation <https://www.spectrochempy.fr/>`_
* `NMR plugin source <https://github.com/spectrochempy/spectrochempy_nmr>`_
* `Hypercomplex plugin source <https://github.com/spectrochempy/spectrochempy_hypercomplex>`_
