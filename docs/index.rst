.. SpectroChemPy NMR documentation master file.

SpectroChemPy NMR documentation
================================

.. warning::

   **Experimental plugin.** The public API is still evolving. Function
   signatures, options, and behavior may change between releases, including
   in workflows that have already been validated. Check the :doc:`changelog`
   before upgrading.

SpectroChemPy NMR is an independently released plugin. It reads vendor and
exchange formats and provides processing for 1D and qualified 2D acquisitions.
Use the :doc:`2D processing guide <userguide/processing_2d>` to choose between
processing the samples directly and replaying a recorded TopSpin recipe.

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
   userguide/processing_2d

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
   bruker_pdata_2d_interop
   agilent_frequency_reference

   examples/gallery/importer/index
   examples/gallery/apodization/index
   examples/gallery/nmr/index
   compatibility
   changelog

* `Main SpectroChemPy documentation <https://www.spectrochempy.fr/>`_
* `NMR plugin source <https://github.com/spectrochempy/spectrochempy_nmr>`_
