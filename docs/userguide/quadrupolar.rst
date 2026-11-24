Process solid-state 2D spectra
==============================

Choose the recipe matching your experiment. Each uses the public example
signal and the same sequence of operations: read, correct the common phase,
straighten the pattern and display the map.

* :doc:`mqmas`: a conventional aluminium-27 3QMAS spectrum.
* :doc:`stmas`: a rubidium-87 satellite-transition spectrum.
* :doc:`multiplex_spam`: select a 3Q signal from a recorded phase cycle.

.. toctree::
   :maxdepth: 1
   :hidden:

   mqmas
   stmas
   multiplex_spam
   czjzek
   mqmas_inversion_synthetic

Simulate and fit
----------------

Use :doc:`pulsar` to simulate a powder spectrum, :doc:`czjzek` for disordered
sites, or :doc:`mqmas_inversion_synthetic` for a joint MQMAS/MAS fit.
Acquisition conventions and reference comparisons are in the
:doc:`technical appendices </appendices/index>`.
