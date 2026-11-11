Experiment and processing
=========================

Use ``scp.nmr.Experiment(dataset)`` to inspect the acquisition and run a
processing workflow. See :doc:`/userguide/processing_1d` and
:doc:`/userguide/processing_2d` for examples.
For explicit shearing of an already transformed MQMAS map, see
:doc:`/userguide/mqmas` and ``Experiment.shear_mqmas()``. The same guide
covers ``Experiment.replay_xfshear()`` for the rectangular Bruker States recipe.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated

   Experiment

Experimental Agilent NOESY helpers
----------------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   process_agilent_noesy_states_tppi
   extract_noesy_components
