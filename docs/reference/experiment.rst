Experiment and processing
=========================

Use ``scp.nmr.Experiment(dataset)`` to inspect the acquisition and run a
processing workflow. See :doc:`/userguide/processing_1d` and
:doc:`/userguide/processing_2d` for examples.
For the Bruker States MQMAS workflow, see :doc:`/userguide/mqmas`.
Use ``Experiment.prepare_xfshear()`` to inspect F2 and
``Experiment.replay_xfshear()`` to produce the final spectrum.

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
