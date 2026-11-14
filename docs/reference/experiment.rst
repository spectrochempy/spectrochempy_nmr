Experiment and processing
=========================

Use ``scp.nmr.Experiment(dataset)`` to inspect the acquisition and run a
processing workflow. See :doc:`/userguide/processing_1d` and
:doc:`/userguide/processing_2d` for examples.
For the Bruker States MQMAS workflow, see :doc:`/userguide/mqmas`.
Use ``Experiment.xf2()`` to transform F2, ``Experiment.xf2p()`` to apply
the common phase, and ``Experiment.xfshear()`` to produce the final spectrum.
For polynomial baseline correction along F1 or F2 and numbered trace
extraction, see :ref:`topspin-slice-and-baseline`.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated
   :template: autosummary/experiment.rst

   Experiment

Experimental Agilent NOESY helpers
----------------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   process_agilent_noesy_states_tppi
   extract_noesy_components
