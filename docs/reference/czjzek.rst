Czjzek distributions
~~~~~~~~~~~~~~~~~~~~

Access the distribution module as ``scp.nmr.czjzek``.
See :doc:`../userguide/czjzek` for the tensor noise scale and examples.

.. automodule:: spectrochempy_nmr.czjzek
   :members: CzjzekDistribution, CzjzekSamples, simulate_czjzek

Finite-RF libraries and fitting
--------------------------------

See :doc:`../userguide/czjzek_fit` for the complete two-stage workflow.
The classes and function are available directly through ``scp.nmr``.

.. autoclass:: spectrochempy_nmr.CzjzekRFLibrary
   :members: build, save, load, simulate

.. autofunction:: spectrochempy_nmr.fit_czjzek

.. autoclass:: spectrochempy_nmr.CzjzekFitResult
   :members:

.. autoclass:: spectrochempy_nmr.CzjzekFitStage
   :members:
