Installation
============

Install SpectroChemPy 1.3.1 with the NMR extra to use the released plugin.
Workflows marked as development features, including PST-5 processing
with ``Experiment.rr()``, require the plugin source checkout described below.

.. code-block:: console

   python -m pip install "spectrochempy[nmr]==1.3.1"

The NMR plugin installs ``nmrglue-ng``, which provides the same Python
package (``import nmrglue``) as the original ``nmrglue`` distribution.
Use only one of these distributions per environment. Installing both can
overwrite shared files even if pip reports a successful installation.
The plugin reports the conflict when you access its public API.
If you use the original ``nmrglue`` for other projects, install the NMR
plugin in a separate environment.

For a pip-managed environment already containing the original ``nmrglue``,
remove both distributions before reinstalling the NMR plugin. Removing both
lets pip restore any shared files removed during uninstallation.

.. code-block:: console

   python -m pip uninstall nmrglue nmrglue-ng
   python -m pip install "spectrochempy[nmr]==1.3.1"

If you use SpectroChemPy 1.2.0, install ``spectrochempy-nmr`` directly.
The ``nmr`` extra in that version selects the plugin's 0.2 series.
See :doc:`/compatibility` for version requirements. For workflows marked as
available in development, install the plugin from a source checkout as
described in the :doc:`developer guide </devguide/processing_conventions>`.

The plugin stores associated 2D branches as
:doc:`native complex channels </concepts/hypercomplex>`.
