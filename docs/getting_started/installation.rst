Install the NMR plugin
======================

NMR support is provided by the ``spectrochempy-nmr`` plugin.

Prepare Python and SpectroChemPy
--------------------------------

Follow the `SpectroChemPy installation guide
<https://www.spectrochempy.fr/stable/gettingstarted/install/index.html>`_
for your operating system. It explains how to prepare your Python environment
and install SpectroChemPy and JupyterLab.

Then return here to add the NMR plugin in the same environment.

Install the NMR plugin
----------------------

Open the terminal for the environment you prepared using the Core guide.
The following command goes in that terminal, not in a notebook.

Use the command matching the tool you used in the Core guide. These commands
install the versions used by the recipes.

**If you used uv:**

.. code-block:: console

   uv pip install "spectrochempy[nmr]==1.4.0" "spectrochempy-nmr==0.6.0"

**If you used pip:**

.. code-block:: console

   python -m pip install "spectrochempy[nmr]==1.4.0" "spectrochempy-nmr==0.6.0"

**If you used conda / mamba:** follow the `Core instructions for installing
the NMR plugin with conda / mamba
<https://www.spectrochempy.fr/stable/gettingstarted/install/install_adds.html#install-adds-plugins>`_,
then continue below.

Open a notebook
---------------

In the same terminal, run:

.. code-block:: console

   python -m jupyter lab

JupyterLab opens in your browser. Under **Notebook**, choose **Python 3**.
A notebook is a page of cells: paste code into a cell and press
**Shift+Enter** to run it. Run the cells from top to bottom. The figures appear
below the code. A line beginning with ``#`` is a comment, not a command.

Continue with :doc:`Your first spectrum <quickstart>`.
The first read may download the example data, so keep an internet connection
for that first run.

Coming back later
-----------------

Reopen the environment you prepared using the Core installation guide, then
run ``python -m jupyter lab``. Open your saved notebook to continue.

If installation or opening a notebook fails, see
:doc:`installation troubleshooting <troubleshooting>`.
