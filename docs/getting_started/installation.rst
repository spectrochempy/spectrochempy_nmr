Install and open a notebook
===========================

You need Python 3.11 or later. If Python is not installed yet, follow the
`Python installation guide <https://www.python.org/about/gettingstarted/>`_
for your operating system, then return here.

Install the tools
-----------------

Open a terminal (Command Prompt on Windows). These commands go in the terminal,
not in a notebook. Create a folder for your NMR work, open a terminal in that
folder, then create a separate Python environment:

.. code-block:: console

   python -m venv .venv

Activate it using the command for your system.

**Windows (Command Prompt):**

.. code-block:: console

   .venv\Scripts\activate.bat

**macOS or Linux:**

.. code-block:: console

   source .venv/bin/activate

On Windows, open Command Prompt for the activation command above. On macOS
or Linux, use ``python3`` in place of ``python`` if that is how Python is
installed on your computer.

Now install SpectroChemPy, the NMR plugin and the notebook application:

.. code-block:: console

   python -m pip install "spectrochempy[nmr]==1.4.0" "spectrochempy-nmr==0.6.0" jupyterlab

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

Open a terminal in your NMR folder, activate ``.venv`` again using the command
above, and run ``python -m jupyter lab``. Your notebooks stay in that folder.

If installation or opening a notebook fails, see
:doc:`installation troubleshooting <troubleshooting>`.
