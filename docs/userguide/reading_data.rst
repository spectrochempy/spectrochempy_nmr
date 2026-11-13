Reading NMR data
================

The development version of the Bruker reader leaves digital-filter and
DC-offset correction to processing by default. They decode the data, represent the acquisition
quadrature, and calibrate the time coordinates when the format provides
the required metadata. Bruker FID and SER readers also normalize amplitudes
by scan count and receiver gain. Use :doc:`processing_1d` or
:doc:`processing_2d` to select a correction during processing.

``scp.nmr.read(...)`` dispatches public readers for TopSpin/Bruker, Agilent or
Varian, JEOL and SIMPSON inputs. A protocol can be supplied explicitly
when automatic detection is not appropriate.

.. code-block:: python

   import spectrochempy as scp

   bruker = scp.nmr.read("path/to/1/fid")
   processed = scp.nmr.read("path/to/experiment", expno=1, procno=1)
   agilent = scp.nmr.read("path/to/fid", protocol="agilent")

Reader catalogue
----------------

Use the calls below with your file or experiment directory. The available
processing steps depend on the acquisition and file layout. See the
:doc:`../reference/readers` for reader options.

.. list-table::
   :header-rows: 1
   :widths: 27 34 39

   * - Format
     - Recommended call
     - Typical inputs
   * - Bruker TopSpin
     - ``scp.nmr.read_topspin(path)``
     - FID, SER, or processed ``pdata`` in an experiment directory
   * - Agilent/Varian
     - ``scp.nmr.read_agilent(path)``
     - Acquisition directory containing ``fid`` and ``procpar``
   * - JEOL
     - ``scp.nmr.read_jeol(path)``
     - JDF files
   * - SIMPSON
     - ``scp.nmr.read_simpson(path)``
     - Simulated FIDs and spectra, or a simulation input with a sibling output
   * - NMRPipe
     - ``scp.nmr.read(path, protocol="interop")``
     - Supported time- and frequency-domain files
   * - nmrML
     - ``scp.nmr.read(path, protocol="interop", profile=...)``
     - Supported acquisition and spectrum profiles
   * - NMR JCAMP-DX
     - ``scp.nmr.read(path, protocol="interop", profile=...)``
     - Supported NMR profiles, such as ``"jcampdx-spectrum-1d"``

See :ref:`reading-exchange-formats-through-interop` for the profile and
convention options for exchange formats.

Read an already processed TopSpin spectrum
------------------------------------------

Pass ``procno`` to select the processed spectrum in an experiment. The
returned dataset already has a frequency axis and can be plotted or
analyzed directly.

.. code-block:: python

   spectrum = scp.nmr.read_topspin("path/to/experiment", expno=1, procno=1)
   _ = spectrum.real.plot()

Use ``proc_axis=True`` to use the reference saved during TopSpin processing.
This is useful when the spectrum was re-referenced after acquisition.

Bruker F1 acquisition mode
--------------------------

For a raw Bruker SER, the reader uses the indirect acquisition parameter
``FnMODE``. If it is missing or undefined, the reader uses ``MC2`` from
the indirect processing parameters when available.

When the mode is still unknown, you can read the traces and transform F2.
If you need F1, reopen the acquisition with its known mode, for example
``scp.nmr.read_topspin(path, quadrature="states-tppi")``, then apply the
F2 and F1 transforms in order. The accepted choices are ``"qf"``,
``"states"``, ``"states-tppi"`` and ``"echo-antiecho"``.

Agilent/Varian States quadrature
--------------------------------

For a paired 2D acquisition with ``phase=[1,2]``, choose the F1 convention
explicitly when it is known::

   fid = scp.nmr.read_agilent("sample.fid", quadrature="states")
   spectrum = scp.nmr.Experiment(fid).process()

Use ``quadrature="states-tppi"`` if the acquisition encodes sign alternation
in F1. Both choices retain the two stored branches at import and use the
physical time grids. Processing applies the declared NMRPipe input
convention and performs F2 before F1. Only States-TPPI applies F1
alternation.

The default ``quadrature="auto"`` uses recognized pulse-program sources.
Supply the source used for the acquisition with ``pulse_program_source``,
or put it next to ``fid`` and ``procpar`` with the name given by ``seqfil``
and a ``.c`` suffix. Check the selected quadrature
in the reader's log. If the mode remains undetermined, you can read the
traces and transform F2. Choose ``"states"`` or ``"states-tppi"`` explicitly
before transforming F1, once you know the acquisition convention.

.. _reading-exchange-formats-through-interop:

Reading exchange formats through interop
----------------------------------------

Use ``scp.nmr.read(path, protocol="interop")`` to read and normalize one file
with
`nmrglue-ng <https://nmrglue-ng.readthedocs.io/latest/tutorials/interop.html>`_,
then convert the result to an ``NDDataset``. NMRPipe profiles
can be detected from the header. Supply an explicit provider profile for
nmrML and NMR JCAMP-DX.

.. code-block:: python

   pipe_spectrum = scp.nmr.read("spectrum.ft2", protocol="interop")
   jcamp_spectrum = scp.nmr.read(
       "epicatechin.jdx", protocol="interop", profile="jcampdx-spectrum-1d"
   )
   xml_spectrum = scp.nmr.read(
       "spectrum.nmrML",
       protocol="interop",
       profile="nmrml-1d",
       read_options={"read_processed": True, "axis_endpoint": "last-point"},
   )

Use ``read_options`` for file decoding and ``normalize_options`` for
convention choices, such as the sign of an FID. The optional
``conversion_profile`` confirms an interpretation in ``from_interop()``,
such as ``"states-tppi"`` for adjacent NMRPipe branches with alternation
still encoded. The reader does not run an FFT. Unsupported profiles or
missing convention choices raise the corresponding provider or converter
error. The nmrML reader requires ``xmltodict``.

The usual TopSpin, Agilent, JEOL and SIMPSON dispatch continues to apply
outside this mode. Use the core's ``scp.read_jcamp()`` for infrared JCAMP-DX.
See :doc:`../nmrpipe_interop` and :doc:`../nmrml_interop` for the underlying
profiles and their required convention choices.

SIMPSON
-------

Use ``scp.nmr.read_simpson(...)`` for SIMPSON data files or a simulation
input file with a sibling output. Supported standalone FIDs use calibrated
time coordinates. Supported standalone spectra use frequency coordinates
in Hz. The direct FFT is available for supported 2D FIDs. The indirect quadrature convention
has yet to be established. When reading a simulation input file, keep
its output in the same directory.

Additional formats available in nmrglue-ng
------------------------------------------

The following readers can be called directly with ``import nmrglue as ng``.
Their results use the nmrglue-ng API. If you need one of these formats in
``scp.nmr.read()``, open a request in the
`NMR plugin issue tracker <https://github.com/spectrochempy/spectrochempy_nmr/issues>`_
and describe the acquisition and the workflow you need.

.. list-table::
   :header-rows: 1

   * - Format
     - Direct reader
   * - Tecmag TNMR
     - ``ng.tecmag.read(path)``
   * - Magritek Spinsolve
     - ``ng.spinsolve.read(path)``
   * - RS2D
     - ``ng.rs2d.read(path)``
   * - Sparky UCSF
     - ``ng.sparky.read(path)``
   * - Rowland NMR Toolkit
     - ``ng.rnmrtk.read(path)``

Previously saved SpectroChemPy datasets from Tecmag retain their metadata
interpretation.
