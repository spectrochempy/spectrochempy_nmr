Reading NMR data
================

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
:doc:`../reference/nmr_api` for reader options.

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

The default ``quadrature="auto"`` recognizes versioned pulse-program
sources. Pass the exact source used for acquisition with
``pulse_program_source="path/to/NOESY.c"`` or include the source next to
``fid`` and ``procpar``, named after ``seqfil`` with a ``.c`` suffix.
The initial catalogue recognizes the audited OpenVnmrJ `NOESY.c
<https://github.com/OpenVnmrJ/OpenVnmrJ/blob/0f6b0a4664def0cdf2ee9eb37cb2df7369f38ca3/src/psglib/NOESY.c>`_
as States-TPPI when the name and SHA-256 match and ``d2=0`` establishes
the first increment. The source must correspond
to the acquisition. A sequence name or a two-branch phase array alone
cannot distinguish States from States-TPPI.

The reader logs the choice at INFO level and records the mode and its
source in ``fid.meta.acquisition["quadrature_selection"]``. For an unknown
program or absent source, automatic selection leaves the mode unset and
F1 processing guarded. Reading the stored traces and the historical F2
operation remain available. Quadrature selection requires an nmrglue-ng
provider that supports it and a SpectroChemPy core with FFT
prevalidation (1.2.0 or later).

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
input file with a sibling output. Qualified standalone FIDs use calibrated
time coordinates. Qualified standalone spectra use frequency coordinates
in Hz. The direct FFT is available for qualified 2D FIDs. Their indirect
quadrature remains unqualified. Files with simulation metadata from an
``.in`` file and other legacy layouts use the compatibility presentation.
nmrglue-ng decodes these data files. For indexed XYREIM, the plugin keeps
the historical ascending order on both axes.

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
