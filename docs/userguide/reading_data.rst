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

TopSpin acquisition and ``procs`` values are exposed as metadata where
available. ``vendor_profile`` describes the vendor file; it does not claim to
be a replayable historical processing recipe. Compatibility aliases
``scp.nmr.read_topspin`` and ``scp.read_topspin`` remain available.

The readers cover 1D and 2D raw and processed data. The available processing
steps depend on the acquisition profile.

The separate experimental ``read_agilent_interop`` facade is characterized on
real acquisitions with a pinned external provider. See
:doc:`../agilent_qualification` for reproducible checks, observed-nucleus handling and remaining limitations.

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

Reading exchange formats through interop
----------------------------------------

Use ``scp.nmr.read(path, protocol="interop")`` to read and normalize one file
with nmrglue-ng, then convert the result to an ``NDDataset``. NMRPipe profiles
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

Tecmag
------

Tecmag TNT acquisition files are no longer read by the plugin. Previously
saved Tecmag datasets retain their metadata interpretation.
