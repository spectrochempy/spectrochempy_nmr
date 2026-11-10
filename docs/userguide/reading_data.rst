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
