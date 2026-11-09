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

For the explicit, optional NMRPipe processed 1D interop path, see
:doc:`../nmrpipe_interop`. This does not change automatic reader dispatch.

See :doc:`../nmrml_interop` for explicit 1D nmrML exchange-format profiles,
writer round trips and the required convention choices.

SIMPSON
-------

Use ``scp.nmr.read_simpson(...)`` for SIMPSON data files or a simulation
input file with a sibling output. Qualified standalone FIDs use calibrated
time coordinates. Qualified standalone spectra use frequency coordinates
in Hz. The direct FFT is available for qualified 2D FIDs. Their indirect
quadrature remains unqualified. Files with simulation metadata from an
``.in`` file and other legacy layouts use the compatibility presentation.

Tecmag
------

Tecmag TNT acquisition files are no longer read by the plugin. Previously
saved Tecmag datasets retain their metadata interpretation.
