Reading NMR data
================

``scp.nmr.read(...)`` dispatches public readers for TopSpin/Bruker, Agilent or
Varian, JEOL, TecMag and SIMPSON inputs. A protocol can be supplied explicitly
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

The reader scope includes 1D and 2D raw/processed data. 2D reading and
native complex-channel storage must not be confused with validated 2D processing;
the latter is outside the current public workflow.

The separate experimental ``read_agilent_interop`` facade is characterized on
real acquisitions with a pinned external provider. See
:doc:`../agilent_qualification` for reproducible checks and the known
observed-nucleus metadata blocker.
