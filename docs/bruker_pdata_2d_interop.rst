Bruker processed 2D interop (Unreleased)
=======================================

For a complete Bruker ``pdata`` directory containing ``2rr``, ``2ri``, ``2ir``
and ``2ii``, use the qualified reader from the optional ``nmrglue-ng``
provider::

   import nmrglue as ng
   from spectrochempy_nmr import from_interop

   stored = ng.interop.read("experiment/pdata/1", profile="bruker-pdata-2d")
   spectrum = from_interop(ng.interop.normalize(stored))

The result has dimensions ``(c, y, x)`` with physical axes F1 and F2 on
``y`` and ``x``. The labels of ``c`` are the original Bruker file names, in
their supplied order. The adapter keeps the four planes separate and
uses the calibrated axes from the provider. Hypercomplex is unnecessary
for this conversion.

The qualified reader applies ``2**NC_proc`` by default and produces
float64 planes. Pass ``scale_data=False`` to retain the stored int32 values.
``from_interop()`` preserves either dtype and records the provider's
scaling metadata without applying another scale. The core SCP reader
can widen int32 to float64 on reload while retaining all sample values.

CI checks a synthetic big-endian tiled directory. The local qualification
checks the complete proton-exchange 2D reference with and without scaling,
using the binary files and the calibrated grids from ``procs`` and
``proc2s``. The test pins the six file checksums and verifies sample values,
axis references and labels after SCP reloading. The reference remains
in the local corpus. Set ``NMR_PDATA_2D_REFERENCE`` to its ``pdata/1``
directory and run::

   python -I -m pytest tests/qualification/test_bruker_pdata_2d.py

This qualifies conversion of complete, already processed planes. The
provider's storage-only reader also supports layouts outside this
calibrated profile. Raw processing and processing via ``Experiment``
remain separate.
