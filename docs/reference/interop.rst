Interoperability and referencing
================================

Convert normalized nmrglue-ng results to SpectroChemPy datasets and
work with NMR frequency references. For exchange files, start with
the :ref:`public reader <reading-exchange-formats-through-interop>`.

Raw 2D component contract
-------------------------

For qualified QF, States, States-TPPI, TPPI and echo/anti-echo inputs,
``read()`` and ``from_interop()`` arrange the declared components and retain
the physical axes and acquisition widths. Explicit component labels determine
the ordering: permutations of the same complete label set produce the same
native array. A selected States input convention is converted at this boundary,
with the operation recorded in the dataset metadata.

``Experiment(dataset).process()`` selects the Fourier reconstruction from
quadrature and component meaning. It does not select an algorithm from the
vendor, file profile or experiment name. A saved SCP dataset retains this
information and can be processed without reopening the original file.
Echo/anti-echo inputs use the normalized Rance–Kay arrangement by default;
``echo_antiecho="rance-kay"`` is optional, and the historical
``"bruker-rance-kay"`` spelling remains an alias. Digital-filter correction
uses the imported delay descriptor or an explicit caller override, rather
than rereading vendor parameters during this common processing path.

An undetermined physical quadrature remains unsupported for a complete
indirect transform. Importing stored components does not establish that
quadrature. This contract does not qualify additional file formats or modes.
Recorded TopSpin replay and specialized experimental helpers remain separate
entry points.

JCAMP-DX FID pages and QF processing
------------------------------------

With nmrglue-ng 1.4.0 or later providing ``jcampdx-fid-2d``, use the explicit
interop reader for regular ``T1,T2,R,I`` FID pages. Confirm QF from the
acquisition convention; the declared scheme must also be non-phase-sensitive.
The complex-sign choice is explicit. For a TopSpin export whose stored
complex samples are to be retained:

.. code-block:: python

   import spectrochempy as scp

   raw = scp.nmr.read(
       "qf_fid.dx", protocol="interop", profile="jcampdx-fid-2d",
       normalize_options={"quadrature": "qf", "conjugation": "none"},
   )
   spectrum = scp.nmr.Experiment(raw).process()

The existing generic QF processor consumes the normalized components and
time grids; it does not recognize COSY or HMBC names. Results have relative
frequency coordinates in Hz. Reading does not remove the digital filter or
establish a chemical-shift reference, and this FFT-only example does not
apply phasing, apodization or magnitude. Inspect any recorded group delay
and qualify its correction before interpreting phases or intensities.
Saving the raw dataset as SCP retains its processing contract.

Echo/anti-echo processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Development nmrglue-ng also supports declared echo/anti-echo FIDs with direct
simultaneous DQD acquisition. Confirm that each adjacent trace pair belongs
to one physical indirect increment, and establish the time between pairs
from acquisition provenance. Do not infer it by simply subsampling the stored
T1 row coordinates. The reader preserves the pair as two complex channels;
``Experiment.process()`` performs the existing Rance-Kay reconstruction.

For the qualified EPFL HSQC acquisition, the indirect spectral width checked
against TopSpin import is 16608.16082508 Hz. For another acquisition, replace
this value with its established width:

.. code-block:: python

   indirect_width_hz = 16608.16082508
   raw = scp.nmr.read(
       "echo_antiecho_fid.dx", protocol="interop", profile="jcampdx-fid-2d",
       normalize_options={
           "quadrature": "echo-antiecho", "conjugation": "none",
           "confirm_adjacent_pairs": True,
           "indirect_dwell_s": 1.0 / indirect_width_hz,
       },
   )
   spectrum = scp.nmr.Experiment(raw).process()

The result has four signed component planes (rr, ri, ir, ii), with relative
Hz axes. No filter correction or chemical-shift referencing is implicit.
If an unapplied positive group delay is recorded and its meaning has been
established, phase-domain correction can be requested explicitly:

.. code-block:: python

   corrected = scp.nmr.Experiment(raw).process(digital_filter="phase")

The EPFL reference records a delay of 76 complex direct points. Four TopSpin
compression modes give identical normalized data and processed planes.
The planes were compared to NMRPipe with and without phase-domain correction;
this qualifies the stated operations, not an absolute experimental phase or
ppm reference. Time-domain correction is also checked against the native
Bruker route. Saved SCP data retains the common processing contract, without
requiring the source reader or original vendor metadata during processing.
States and States-TPPI JCAMP normalization remain unsupported.

.. currentmodule:: spectrochempy_nmr

.. autosummary::
   :nosignatures:
   :toctree: generated

   from_interop
   set_nmr_context

Export processed spectra
-------------------------

See :doc:`/userguide/exchanging_spectra` for the supported components,
axis units and round-trip examples.

.. autosummary::
   :nosignatures:
   :toctree: generated

   write_jcamp
   write_csdf

Experimental entry points
-------------------------

.. autosummary::
   :nosignatures:
   :toctree: generated

   read_agilent_interop
   reference_agilent_1d
   reference_interop_1d
