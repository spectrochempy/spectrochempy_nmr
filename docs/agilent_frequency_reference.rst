Agilent 1D frequency referencing
================================

This **Unreleased, experimental** qualification covers the two original
acquisitions described in :doc:`agilent_qualification`. It does not change
public reader dispatch or establish which reference is chemically correct
when a file contains contradictory parameters.

Reference evidence
------------------

The vendor parameters have distinct roles:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Unit
     - Meaning
   * - ``sfrq``
     - MHz
     - Observe transmitter frequency, used for an absolute center estimate.
   * - ``reffrq``
     - MHz
     - Chemical zero frequency and Hz/ppm conversion factor.
   * - ``rfl``
     - Hz
     - Reference-line position measured from the spectrum's right edge.
   * - ``rfp``
     - Hz
     - Assigned frequency of that reference line relative to chemical zero.
   * - ``tof``
     - Hz
     - Transmitter offset; not automatically a chemical-shift carrier.

These definitions follow the vendor's `VNMR Getting Started, Referencing
<https://www2.chem.wisc.edu/~cic/nmr/Guides/Varian_VNMR_manuals/VNMR_Getting_Started--VNMR_6_1C_01-999160-00_rev_A0800--0199916000a.pdf>`_
and `Agilent VnmrJ 3.2 Parameter Reference
<https://imserc.northwestern.edu/downloads/nmr-vj_cpr.pdf>`_. The two candidate
center offsets relative to chemical zero are:

.. math::

   c_{graphical} = SW/2-rfl+rfp

   c_{absolute} = 10^6(sfrq-reffrq).

For the hop proton acquisition they are 4806.1110128254 and
4806.111000107194 Hz: agreement within 0.000013 Hz. For the historical carbon
acquisition they are 0 and 281.6000000081 Hz: a real conflict. Its chemical
reference cannot be chosen from this evidence alone. Both original reference
sets remain in acquisition metadata; ``tof`` is not substituted for either.

Explicit application after FFT
------------------------------

.. code-block:: python

   from spectrochempy_nmr import read_agilent_interop, reference_interop_1d

   fid = read_agilent_interop("acquisition.fid")
   spectrum = fid.fft()  # centered Hz, not a chemical reference
   referenced = reference_interop_1d(spectrum, reference="rfl-rfp")
   # An explicit alternative is reference="sfrq-reffrq".
   # units="Hz" keeps a referenced Hz axis with the same ppm context.

``reference`` is required. The operation returns a copy, preserving sample
values, masks and original acquisition metadata. It accepts only the qualified
complex 1D interop profile on its complete ascending centered Hz FFT grid.
Cropped, reordered, already referenced and disabled-reference inputs are
refused. ``reffrq`` must be finite and positive. Missing calibration is not
guessed from observation frequency alone.

``meta.frequency_reference`` records the selected model, both center estimates,
their absolute disagreement, reference frequency and FFT-grid evidence. The
``consistent_within_0_1_hz`` field is an absolute comparison at 0.1 Hz, not a
metrological uncertainty or a rule for choosing a model. Selecting one model
on the conflicting carbon file is an explicit interpretation, not proof of
its experimental correctness.

Orientation and precision
-------------------------

The core FFT is the centered negative-sign DFT. Agilent's default ``var2pipe``
sign adjustment conjugates the FID; NMRPipe's positive-sign FT then gives the
conjugate of the core FFT at the same indices. The matching chemical axis is:

.. math::

   f_i = (i-\lfloor N/2\rfloor) SW/N, \qquad
   \delta_i = (c-f_i)/reffrq.

No conjugation is applied by the referencing operation. The axis decreases
with index; its spacing is ``-SW/(N*reffrq)`` ppm, not an endpoint-inclusive
``SW/(N-1)`` grid. The zero-frequency bin is at index ``N//2``, including odd
sizes. The external NMRPipe comparison here covers the even-sized real FIDs.

Released core coordinates can round intermediate values during conversion and
assignment. Input-grid validation allows those rounding bounds, capped below a
quarter bin, and records the measured error and tolerance. The final axis is
rebuilt from the source spectral width and FFT size at 15 significant digits.
Collapsed or unrepresentable coordinates are rejected.

Persistence and inverse transforms
----------------------------------

SCP save/reload preserves the ppm axis, reference evidence and Hz conversion
context. A same-size inverse FFT restores the time grid and moves the applied
choice to ``meta.last_frequency_reference``. A subsequent forward FFT remains
unreferenced Hz; a reference is never reapplied silently. Resized inverse
transforms and physical phase/amplitude calibration are not qualified here.

External numerical oracle
-------------------------

``tools/qualify_agilent_frequency.py`` compares a temporary acquisition copy
with NMRPipe, using independently supplied OBS/CAR rather than taking the
oracle parameters from the function under test. For the graphical model::

   python tools/qualify_agilent_frequency.py /path/to/FAM013_AHTM.PROTON_04.fid \
     --nmrpipe-bin /path/to/nmrpipe/bin --reference rfl-rfp \
     --observation-mhz 599.826355589 --carrier-ppm 8.01250389890927

   python tools/qualify_agilent_frequency.py /path/to/agilent_1d \
     --nmrpipe-bin /path/to/nmrpipe/bin --reference rfl-rfp \
     --observation-mhz 125.6808291 --carrier-ppm 0

NMRPipe 13.0 Rev 2026.072.12.03 reproduced the conjugation convention with
relative complex-spectrum error below 2e-7 on both acquisitions. Maximum ppm
axis differences were below 1e-5 ppm; the tool allows 2e-5 ppm for float32
NMRPipe headers. This verifies numerical conventions for an explicitly supplied
reference, not the validity of the chemical standard or a vendor processing
recipe. Originals are hash-checked before/after; no binaries or spectra are
redistributed. NMRPipe remains optional and is not required by CI.

Normalized Experiment workflow (Unreleased)
-------------------------------------------

The reader is format-specific; the normalized processing workflow is not::

   from spectrochempy_nmr import read_agilent_interop
   from spectrochempy_nmr.experiment import Experiment

   fid = read_agilent_interop("/path/to/acquisition.fid")
   spectrum_hz = Experiment(fid).process(apodization="em", lb=1, size=65536)
   spectrum_ppm = Experiment(fid).process(reference="rfl-rfp")

``Experiment`` classifies normalized input using its structural contract and
current domain. It consumes ``acquisition.frequency_axis`` for explicit
referencing. Candidate labels are opaque: a different producer can provide
other labels, either frequency sign, and its own chemical-zero frequency.
Changing or omitting origin, the descriptive profile name, or raw metadata does
not select a different processing path. Archived ``interop_domains`` describes
the input; ``data_domain`` tracks the current processed state.

Without ``reference``, FFT output remains on the centered Hz grid. Missing
reference evidence does not prevent a calibrated time-domain FFT. A selected
reference requires the normalized numeric descriptor; raw procpar fields are
never interpreted by this workflow. Older provider results can still be
transformed but must be read again with the qualified provider to reference.
``reference_agilent_1d`` remains a compatibility name for the generic helper.

Only one complex 1D acquisition is supported here. Independent acquisition
stacks, uncalibrated or nonuniform time axes and masked FIDs are rejected.
The processing trace records explicit operations and the selected reference;
source samples and acquisition metadata are preserved. This does not switch
legacy readers to interop or qualify arbitrary vendors or multidimensional data.
