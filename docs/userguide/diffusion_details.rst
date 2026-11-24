PFG measurement conventions and reference
=========================================

Start with :doc:`diffusion_pfg` for the executable recipe.

The measured coefficient
-------------------------

``Diffusion.fit(region=...)`` fits the unweighted monoexponential model
I(b) = I0 exp(-b D), without an offset. It returns an apparent D in m²/s,
I0, measured areas, fitted areas, residuals and optimizer diagnostics.
A successful optimization does not establish a single physical component,
an independent gradient calibration or an uncertainty on D. The
``at_lower_bound`` diagnostic flags an effectively zero fitted decay rate.

The ppm limits select the nearest bins and include both endpoints. Area is
the sum of those bins times their positive, uniform ppm spacing. No baseline
correction or normalization by each row's maximum is performed. The reported
area units combine the spectral signal unit and ppm. The b axis is never
Fourier transformed.

Factors b and raw processing
----------------------------

Automatic reading is limited to Bruker ``diffSte`` with QF series rows,
DQD direct acquisition and a ``diff.xml`` table of indexed B values.
Those exported values are read in s/m² and attached in acquired row order.
They are not inferred from a gradient percentage or recalibrated by the
plugin. A missing or unsupported table requires calibrated b supplied with
units. For another pulse sequence, prepare phased real spectra and supply
those spectra together with calibrated b; see :doc:`/reference/diffusion`.

For the raw workflow, ``process()`` removes the integer part of the recorded
Bruker digital delay with the existing time-domain correction and compensates
the fractional remainder on the final F2 grid. It uses the saved common phase
unless overridden, with no exponential broadening by default. Each call
starts from the original SER. Phase and integration choices are recorded in
the spectra and fitted curves' diffusion metadata.

PEG 600 comparison
-------------------

The public acquisition is ``nmrdata/bruker/tests/nmr/relax/102``. The
historical qualification is available in `NMR PR 107
<https://github.com/spectrochempy/spectrochempy_nmr/pull/107>`_. It reconstructed
the twelve b values to within 8.34 × 10⁻⁶ relative to the exported table.
Using the same exported b grid and fitting method for both area series,
the residual difference in D was +0.351% in the first band. This compares two
processing results for the same acquisition, not a certified diffusion standard.

The new public recipe reads that exported table directly. With its inclusive
nearest-bin integration and a 65,536-point F2 transform, the first band gives
approximately 7.59 × 10⁻¹² m²/s, about +0.36% relative to the archived TopSpin
value 7.559092 × 10⁻¹² m²/s. The different numerical fit settings and bin
selection need not reproduce every digit of the historical script.
The original processed spectrum is unavailable, so the historical integration
rule and exact source of the remaining difference are unresolved.

DOSY inversion
---------------

A diffusion distribution or a DOSY map requires a separate inversion and
validation. PALMA and NNLS were evaluated experimentally outside the public
plugin API. This recipe provides neither solver and makes no claim of
resolving multiple diffusion components.

Dataset provenance
-------------------

The measurement author is **Cassandre Kouvatas**, confirmed by the maintainer
on 24 November 2026. The acquisition records name the shared
``nmr`` account on ``lcs175n`` on 16 June 2023. Their dataset path contains
``Cassandre/13C_PEG600_8mm_diff/102``. The archived processing log records
another F2 processing on 30 June 2023 under the ``Cassandre`` Windows account.

Christian Fernandez imported the files through `data PR 22
<https://github.com/spectrochempy/spectrochempy_data/pull/22>`_ on 5 July 2023.
That contribution does not establish experimental authorship. The previous
credit to Christian Fernandez as the measurement author is withdrawn.
The maintainer confirms that this acquisition is licensed under CC BY 4.0,
the same licence as his files. Credit Cassandre Kouvatas as the author.
