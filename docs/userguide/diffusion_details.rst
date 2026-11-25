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
unless overridden, with no exponential broadening by default. Before F2,
inherited phase state is reset so the saved common phase is applied exactly
once, including on older Core versions that phase during FFT. Each call
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

Start with :doc:`diffusion_dosy` for the experimental map recipe.

The DOSY recipe applies the same 1 Hz exponential F2 apodization to all
rows before the FFT. On PEG 600, the RMS after removing a straight
background in 73.8–73.3 ppm falls from 23.04 to 10.05 signal units across
the twelve rows. The monoexponential D in 72.649–72.916 ppm changes from
7.58610 × 10⁻¹² to 7.58665 × 10⁻¹² m²/s (+0.0074%). These are comparisons
for this acquisition and window, not an uncertainty estimate. Apodization
correlates noise and broadens lines; increasing it can mix nearby signals
and change finite-window areas. No per-row rescaling is applied.
The PFG reference recipe keeps ``lb=0`` for its archived comparison.

For each ppm column, the map fits nonnegative amplitudes on a uniform
logarithmic D grid with K(b, D) = exp(-b D). The objective is
mean((K a - signal)²) + λ sum(a²), with λ = 0.001 by default. The
implementation uses `SciPy's nonnegative least-squares solver
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html>`_ on the
augmented matrix; no extra library import is needed in the recipe.
Map values are signal amplitudes per log-D bin, not probability densities.
Changing the grid or penalty changes the apparent width and separation.
The method applies no baseline correction, clipping, per-row normalization
or offset. Positive and negative residuals are retained.

The result provides ``map``, ``measured``, ``fitted``, ``residual`` and
``diagnostics``. The latter includes the amplitude fraction in the first
and last D bins for every ppm column; appreciable boundary intensity calls
for reviewing the grid and data. A successful inversion does not establish
component count, molecular identity, diffusion resolution or uncertainty.
A small residual alone cannot establish these either.
Noise and imperfect baselines can accumulate intensity at the grid edges.
Summing all ppm columns, including baseline, is therefore not a measure of
the sample's diffusion distribution. The plot starts at 5% of the strongest
map intensity; this changes the display only.

The PEG 600 example checks the map on a real acquisition; it does not
establish the separation of different species. Numerical tests use
simulated mixtures with known D, noise, overlapping peaks, reversed axes
and reconstruction checks. PALMA remains outside this public workflow.

Dataset provenance
-------------------

* File: ``nmrdata/bruker/tests/nmr/relax/102/ser``
* Author: Cassandre Kouvatas
* Licence: CC BY 4.0
