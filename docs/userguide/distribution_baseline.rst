Experimental distribution baseline
==================================

A core build exposing ``Baseline(model="distribution")`` provides an optional
noise-based classifier followed by local interpolation and mean smoothing.
The default baseline models remain unchanged. This option is experimental:
it can remove broad weak resonances as if they were background. Inspect the
estimated baseline and peak integrals before using the corrected spectrum
quantitatively. It is not a validated general-purpose replacement for
peak-free polynomial support regions.

The core implementation uses NumPy/SciPy and adds no NMR package dependency.
An installed core that lacks this model cannot use the examples below.
The plugin's optional path reports that requirement explicitly; updating the
plugin alone does not add the core model.

On a phased real spectrum
-------------------------

Use the same processor as for other core baseline models. The following
reproducible synthetic example has a narrow positive resonance, a linear
background and weak Gaussian noise. Widths are sample counts: the default
``distribution_width=20`` uses up to 41 centered samples, independently of
whether the axis is in ppm or Hz.

.. code-block:: python

   import spectrochempy as scp

   axis = scp.Coord.linspace(10, 0, 4096, units="ppm", title="chemical shift")
   peak = scp.gaussian(axis, pos=5, width=0.03, normalized=False)
   noise = scp.normal(scale=0.0002, size=axis.size, seed=111)
   source = scp.NDDataset(
       peak.data + 0.025 + 0.0005 * axis.data + noise.data,
       coordset=[axis],
       units="a.u.",
   )
   processor = scp.Baseline(model="distribution", distribution_width=20)
   processor.fit(source)
   estimated_baseline = processor.baseline
   corrected = processor.corrected
   _ = scp.plot_multiple(
       [source, estimated_baseline, corrected],
       labels=["Input", "Estimated baseline", "Corrected"],
   )

The source is retained. The outputs keep the axis, units and shape, and the
corrected dataset records the model and effective width in its history.
The same options work through ``source.get_baseline(...)`` and
``source.basc(...)``. The model supports finite real unmasked 1D data or
row-wise 2D data on regular grids. Multivariate fitting, LLS transforms and
breakpoints are unsupported. A classification with no noise-only support
raises an error rather than guessing a baseline.

Direct masked input is rejected so that sample positions are never silently
compressed. The width is not a physical linewidth: review it when changing
the point spacing. The diffusion-edited extension and resampling described
in the original publication are not implemented.

After automatic phase correction
--------------------------------

For a complex frequency-domain spectrum, pass the options through the
plugin's existing ``autophase`` entry point. Phase the complex spectrum
first and fit the baseline on its real absorption channel:

.. code-block:: python

   from spectrochempy_nmr import autophase

   result = autophase(
       complex_spectrum,
       baseline_options={"model": "distribution", "distribution_width": 20},
   )
   if result.estimate.success:
       phased_complex = result.phased
       estimated_baseline = result.baseline
       corrected_real = result.corrected

Keep ``result.phased`` for any later phase correction; the imaginary channel
has not undergone this real-only baseline subtraction. Phase estimation has
its own reference and exclusion options; see :doc:`processing_1d`.

For this fit, the plugin interpolates masked absorption values in sample
index on a separate unmasked working copy, with nearest valid values at the
ends. It restores the source mask on the returned baseline and corrected
real spectrum. The original masked samples and the complex intermediate are
preserved. This filling strategy does not establish the baseline inside a
missing region; inspect the neighborhood of masked gaps, especially where
they overlap resonances. The history records the filling rule, experimental
status and effective width.

Scientific scope
-----------------

The algorithm follows the main routine in Wang et al., *Analytical Chemistry*
2013, 85, 1231-1239, `Supporting Information S5-S7
<https://doi.org/10.1021/ac303233c.s001>`_. It uses locally averaged segment
endpoints to resolve the supplement's Y/R interpolation ambiguity, population
standard deviations and positive noise estimates. Those choices are explicit;
this is not a bitwise reproduction of the authors' software.

Synthetic qualification supports narrow positive and negative peaks. A broad
low-contrast test peak was almost entirely classified as baseline, which is
a counterexample to universal quantitative use. No new instrumental corpus
qualification is asserted for this option.
