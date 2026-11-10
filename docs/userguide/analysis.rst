Analyzing NMR spectra
=====================

Use the core analysis tools on the ``NDDataset`` returned by a reader or
by :doc:`processing_1d` and :doc:`processing_2d`. Begin with a phased spectrum.
Use its real component for peak finding and integration. On a spectrum
with several quadrature components, select the physical component you
want to analyze, typically RR for an absorptive 2D spectrum.

Find peaks and integrate a 1D region
------------------------------------

Continue from the :doc:`../getting_started/quickstart` spectrum. Choose a
baseline correction appropriate to the signal. Here, SNIP estimates the
baseline before peak picking and integration. The intensity threshold and
integration limits are specific to this example.

.. code-block:: python

   corrected = spectrum.real.snip()
   peaks, properties = corrected.find_peaks(height=50000.0, use_coord=False)
   region = corrected[-0.1:1.5]
   area = region.trapezoid()

The slice uses chemical shifts in ppm. ``use_coord=False`` makes peak width
and spacing options work in samples. The integral uses the region's
calibrated axis. Choose integration limits that contain the full peak
and avoid nearby signals when comparing peak areas.

For peak assignment on a plot and fitting overlapped 1D peaks with
``scp.Optimize``, see :doc:`/examples/gallery/nmr/plot_processing_nmr`.

Find cross-peaks and integrate a 2D region
------------------------------------------

Use the real COSY plane from :doc:`../nmrpipe_interop`. Supply both
frequency dimensions explicitly to peak finding. This example integrates
a rectangular region in Hz using the core's trapezoid rule along each axis.

.. code-block:: python

   plane = scp.nmr.read("nmrpipe_2d_cosy_example.ft2", protocol="interop")
   result = plane.find_peaks(dims=("y", "x"), as_result=True)
   region = plane[2250.0:2450.0, 880.0:1010.0]
   area = region.trapezoid(dim="x").trapezoid(dim="y")

Adjust the bounds to the cross-peak of interest. For negative peaks, apply
peak finding to the negated map. Select the same quadrature component and
sign convention when comparing spectra. Core versions that expose
``integrate_region()`` also support physical bounds and an optional
uncertainty estimate from a noise region.

For segmentation, peak simulation, and multidimensional fitting,
see the :external+core:doc:`core analysis guide <userguide/analysis/index>`.
For an example of fitting relaxation curves across a series, see
:doc:`/examples/gallery/nmr/plot_processing_nmr_relax`.
