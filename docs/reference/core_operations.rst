.. _nmr-core-operations:

Core operations used in NMR
===========================

The links below open the core API reference. The NMR notes explain when to
use each operation on an acquisition or spectrum.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Core API
     - NMR use
   * - :external+core:py:func:`~spectrochempy.em`,
       :external+core:py:func:`~spectrochempy.gm`,
       :external+core:py:func:`~spectrochempy.sp`
     - Apply exponential, Gaussian or sine-bell weighting to time-domain
       data. For explicit 1D processing with a recorded trace, see
       :doc:`/userguide/processing_1d`.
   * - :external+core:py:func:`~spectrochempy.zf_size`
     - Set the number of points before Fourier transformation. Zero filling
       makes the frequency grid denser while retaining the acquisition's
       physical resolution.
   * - :external+core:py:func:`~spectrochempy.fft`,
       :external+core:py:func:`~spectrochempy.ifft`
     - Transform along a selected dimension. For paired 2D acquisitions,
       use the :doc:`supported processing paths </userguide/processing_2d>`
       to respect the indirect quadrature and dimension order.
   * - :external+core:py:func:`~spectrochempy.pk`
     - Apply zero- or first-order phase correction to a complex spectrum.
       In core 1.2, a pivot given as a quantity selects the nearest sample
       in the axis units. A plain number specifies a sample index.
   * - :external+core:py:func:`~spectrochempy.dc`
     - Subtract a constant estimated from the end of a time-domain signal.
       For a normalized FID, :meth:`~spectrochempy_nmr.Experiment.prepare_fid`
       provides a preparation step that tracks acquisition corrections.
   * - :external+core:py:class:`~spectrochempy.NDDataset`,
       :external+core:py:class:`~spectrochempy.Coord`,
       :external+core:py:class:`~spectrochempy.CoordSet`
     - Inspect the samples and calibrated axes. Conversion between hertz
       and ppm requires the reference frequency for the selected axis.
   * - :external+core:py:func:`~spectrochempy.plot`,
       :external+core:py:func:`~spectrochempy.plot_map`
     - Display a 1D spectrum or a 2D map. For paired 2D data, select the
       spectral component before plotting.
   * - :external+core:py:func:`~spectrochempy.trapezoid`,
       :external+core:py:func:`~spectrochempy.simpson`
     - Integrate a selected 1D peak region on the calibrated axis after
       phasing and baseline correction.

For users familiar with nmrglue-ng, the corresponding tasks are grouped
below. Compare phase and Fourier conventions before reusing numerical
parameters from an array-based workflow.

.. list-table::
   :header-rows: 1

   * - Task in nmrglue-ng
     - SpectroChemPy workflow
   * - ``proc_base`` apodization, zero filling, FFT, phase
     - Dataset methods ``em()``, ``gm()``, ``sp()``, ``zf_size()``, ``fft()``
       and ``pk()``. See :doc:`/userguide/processing_1d`.
   * - ``pipe_proc`` processing with metadata
     - Process a dataset with calibrated coordinates and history. See
       :doc:`/userguide/processing_2d` for quadrature and transform order.
   * - ``proc_bl`` baseline correction
     - Choose a baseline model with ``scp.Baseline`` or apply
       ``spectrum.snip()`` for a suitable 1D spectrum.
   * - ``peakpick``, ``integration`` and ``linesh``
     - Use ``find_peaks()`` for detection, ``trapezoid()`` or ``simpson()``
       for numerical integration, and ``scp.Optimize`` for fitting.
       See :doc:`/userguide/analysis` for 1D and 2D examples.

See :ref:`topspin-xfb-replay` for applying a recorded TopSpin recipe with
its window, phase and acquisition conventions.
