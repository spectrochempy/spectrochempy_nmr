NMR API
-------

.. _nmr-core-operations:

Core operations used in NMR
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       use the :doc:`qualified processing paths </userguide/processing_2d>`
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
   * - :external+core:py:func:`~spectrochempy.trapezoid`
     - Integrate a selected 1D peak region on the calibrated axis after
       phasing and baseline correction.

See :ref:`topspin-xfb-replay` for applying a recorded TopSpin recipe with
its window, phase and acquisition conventions.

Plugin API
~~~~~~~~~~

.. currentmodule:: spectrochempy_nmr

The recommended reader is exposed after plugin discovery as
``scp.nmr.read(...)``. It dispatches to the concrete reader functions below.
The dispatch function itself is documented in the user guide because it is a
plugin-proxy entry point rather than a module-level function.

``scp.nmr.read`` accepts TopSpin/Bruker, Agilent/Varian, JEOL and
SIMPSON inputs and dispatches to the reader functions below. It is the
recommended public entry point even though the callable is created lazily by
the plugin proxy.

.. autoclass:: Experiment
   :members:

.. automodule:: spectrochempy_nmr.readers.read_topspin
   :members: read_topspin

.. automodule:: spectrochempy_nmr.readers.read_agilent
   :members: read_agilent

.. automodule:: spectrochempy_nmr.readers.read_jeol
   :members: read_jeol

.. automodule:: spectrochempy_nmr.readers.read_simpson
   :members: read_simpson

.. automodule:: spectrochempy_nmr.units
   :members: set_nmr_context

Experimental processing
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: spectrochempy_nmr.reference_agilent_1d

.. autofunction:: spectrochempy_nmr.process_agilent_noesy_states_tppi

.. autofunction:: spectrochempy_nmr.extract_noesy_components

.. autofunction:: spectrochempy_nmr.reference_interop_1d
