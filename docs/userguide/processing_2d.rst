Processing 2D NMR data
======================

Use the plugin's :doc:`reader <reading_data>` to load the 2D dataset.
The acquisition and quadrature determine which operations are available.
See :doc:`/concepts/quadrature` for the supported conventions and
:ref:`nmr-processing-validation` for the distinction between runtime checks
and qualification against reference data.

Process the direct dimension (F2) before the indirect dimension (F1) for
supported paired acquisitions. The plugin uses the acquisition metadata to
select the quadrature convention and check the transform order. When
processing explicitly on a supported dataset, you can inspect the
mixed-domain result after F2 before transforming F1.

.. code-block:: python

   f2 = fid.fft(dim="x")
   spectrum = f2.fft(dim="y")

See the
:external+core:doc:`Fourier transformation guide <userguide/processing/fourier>`
for the core FFT and phase correction operations.

A direct FFT uses the supplied samples, including the original group delay
when reading a Bruker SER without import correction. TopSpin XFB replay applies
the recorded correction for supported acquisitions. For echo/anti-echo,
``Experiment.process(echo_antiecho="bruker-rance-kay", digital_filter="phase")``
compensates the digital-filter delay in F2. Use ``digital_filter="time"`` for
time-domain correction instead. The ordinary QF, States, and States-TPPI
FFT workflows transform the supplied traces without filter correction.
For these workflows, or for a direct FFT script that used the former
import correction, pass ``remove_digital_filter=True`` to the reader.

Here ``x`` and ``y`` are the direct and indirect dimensions of the
imported FID. Keep both quadrature branches through the indirect FFT.
Select the desired spectral component before plotting or extracting a
1D slice. The :doc:`JEOL COSY example </examples/gallery/nmr/plot_processing_nmr_2d>`
shows a mixed-domain result for an acquisition that supports the F2 FFT.
Use :doc:`analysis` to find and integrate cross-peaks in a processed spectrum.

For a supported uniform Bruker SER acquisition, use the recorded TopSpin
processing parameters explicitly:

.. code-block:: python

   import spectrochempy as scp

   fid = scp.nmr.read_topspin(
       "path/to/experiment/ser", remove_digital_filter=False, remove_dc_offset=False
   )
   spectrum = scp.nmr.Experiment(fid).process(replay="topspin")

XFB replay supports indirect QF magnitude, States-TPPI, and echo/anti-echo
with direct DQD. Use the raw SER from the same experiment as the recorded TopSpin processing
settings. The :ref:`XFB qualification notes <topspin-xfb-replay>` list the
supported settings. The ordinary ``Experiment.process()`` path
applies SpectroChemPy's processing sequence to supported acquisitions.

For the underlying dataset, plotting, and general processing API, see
the `NMR plugin overview in the SpectroChemPy core documentation
<https://www.spectrochempy.fr/latest/userguide/plugins/nmr.html>`_.

.. _topspin-slice-and-baseline:

TopSpin-style slice and baseline commands
-----------------------------------------

Wrap a single dataset in :class:`~spectrochempy_nmr.Experiment`. These
commands return new datasets in memory. They use TopSpin numbering, so row
or column **1** corresponds to index **0** in SpectroChemPy.

.. code-block:: python

   fid = scp.nmr.Experiment(raw).rser(1)
   experiment = scp.nmr.Experiment(spectrum)
   row = experiment.rsr(1)
   column = experiment.rsc(1)

For a plain F1 × F2 matrix, ``rser(1)`` and ``rsr(1)`` select
``dataset[0, :].squeeze("y")``. For a real spectrum, ``rsc(1)`` selects
``dataset[:, 0].squeeze("x")``. In a paired acquisition, ``rser()`` numbers the
two stored traces of each F1 increment consecutively. With four spectral
components, ``rsr()`` keeps the complex F2 trace from the F1-real channel, and
``rsc()`` builds a complex F1 trace from the F2-real components. After an
F2-only transform, ``rsc()`` returns an F1 time trace with the acquisition
convention still encoded.

Use ``abs2()`` to fit a polynomial separately along each F2 row and
``abs1()`` to fit along each F1 column of a fully transformed spectrum.
Both correct only the real-real plane, as in TopSpin. The default polynomial
degree is 5. Set ``limits`` to correct a specific interval and ``ranges`` to
identify peak-free regions for the fit. Both take values in the units of
the selected axis. By default the fit iteratively excludes positive and negative peaks
using each trace's residual noise estimate.

.. code-block:: python

   corrected = experiment.abs2(
       order=1, limits=(20.0, 65.0), ranges=[(21.0, 24.0), (61.0, 64.0)]
   )
   corrected = scp.nmr.Experiment(corrected).abs1(order=1)

The fit excludes masked points, including when fit ranges are supplied.
A correction interval requires a strictly monotonic axis. Automatic fit-point
selection can differ from TopSpin's peak-region selection.

Agilent TPPI
------------

For a native Agilent acquisition identified as TPPI, use
``Experiment(fid).process()``. The result contains four planes in F1, F2
component order (``rr``, ``ri``, ``ir``, ``ii``). The direct transform
uses the sample convention of ``var2pipe -noaswap`` followed by
``FT -nodf``. In F1 the real and imaginary parts of F2 each form a
real sequence. Each sequence is zero-filled from N to 2N, transformed,
and reduced to bins 0 through N−1. This retains DC and excludes Nyquist.
The F1 bin spacing is the acquisition width divided by 2N. The displayed
half-band follows the carrier convention of NMRPipe
``TP → ZF -size 2N → FT -real``, with the carrier halfway between its
two central bins.

The default axes are carrier-relative Hz. To obtain ppm, pass the
observation frequency in MHz and carrier in ppm explicitly for each
dimension, in F1 then F2 order. For the historical Agilent
``agilent_2d_tppi`` NMRPipe conversion the values are:

.. code-block:: python

   spectrum = scp.nmr.Experiment(fid).process(
       tppi_reference=((125.681, 101.274), (125.681, 101.274))
   )

Use the values from the reference for your acquisition. The Agilent
reader can lack the F1 observation frequency.
