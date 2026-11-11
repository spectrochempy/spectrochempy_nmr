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

A direct FFT uses the supplied samples, including the original group delay
when reading a Bruker SER with default options. TopSpin XFB replay applies
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

   fid = scp.nmr.read_topspin("path/to/experiment/ser")
   spectrum = scp.nmr.Experiment(fid).process(replay="topspin")

XFB replay supports indirect QF magnitude, States-TPPI, and echo/anti-echo
with direct DQD. Use the raw SER from the same experiment as the recorded TopSpin processing
settings. The :ref:`XFB qualification notes <topspin-xfb-replay>` list the
supported settings. The ordinary ``Experiment.process()`` path
applies SpectroChemPy's processing sequence to supported acquisitions.

For the underlying dataset, plotting, and general processing API, see
the `NMR plugin overview in the SpectroChemPy core documentation
<https://www.spectrochempy.fr/latest/userguide/plugins/nmr.html>`_.
