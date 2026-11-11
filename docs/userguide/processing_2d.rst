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
the recorded correction for supported acquisitions. The echo/anti-echo
workflow also offers explicit time-domain and phase correction. To reproduce
a direct FFT script that relied on correction during import, pass
``remove_digital_filter=True`` to the reader.

Here ``x`` and ``y`` are the direct and indirect dimensions of the
imported FID. Keep both quadrature branches through the indirect FFT.
Select the desired spectral component before plotting or extracting a
1D slice. The :doc:`JEOL COSY example </examples/gallery/nmr/plot_processing_nmr_2d>`
shows a mixed-domain result for an acquisition that supports the F2 FFT.
Use :doc:`analysis` to find and integrate cross-peaks in a processed spectrum.

When a TopSpin read includes a ``procs`` file, the dataset can expose its
processing parameters in ``dataset.meta.nmr_processing["vendor_profile"]``.
The profile records the vendor, the source file, and the parameter values.
Replay requires a supported combination of acquisition and processing
settings, as described in :ref:`topspin-xfb-replay`.

For a supported uniform Bruker SER acquisition, use the recorded TopSpin
processing parameters explicitly:

.. code-block:: python

   import spectrochempy as scp

   fid = scp.read_topspin("path/to/experiment/ser")
   spectrum = scp.nmr.Experiment(fid).process(replay="topspin")
   trace = spectrum.meta.nmr_processing["scp_processing"]["applied"]

XFB replay supports indirect QF magnitude, States-TPPI, and echo/anti-echo
with direct DQD. See :ref:`topspin-xfb-replay` for the supported settings
and reference comparisons. The ordinary ``Experiment.process()`` path
applies SpectroChemPy's processing sequence to supported acquisitions.

For the underlying dataset, plotting, and general processing API, see
the `NMR plugin overview in the SpectroChemPy core documentation
<https://www.spectrochempy.fr/latest/userguide/plugins/nmr.html>`_.
