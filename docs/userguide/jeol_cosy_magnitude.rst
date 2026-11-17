JEOL COSY in magnitude mode
===========================

Use :func:`spectrochempy_nmr.process_jeol_cosy_magnitude` to reconstruct a
uniform proton COSY from a raw JEOL JDF acquired with ``cosy.jxp`` or
``cosy_pfg.jxp``. The recipe applies squared sine windows, Fourier transforms
both axes, and takes the magnitude. It symmetrizes the spectrum by taking the
minimum of the amplitudes at each pair of exchanged frequencies.

Download the raw `luteolin COSY (nmrXiv D1155)
<https://doi.org/10.57992/nmrxiv.p33.s214.d1155>`_ and use the path to the JDF
in the example.

.. code-block:: python

   from spectrochempy_nmr import process_jeol_cosy_magnitude

   path = "Luteolin_3120ug200uL_DMSOd6_COSY_400MHz_Jeol.jdf"
   cosy = process_jeol_cosy_magnitude(path)
   levels = [0.005 * cosy.max() * 2**i for i in range(8)]
   _ = cosy.plot(method="contour", levels=levels)

The result has F1 and F2 coordinates in Hz. Set ``f1_size`` to zero-fill F1
to a chosen length. By default F1 retains its acquired size and F2 is
zero-filled to the next power of two.

On luteolin the recipe recovers the principal cross-peaks in the
`processed reference (nmrXiv D1152)
<https://doi.org/10.57992/nmrxiv.p33.s214.d1152>`_. Relative intensities
can differ from those in the export. This experimental recipe records the
physical F1 acquisition mode as indeterminate.

To compare directly with the export, evaluate the recipe on the export's
frequency grid. The interpolation then precedes the symmetric minimum.

.. code-block:: python

   from spectrochempy_nmr import read

   reference = read(
       "Luteolin_3120ug200uL_DMSOd6_COSY_400MHz_JDX.jdx",
       protocol="interop",
       profile="jcampdx-spectrum-2d",
   )
   comparison = process_jeol_cosy_magnitude(
       path, frequency_grid=(reference.y.data, reference.x.data)
   )
