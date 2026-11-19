Processing a 2D QF spectrum
===========================

Read the acquisition with ``scp.nmr.read()``, then use ``Experiment`` to
process the normalized QF data. QF acquires one complex F2 trace for each
F1 increment, without a cosine/sine pair. For a supported, uniformly sampled
acquisition, the windowing, two-dimensional Fourier transform and magnitude
operations depend on that representation and its physical coordinates.
They are independent of the instrument manufacturer and experiment name.

The worked example uses a public epicatechin COSY acquisition. Its JEOL
origin matters at the reading step; subsequent processing uses the common
QF representation. Minimum symmetrization is an optional additional operation
for a homonuclear correlation map such as this COSY.

Read the complex FIDs
---------------------

This example reads the `epicatechin COSY JDF (nmrXiv D875)
<https://doi.org/10.57992/nmrxiv.p33.s192.d875>`_. For this uniform two-section
JDF, explicitly declare ``quadrature="qf"`` from the acquisition convention.
Other supported readers supply the mode from their acquisition metadata;
use their documented reading options. See :doc:`reading_data`.

.. code-block:: python

   import spectrochempy as scp

   path = scp.preferences.datadir / "nmrdata/jeol/epicatechin_cosy.jdf"
   raw = scp.nmr.read(path, quadrature="qf")

The returned dataset has F1 and F2 time coordinates in seconds. Each indirect
increment contains one complex F2 trace. Reading arranges the stored components
and retains the acquired sizes, widths, observation frequencies and carrier
references. It performs neither an FFT nor apodization.

Window and transform both axes
------------------------------

A squared sine window suppresses discontinuities at the ends of each acquired
FID. Apply it before zero filling, so the window is defined on the acquired
points rather than on the enlarged FFT grid.

.. code-block:: python

   weighted = raw.sp(ssb=0, pow=2, dim="x").sp(ssb=0, pow=2, dim="y")
   spectrum = scp.nmr.Experiment(weighted).process(size=(2048, 2048))

``size`` gives the F1 and F2 FFT lengths. Each must retain every acquired point;
without ``size``, both transforms retain their acquired lengths. The processor
uses the same QF Fourier convention as other normalized QF inputs. It consumes
the complex samples, time grids and physical descriptors, without reopening
the source file or consulting its manufacturer or experiment name.

The complex result retains phase information under the declared QF convention.
Magnitude is a separate operation:

.. code-block:: python

   magnitude = abs(spectrum)
   view = magnitude.copy()
   view.x.ito("ppm")
   view.y.ito("ppm")
   aromatic = view[8.0:5.5, 8.0:5.5]
   plot_options = dict(
       method="contour", levels=[0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
       xlim=(8.0, 5.5), ylim=(8.0, 5.5), xlabel="F2 (ppm)", ylabel="F1 (ppm)",
   )
   _ = (aromatic / aromatic.max()).plot(**plot_options)

.. figure:: /_static/jeol-common-magnitude-aromatic.png
   :alt: Epicatechin magnitude in the same aromatic ppm window as the comparison below

   Epicatechin magnitude before symmetrization, displayed over 5.5–8 ppm.
   This view and the two maps below use identical axes and contour thresholds,
   on the same intensity scale relative to the unsymmetrized maximum.

Frequency coordinates in this example are in Hz. The reader supplies the
physical calibration, which remains fixed when the FFT size changes. The
coordinates retain the observation frequency and can be converted to ppm.

Minimum symmetrization
----------------------

For a homonuclear correlation map, the symmetric minimum suppresses a feature
that has no reciprocal partner. It changes amplitudes and is therefore a
separate, explicit treatment after magnitude. Apply it only when reciprocal
peaks are expected at the same physical frequencies on both axes; it is not a
general requirement of QF processing:

.. code-block:: python

   symmetric = scp.nmr.Experiment(magnitude).symmetrize()
   symmetric.x.ito("ppm")
   symmetric.y.ito("ppm")
   symmetric_region = symmetric[8.0:5.5, 8.0:5.5]
   _ = scp.multiplot(
       [aromatic / aromatic.max(), symmetric_region / aromatic.max()],
       ncol=2, labels=["QF magnitude", "Minimum symmetrization"],
       figsize=(9, 4), **plot_options,
   )

The operation evaluates :math:`\min(S(F_1,F_2),S(F_2,F_1))` at exchanged
physical frequencies. It interpolates between coordinates when the axes have
different sampling grids; it does not transpose array indices and assume the
frequencies match. An exchanged coordinate outside the source grid contributes
zero. Keep the unsymmetrized magnitude for comparison and quantitative analysis.

.. figure:: /_static/jeol-common-epicatechin-aromatic.png
   :alt: Epicatechin QF magnitude before and after minimum symmetrization

   The left map repeats the magnitude shown above; the right map adds minimum
   symmetrization. The window, axes, contour thresholds and amplitude scale
   are the same in all three maps. Only the symmetrization changes the data.

Save and reuse
---------------

A native SCP save retains the normalized components and physical references.
Reload it through ``scp.nmr.read()`` and continue with ``Experiment``. The
processing does not require the original acquisition file or the provider library after
reading.

Two short gallery examples cover the
:doc:`complete raw-data workflow </examples/gallery/nmr/plot_qf_2d>` and
:doc:`reading an already processed spectrum </examples/gallery/importer/plot_read_jeol_processed_2d>`.
