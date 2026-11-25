Diffusion measurements
=======================

Use ``scp.nmr.Diffusion`` for a PFG measurement or an experimental DOSY map. Follow
:doc:`/userguide/diffusion_pfg` for the first figure; processing conventions
and qualifications are in :doc:`/userguide/diffusion_details`.
Continue with :doc:`/userguide/diffusion_dosy` for the map.

For installation, see :doc:`/getting_started/installation`.

Prepared spectra
-----------------

Already processed series can be passed directly with calibrated diffusion
weights. Rows must correspond to the supplied b values, with at least four
distinct weights, and columns to a uniform ppm axis. The spectra must be
real, phased, finite and unmasked. Supply b as a ``scp.Coord`` or a quantity
with units convertible to s/m²; plain numbers without units are refused::

    weights = scp.Coord(b_values, units="s/m^2", rounding=False)
    diffusion = scp.nmr.Diffusion(spectra, b=weights)
    result = diffusion.fit(region=(low_ppm, high_ppm))

``process()`` is unnecessary for prepared spectra. The caller is responsible
for their common phase, relative intensity scaling and calibrated b axis.

.. currentmodule:: spectrochempy_nmr

.. autoclass:: Diffusion
   :members: b, process, fit, dosy

.. autoclass:: DiffusionFit
   :members: plot
   :exclude-members: D, I0, areas, fitted, residual, rmse, provenance, diagnostics

.. autoclass:: DOSYResult
   :members: plot
