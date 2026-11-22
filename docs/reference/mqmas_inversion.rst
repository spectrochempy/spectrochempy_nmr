MQMAS inversion
~~~~~~~~~~~~~~~

Access the module through ``scp.nmr.mqmas_inversion``. See
:doc:`../userguide/mqmas_inversion_synthetic` for the synthetic workflow.

The inversion accepts phased, real ``NDDataset`` spectra with calibrated
ppm coordinates and optional masks. The MQMAS input is two-dimensional
and the optional MAS input is one-dimensional. Kernel matrices preserve
the order of the spectral axes, including decreasing ppm axes.

The parameter grids are increasing, uniformly spaced vectors with at
least three points each. Supply isotropic shifts in ppm and nonnegative
quadrupolar products in MHz. Population maps store shift on ``y`` and
quadrupolar product on ``x``. Their weights use the MQMAS peak as the
intensity reference. Local population constraints sum cell weights on
this scale. A zero constraint covering every supported cell currently
raises ``ValueError``.

.. automodule:: spectrochempy_nmr.mqmas_inversion
   :members: Kernel, build_kernel, FitResult, fit, ProfilePoint, profile

.. autofunction:: spectrochempy_nmr.mqmas_inversion.mqmas_response

.. autofunction:: spectrochempy_nmr.mqmas_inversion.z_filtered_3qmas_response

.. autofunction:: spectrochempy_nmr.mqmas_inversion.mas_response
