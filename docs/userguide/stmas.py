# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Processing a States STMAS acquisition

# %% [raw] raw_mimetype="text/restructuredtext"
# Satellite-transition magic-angle spinning (STMAS) correlates satellite
# evolution during the indirect period with central-transition detection in
# F2. A shear removes the correlated second-order quadrupolar contribution
# from the indirect coordinate. Its coefficient and isotropic scale depend
# on the satellite pathway and spin :cite:p:`fernandez2011quadrupolar`.
#
# Acquisition and States pairs
# -----------------------------
#
# The spin-3/2 ⁸⁷Rb example was acquired with ``dqf-stmas-old.av``.
# The reader decodes States pairs independently of the pulse-program name
# and nuclear spin. Select ``experiment="STMAS"`` in ``xfshear()`` to derive
# the inner-satellite coefficients from the observed spin. The archived
# example uses ⁸⁷Rb; ``spin`` need only be supplied if metadata and isotope
# lookup cannot establish it.
# The double-quantum filter selects the transfer pathway, and States
# phase cycling records two adjacent FIDs for each indirect increment.
# The archived experiment RbNO₃/15 has 111 acquired traces in 128 stored
# slots. Its processing status gives ``TDeff = 110``. Pass ``use_tdeff=True`` to
# ``scp.nmr.read()`` to select those 55 complete pairs. The last
# acquired trace has no partner and the
# remaining slots are padding. Keep the acquisition status, pulse program
# and processing status together to distinguish these counts.
#
# Pass ``indirect_dwell="IN[0]"`` to calibrate F1 with the acquisition
# increment used in this dataset. A positive number supplies the time step
# directly in seconds. With ``use_tdeff=True`` and no dwell override, the
# reader uses the indirect acquisition spectral width. These are physical
# calibration choices, not experiment selectors. This explicit States read
# keeps uncorrected samples without scan/gain scaling; corrections belong
# in the subsequent processing. The selected counts, step, frequency
# references and isotope are retained in the common acquisition metadata.
# ``read_stmas()`` remains a compatibility shortcut for the ``IN[0]`` read.
#
# Transform, shear and reference
# ------------------------------
#
# Correct the direct digital-filter delay, transform F2 for each branch,
# and apply a common phase. States-TPPI inputs first have their alternating
# F1 sign decoded by ``xf2()``; States inputs already have that arrangement.
# Echo/anti-echo uses the existing Rance–Kay conversion to that same States
# arrangement. All three then use the same reconstruction and shear,
# independently of whether the coefficients describe MQMAS, STMAS or ZQ.
# Their equivalence is tested on synthetic data. Qualification of the
# XFSHEAR route on a real echo/anti-echo acquisition remains open.
#
# In the mixed domain, the common shear multiplies the complex F1 signal by
# :math:`\exp[2\pi i t_1(k\nu_2+\Delta\nu)]`, where :math:`\nu_2` is the
# centered F2 frequency in Hz. For the spin-3/2 inner-satellite pathway use
# ``experiment="STMAS"`` computes ratio -8/9 and scale 17/9 from ⁸⁷Rb.
# The scale divisor calibrates the
# isotropic F1 axis; a shear ratio alone does not determine that calibration.
# The common ``f1_shift`` is in isotropic ppm and gives
# :math:`\Delta\nu=\delta_{F1}\nu_0\,\mathrm{f1\_scale}`.
#
# ``xfshear()`` reconstructs the F2 quadrature by Hilbert transformation
# with zero extension to twice the F2 FFT size, applies half weighting to
# the first F1 point, then transforms F1. The central RR point uses the
# same correction as the MQMAS workflow. The result retains all four
# spectral components in two complex channels. Extract ``RR``, ``RI``,
# ``IR`` and ``II`` through ``Experiment``; plot a real component or form a
# magnitude view explicitly. This replaces the former STMAS-only complex
# F1 map and its unweighted first point.
#
# ``stmas_shear()`` is a convenience shortcut to this complete shared
# reconstruction with automatic coefficients and isotope checks. Its historical shift
# direction is retained: ``stmas_shear(f1_shift=0.1)`` corresponds to
# ``xfshear(experiment="STMAS", f1_shift=-0.1)``. This guide uses
# the latter form so the processing follows the same API as MQMAS.
#
# See the :doc:`short STMAS workflow </examples/gallery/nmr/plot_stmas>`.

# %%
import os
from cmath import phase as phase_angle
from math import degrees
from pathlib import Path

import spectrochempy as scp


root = os.environ.get("SCP_RBNO3_DATA")
if root:
    states = scp.nmr.read(Path(root) / "15", use_tdeff=True, indirect_dwell="IN[0]")
    xf2 = scp.nmr.Experiment(states).xf2(size=1024)
    peak = xf2.data[abs(xf2).argmax()]
    phased = scp.nmr.Experiment(xf2).xf2p(phc0=-degrees(phase_angle(peak)))
    spectrum = scp.nmr.Experiment(phased).xfshear(
        size=512, experiment="STMAS", f1_shift=-0.1
    )
    view = scp.nmr.Experiment(spectrum)
    _ = abs(view.RR + 1j * view.RI)[:, -10.0:-60.0].plot(method="contour")
else:
    print("Set SCP_RBNO3_DATA to the directory containing RbNO₃/15.")

# %% [raw] raw_mimetype="text/restructuredtext"
# Comparison with MQMAS
# ---------------------
#
# The archived comparison puts ²⁷Al PST-5 conventional MQMAS alongside
# two separate RbNO₃ acquisitions, Multiplex 3Q MQMAS and STMAS. Each
# panel has its own scale and contour levels are fractions of its
# maximum. The RbNO₃ panels show magnitude and PST-5 shows real RR
# contours, so the figure compares processing outcomes without measuring
# relative sensitivity.
#
# The archived STMAS panel predates the shared four-component reconstruction.
# See the :ref:`archived comparison and its plotting code <rbno3-comparison>`.
#
# See :doc:`multiplex_spam` for the phase-cycle reconstruction and
# :doc:`mqmas` for the conventional States MQMAS workflow.
