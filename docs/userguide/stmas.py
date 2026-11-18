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
# and nuclear spin. The explicitly selected ``Experiment.stmas_shear()``
# operation checks the observed isotope and requires spin 3/2. This DQF
# pathway was validated on ⁸⁷Rb; reading another acquisition does not qualify
# its transfer pathway for the same shear.
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
# and apply a common phase. Combine the real cosine and sine branches
# into a complex F1 signal. On the positive FFT convention of the
# example, multiply that signal by the phase ramp
# :math:`\exp[2\pi i t_1(k\nu_2-\delta I_0)]`, where
# :math:`k=-8/9`, :math:`\nu_2` is the centered F2 frequency in Hz,
# and :math:`\delta I_0` is the frequency offset for the selected
# isotropic shift. The example uses the historical shift of +0.1 ppm
# and the F1 scale divisor 17/9. Use the indirect dwell from ``IN[0]``
# and apply the F1 Fourier transform after the phase ramp.
#
# The :doc:`short STMAS workflow </examples/gallery/nmr/plot_stmas>`
# uses the common ``read()`` entry point, followed by ``xf2()``, ``xf2p()`` and
# ``stmas_shear()`` API. The shear implements the spin-3/2 DQF pathway
# and returns the complex F1 spectrum; the example plots its magnitude.
# Inspect phase and references before transferring the settings to another
# acquisition.

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
    spectrum = scp.nmr.Experiment(phased).stmas_shear(size=512, f1_shift=0.1)
    spectrum.abs()[:, -10.0:-60.0].plot(method="contour")
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
# See the :ref:`archived comparison and its plotting code <rbno3-comparison>`.
#
# See :doc:`multiplex_spam` for the phase-cycle reconstruction and
# :doc:`mqmas` for the conventional States MQMAS workflow.
