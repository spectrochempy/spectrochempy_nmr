Native complex channels in multidimensional NMR
================================================

.. important::

   This representation is available in NMR 0.3.0. Published 0.2.0 used quaternion
   storage. The development plugin no longer uses that storage or requires
   the Hypercomplex plugin.

A phase-sensitive 2D acquisition has two associated complex branches. They
are stored in an ordinary complex ``NDDataset`` with shape ``(2, nF1, nF2)``
and dimensions ``("c", "y", "x")``. The component axis ``c`` is not a Fourier
axis. Keep it first when reordering the physical axes.
The NOESY adapter restores physical axes by their names before processing and
requires the component coordinate to retain its declared branch order.

Native channel workflows and unvalidated indirect acquisition profiles require
a core exposing ``FFT_PREVALIDATION_API >= 1``. The published cores 1.1.0 and
1.1.4 do not provide it, so the affected readers and processing helpers refuse
these workflows before returning native data. Ordinary 1D support remains
available on those cores. The core and plugin changes must be deployed together.

Unsupported inverse, component-axis and indirect transforms are rejected by
the read-only ``fft.validate_input`` hook before any axis permutation, resizing
or sample change, including when ``inplace=True`` is requested.

Raw Bruker States, States-TPPI and Echo-Antiecho trace pairs use channel labels
``branch_0`` and ``branch_1``. Transform F2 (``x``) before F1 (``y``). The
indirect transform reconstructs the appropriate complex pairs; its resulting
channels are labelled ``F1_R`` and ``F1_I``. Component meanings depend on the
processing stage and must not be inferred solely from the array shape.

Bruker processed 2D components use ``S_R = RR + i IR`` and
``S_I = RI + i II``. The experimental Agilent NOESY adapter returns the same
explicit mathematical component mapping; its physical acquisition convention
and calibrated frequency axes remain experimental.

Agilent phase=1,2 trace pairs are decoded without changing their complex
samples. This storage decoding does not establish States-TPPI normalization.
The generic indirect FFT therefore rejects them. JEOL indirect complex data
are retained in their decoded matrix layout with an unvalidated encoding;
indirect processing is rejected until its storage profile is established.

The former ``dataset.hyper`` accessor is not used. Select channels explicitly
or use ``scp.nmr.extract_noesy_components`` for the NOESY result. The ordinary
core complex operations and native dataset persistence handle the arrays.
