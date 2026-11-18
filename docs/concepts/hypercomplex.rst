Native complex channels in multidimensional NMR
================================================

A phase-sensitive 2D acquisition has two associated complex branches. They
are stored in an ordinary complex ``NDDataset`` with shape ``(2, nF1, nF2)``
and dimensions ``("c", "y", "x")``. The component axis ``c`` is not a Fourier
axis. Keep it first when reordering the physical axes.
See :doc:`../compatibility` for the core version required by native
channel workflows.

The reader supplies the component arrangement for the declared quadrature;
see the :doc:`raw 2D contract <../reference/interop>`. States-family inputs
use two complex branches; echo/anti-echo inputs retain the ordered
``first`` and ``second`` branches for Rance–Kay reconstruction.
Transform F2 (``x``) before F1 (``y``). The indirect transform reconstructs the appropriate complex pairs; its resulting
channels are labelled ``F1_R`` and ``F1_I``. Component meanings depend on the
processing stage and must not be inferred solely from the array shape.

Bruker processed 2D components use ``S_R = RR + i IR`` and
``S_I = RI + i II``. The experimental Agilent NOESY adapter returns the same
explicit mathematical component mapping; its physical acquisition convention
and calibrated frequency axes remain experimental.

For Agilent pairs with ``phase=[1,2]``, use the
:doc:`reader guide <../userguide/reading_data>` to select or verify the
indirect quadrature before transforming F1. For JEOL, see the
:doc:`2D processing guide <../userguide/processing_2d>` for the supported
transforms.

Select the channel and its real or imaginary part before plotting. For
example, the real part of the first channel in a processed Bruker
spectrum is RR.
