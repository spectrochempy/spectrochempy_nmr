"""Compatibility shortcut for reading uniformly sampled States components."""


def read_stmas(path, *, procno=1, use_tdeff=False):
    """Read States components using the acquisition increment ``IN[0]``.

    Compatibility shortcut for ``scp.nmr.read(path, indirect_dwell="IN[0]",
    use_tdeff=use_tdeff, procno=str(procno))``. No pulse-program or nuclear-spin
    selection is performed while reading. Choose the spin-3/2 DQF pathway
    explicitly with ``Experiment.stmas_shear()`` after XF2 and common F2 phase.
    """
    from numbers import Integral
    from pathlib import Path

    from .read_topspin import read_topspin

    if isinstance(procno, bool) or not isinstance(procno, Integral) or procno < 1:
        raise ValueError("procno must be a positive integer")
    path = Path(path)
    if path.is_dir():
        path = path / "ser"
    return read_topspin(
        path, procno=str(procno), use_tdeff=use_tdeff, indirect_dwell="IN[0]"
    )
