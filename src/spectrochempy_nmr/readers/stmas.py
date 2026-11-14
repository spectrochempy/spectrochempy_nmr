"""Read uniform Bruker States STMAS acquisitions for staged processing."""

from numbers import Integral
from pathlib import Path

import numpy as np

from .. import _check_nmrglue_installation
from .states_fids import states_fids


def read_stmas(path, *, procno=1, use_tdeff=False):
    """Read a 2D Bruker States STMAS experiment or its SER.

    Set use_tdeff=True to use the number of traces in the processing
    status (proc2s TDeff), for example to omit a final unpaired trace.
    Without that option, all declared acquisition traces must form pairs.
    """
    _check_nmrglue_installation()
    from nmrglue import interop

    if not isinstance(use_tdeff, bool):
        raise TypeError("use_tdeff must be a boolean")
    if isinstance(procno, bool) or not isinstance(procno, Integral) or procno < 1:
        raise ValueError("procno must be a positive integer")
    directory = Path(path).resolve()
    if directory.name == "ser":
        directory = directory.parent
    pdata = directory / "pdata" / str(procno)
    files = [pdata / name for name in ("procs", "proc2s") if (pdata / name).is_file()]
    if pdata / "procs" not in files or (use_tdeff and pdata / "proc2s" not in files):
        raise FileNotFoundError(f"Missing STMAS processing status files in {pdata}")
    stored = interop.read_bruker_ser_storage(directory, procs_files=files)
    direct, indirect = stored.metadata["acqus"], stored.metadata.get("acqu2s", {})
    if (
        direct.get("PARMODE") != 1
        or direct.get("FnTYPE", 0) != 0
        or direct.get("AQ_mod") not in (1, 3)
        or indirect.get("FnMODE") != 4
        or "nuslist_raw" in stored.metadata
        or stored.dimensions.get("higher_dimension_files")
    ):
        raise ValueError("STMAS requires a uniform 2D Bruker States acquisition")
    declared = indirect.get("TD")
    words = stored.dimensions["declared_td"]
    count = stored.metadata["proc2s"].get("TDeff") if use_tdeff else declared
    if (
        not isinstance(declared, Integral)
        or not isinstance(count, Integral)
        or not 4 <= count <= declared <= stored.data.shape[0]
        or count % 2
        or words < 4
        or words % 2
    ):
        raise ValueError("STMAS requires complete States pairs within the acquired TD")
    traces = stored.data[:count, :words]
    branches = np.empty((2, count // 2, words // 2), dtype=np.complex128)
    for branch in (0, 1):
        branches[branch].real = traces[branch::2, ::2]
        branches[branch].imag = traces[branch::2, 1::2]
    return states_fids(
        stored,
        branches,
        kind="stmas",
        selection={
            "retained_traces": count,
            "declared_traces": declared,
            "stored_traces": stored.data.shape[0],
            "used_tdeff": use_tdeff,
        },
        operation="pair-stmas-states-traces",
        scale=1.0,
        sign="none",
        history=f"Read {count // 2} complete STMAS States pairs",
    )
