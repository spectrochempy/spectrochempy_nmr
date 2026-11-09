"""Fixtures for the installed-wheel qualification tests."""

import numpy as np
import pytest


@pytest.fixture
def paired_ser(tmp_path, request):
    """Create asymmetric stored branches and nonzero phases for both axes."""
    mode = request.param
    path = tmp_path / "paired-ser" / "1"
    proc = path / "pdata" / "1"
    proc.mkdir(parents=True)
    (path / "acqus").write_text(
        "##TITLE= Direct acquisition\n"
        "##$TD= 256\n##$SW= 16\n##$SW_h= 2000\n##$SFO1= 125\n"
        "##$BF1= 124.9995\n##$O1= 500\n##$AQ_mod= 3\n##$DECIM= 16\n"
        "##$DSPFVS= 12\n##$GRPDLY= 2.5\n##$BYTORDA= 1\n##$DTYPA= 0\n"
        "##$PARMODE= 1\n##$NUC1= <13C>\n##$NS= 16\n##$RG= 18\n"
        "##$FnTYPE= 0\n##END=\n"
    )
    (path / "acqu2s").write_text(
        "##TITLE= Indirect acquisition\n"
        "##$TD= 8\n##$SW= 8\n##$SFO1= 125\n##$BF1= 124.9995\n"
        f"##$O1= 500\n##$FnMODE= {mode}\n##$NUC1= <13C>\n##END=\n"
    )
    for name, zero, first in (("procs", 21, -8), ("proc2s", -30, 14)):
        (proc / name).write_text(
            f"##TITLE= Processing\n##$SF= 124.999\n"
            f"##$PHC0= {zero}\n##$PHC1= {first}\n##END=\n"
        )
    values = np.arange(8 * 256, dtype=np.int32)
    (path / "ser").write_bytes((values * (values % 7 - 3)).astype(">i4").tobytes())
    return path
