# NMRPipe QF oracle regeneration

`regenerate_qf_oracle.py` is an external-software helper for regenerating the
small QF checkpoints used by the NMRPipe validation campaign. It is not part
of the plugin runtime and is not executed by standard CI.

The input must be the **post-`EXT` synthetic QF input** (`input.fid`) described
by the committed oracle manifest. The helper does not generate the synthetic
signal, construct a `bruk2pipe` header, or perform `EXT`; those are controlled
provenance steps used when producing the committed fixture. It writes four
NMRPipe checkpoints:

```text
f2.ft2          direct-dimension zero-fill + FT -nodf
f2tp.ft2        explicit transposition
f1.ft2          indirect zero-fill + FT -neg
magnitude.ft2   MC magnitude + final transposition
```

Example:

```bash
python tools/nmrpipe/regenerate_qf_oracle.py \
    /path/to/synthetic-input.fid /tmp/nmrpipe-qf-oracle \
    --nmrpipe /home/christian/pipe/nmrbin.linux239_64/nmrPipe
```

The generated files remain external evidence. Do not commit experimental
corpora or generated binary outputs without a separate provenance and
redistribution decision. The committed synthetic fixture is packaged
separately as `tests/data/nmrpipe/qf_oracle.npz`; this helper intentionally
does not overwrite that package or its manifest.

## States-TPPI oracle

`regenerate_states_tppi_oracle.py` generates a deterministic four-component
States-TPPI signal and runs the manual NMRPipe 13.0 chain:

```text
simTimeND header profile with yMODE States-TPPI
ZF -zf 1 -auto | FT -nodf
TP
ZF -zf 1 -auto | FT -alt
```

Example:

```bash
PYTHONPATH=/path/to/nmrglue-ng python \
  tools/nmrpipe/regenerate_states_tppi_oracle.py \
  /tmp/states-tppi-oracle \
  --nmrpipe /home/christian/pipe/nmrbin.linux239_64/nmrPipe
```

The tool records NMRPipe storage checkpoints and a canonical component decode.
The final NMRPipe interleaving is explicitly decoded as `[q0, q2, q1, q3]`,
followed by conjugation, transpose, and the direct-dimension `reverse+roll`
convention established by the QF campaign. No magnitude is used by the
States-TPPI test. NMRPipe and the external nmrglue reader are manual-generation
dependencies only; neither is required by runtime or CI.

## Echo/anti-echo oracle

`regenerate_echo_antiecho_oracle.py` feeds asymmetric deterministic integer
traces to NMRPipe 13.0. The pipeline uses `bruk2pipe -AMX -swap -yMODE
Echo-AntiEcho` for the Rance–Kay shuffling and `FT -nodf | TP | FT -nodf` for
the two unpadded transforms. It reads the output interleaving in F1/F2 order
and records the carrier-relative Hz axes. The checked-in NPZ contains only
the synthetic input and NMRPipe checkpoints.

```bash
python tools/nmrpipe/regenerate_echo_antiecho_oracle.py \
    /tmp/echo_antiecho_oracle.npz --nmrpipe-bin /path/to/pipe/bin
```

Use `--hsqc-ser /path/to/ser` with the SHA-1-checked official BMRB SER to
generate the experimental oracle outside the repository. The AMX conversion
preserves the raw point count and omits group-delay correction for this
FFT-only comparison.

Add `--digital-filter` to build the `FT -df` checkpoints for the HSQC
acquisition parameters (`DECIM=24`, `DSPFVS=12`). The conversion still
uses AMX to preserve all stored samples, but writes the group delay into
the header. Only the F2 transform uses `-df`. The same flag without
`--hsqc-ser` generates the corrected synthetic CI oracle.
