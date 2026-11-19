"""Generate the small synthetic inputs used by the reader gallery."""

from pathlib import Path
import base64

import nmrglue as ng
import numpy as np
import spectrochempy as scp


def generate(destination):
    """Write synthetic Agilent, SIMPSON and nmrML files below a data directory."""
    root = Path(destination) / "nmrdata" / "synthetic"
    root.mkdir(parents=True, exist_ok=True)
    points, width = 1024, 4000.0
    time = np.arange(points) / width
    fid = np.exp(-time / 0.04) * np.exp(2j * np.pi * 500 * time)

    universal = ng.fileiobase.create_blank_udic(1)
    universal[0].update(size=points, sw=width, obs=400.0, car=0.0, label="1H")
    header = ng.varian.create_dic(universal)
    parameters = {
        "sw": [str(width)],
        "sfrq": ["400"],
        "tn": ["H1"],
        "nt": ["1"],
        "ct": ["1"],
        "ss": ["0"],
        "rg": ["1"],
        "rfl": ["0"],
        "rfp": ["0"],
        "seqfil": ["synthetic"],
    }
    header["procpar"].update(
        {
            key: ng.varian.create_pdic_param(key, values)
            for key, values in parameters.items()
        }
    )
    ng.varian.write(
        str(root / "agilent.fid"), header, fid.astype("complex64"), overwrite=True
    )

    rows = "".join(f"{value.real:.12g} {value.imag:.12g}\n" for value in fid)
    (root / "simpson.fid").write_text(
        f"SIMP\nNP={points}\nSW={width}\nTYPE=FID\nDATA\n{rows}END\n",
        encoding="utf-8",
    )
    encoded = base64.b64encode(fid.astype("<c16").tobytes()).decode("ascii")
    (root / "signal.nmrML").write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<nmrML xmlns="http://nmrml.org/schema" version="1.0.rc1">
  <cvList><cv id="NMR" fullName="NMR CV" URI="https://nmrml.org/cv" version="1"/></cvList>
  <acquisition><acquisition1D><acquisitionParameterSet numberOfScans="1">
    <DirectDimensionParameterSet numberOfDataPoints="{2 * points}">
      <acquisitionNucleus name="hydrogen atom"/>
      <effectiveExcitationField value="400" unitName="megaHertz"/>
      <sweepWidth value="{width}" unitName="hertz"/>
      <irradiationFrequency value="400" unitName="megaHertz"/>
      <irradiationFrequencyOffset value="0" unitName="hertz"/>
      <samplingStrategy cvRef="NMR" accession="NMR:1000349" name="uniform sampling"/>
    </DirectDimensionParameterSet>
  </acquisitionParameterSet><fidData compressed="false" byteFormat="complex128"
    encodedLength="{len(encoded)}">{encoded}</fidData></acquisition1D></acquisition>
</nmrML>
""",
        encoding="utf-8",
    )
    return root


if __name__ == "__main__":
    print(generate(scp.preferences.datadir))
