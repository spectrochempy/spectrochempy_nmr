"""Fetch the public nmrML example into an explicitly selected external directory.

No data are bundled in the plugin or uploaded as CI artifacts. Source availability
does not establish a redistribution licence; see docs/agilent_qualification.rst.
"""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://nmrml.org/examples/1/FAM013_AHTM.PROTON_04.fid.zip"
SHA256 = "847590b0c2531eb026a878718249f636799b023cfa835be6b7d04314541a3f49"
ACQUISITION = "FAM013_AHTM.PROTON_04.fid"
FILES = {
    "fid": "ffdb15ed7ec30d3e01d86a8a743958ea6e165c618f791a728956bee43a06ce81",
    "procpar": "458e0230178a7642effa116ff32f005edda092f698899998300a4620096ca370",
}


def fetch(destination):
    """Verify archive and member hashes before writing only fid and procpar."""
    with urlopen(URL, timeout=60) as response:
        archive = response.read(1_000_001)
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise ValueError("nmrML archive checksum mismatch")
    with ZipFile(io.BytesIO(archive)) as bundle:
        payloads = {name: bundle.read(f"{ACQUISITION}/{name}") for name in FILES}
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != FILES[name]:
            raise ValueError(f"nmrML {name} checksum mismatch")
    acquisition = Path(destination) / ACQUISITION
    acquisition.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (acquisition / name).write_bytes(payload)
    return acquisition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(fetch(parser.parse_args().destination))
