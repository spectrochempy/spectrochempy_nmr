"""Fetch hash-pinned nmrML Bruker processed data into an explicit directory.

The payload remains external to the package and is not uploaded as an artifact.
Public availability is not a redistribution licence. Existing mismatched data
are never overwritten.
"""

import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://nmrml.org/examples/2/MMBBI_10M12-CE01-1a.zip"
SHA256 = "06d553588a7df7a981ba23d624a9ba4f5c5bcc0435a400cb58ec858850172c3f"
ACQUISITION = "MMBBI_10M12-CE01-1a/1"
FILES = {
    "acqus": "363019f0bc4e205b5c456a7c6e609689abc12d04243a7b25cc512dfb4d97fa50",
    "pdata/1/1r": "c188b3bfea49bb764ef1f30316e82d8f08db7c3477937ab43fa48d0833e236b7",
    "pdata/1/1i": "ca0d589493f0346e113b091eb46c7562adc268afdeadcf7a3b13838a437ca3bd",
    "pdata/1/procs": "0f9ee607f574b6d355b44eb0ec7c2ff2981272385e576f557cf6d19891c2bc39",
}


def fetch(destination):
    """Validate archive and member hashes before writing the selected files."""
    with urlopen(URL, timeout=60) as response:
        archive = response.read(1_000_001)
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise ValueError("nmrML pdata archive checksum mismatch")
    with ZipFile(io.BytesIO(archive)) as bundle:
        payloads = {name: bundle.read(f"{ACQUISITION}/{name}") for name in FILES}
    root = Path(destination) / ACQUISITION
    for name, payload in payloads.items():
        if hashlib.sha256(payload).hexdigest() != FILES[name]:
            raise ValueError(f"nmrML pdata {name} checksum mismatch")
        target = root / name
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() != FILES[name]
        ):
            raise ValueError(f"refusing to overwrite existing pdata file {target}")
    for name, payload in payloads.items():
        target = root / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
    return root / "pdata" / "1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(fetch(parser.parse_args().destination))
