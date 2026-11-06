"""Fetch checksum-pinned official nmrML examples for local/CI qualification.

No experimental data are packaged or uploaded. Public download and accepted
local validation do not establish redistribution rights. Only original native
FID/procpar members already used by Agilent qualification are extracted.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen

from fetch_agilent_qualification import fetch as fetch_native

SOURCES = {
    "hop.nmrML": (
        "https://nmrml.org/examples/1/FAM013_AHTM.PROTON_04.nmrML",
        "2f8a0b7cdc4ca8083aecb2df1ab6d81fbede5891ee2c3b562cc8dd079f8837d0",
    ),
    "biosample_quantification.nmrML": (
        "https://nmrml.org/examples/4/biosample_quantification.nmrML",
        "38712b01526635b3bf2639a5b08c7e00e15c4e56d0c1174295595b9ab939c448",
    ),
}


def fetch(destination):
    payloads = {}
    for name, (url, checksum) in SOURCES.items():
        with urlopen(url, timeout=60) as stream:
            payload = stream.read(2_000_001)
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError("nmrML example checksum mismatch: " + name)
        payloads[name] = payload
    destination = Path(destination)
    original = fetch_native(destination)
    target = destination / "hop-native"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("fid", "procpar"):
        shutil.copyfile(original / name, target / name)
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    print(fetch(parser.parse_args().destination))
