"""Fetch and verify only the licensed NMR fixtures used by the gallery."""

from __future__ import annotations

import hashlib
import os
import tomllib
from pathlib import Path
from urllib.request import urlopen

import spectrochempy as scp


ROOT = Path(__file__).parent
MANIFEST = ROOT / "gallery-fixtures.toml"


def main() -> None:
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    base_url = (
        "https://raw.githubusercontent.com/spectrochempy/spectrochempy_data/"
        f"{manifest['source_commit']}/{manifest['source_root']}"
    )
    destination = Path(os.environ.get("SCP_GALLERY_DATADIR", scp.preferences.datadir))
    total = 0
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{base_url}/{entry['path']}", timeout=60) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != entry["size"] or digest != entry["sha256"]:
            raise RuntimeError(
                f"Fixture verification failed for {relative}: "
                f"size={len(payload)} sha256={digest}"
            )
        target.write_bytes(payload)
        total += len(payload)
    print(
        f"Fetched and verified {len(manifest['files'])} fixtures "
        f"({total} bytes) from {manifest['source_commit']} into {destination}"
    )


if __name__ == "__main__":
    main()
