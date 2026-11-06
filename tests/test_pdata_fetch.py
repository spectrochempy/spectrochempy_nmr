"""A corpus fetch must validate before writing or replacing user files."""

import hashlib
import importlib.util
import io
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def fetcher(monkeypatch):
    source = Path(__file__).parents[1] / "tools" / "fetch_pdata_qualification.py"
    spec = importlib.util.spec_from_file_location("pdata_fetcher_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"synthetic source data"
    archive = io.BytesIO()
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("synthetic/1/pdata/1/1r", payload)
    content = archive.getvalue()
    monkeypatch.setattr(module, "ACQUISITION", "synthetic/1")
    monkeypatch.setattr(
        module, "FILES", {"pdata/1/1r": hashlib.sha256(payload).hexdigest()}
    )
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: io.BytesIO(content))
    return module, payload


def test_verified_fetch_is_idempotent_and_does_not_overwrite_existing_data(
    fetcher, tmp_path
):
    module, payload = fetcher
    location = module.fetch(tmp_path)
    assert (location / "1r").read_bytes() == payload
    assert module.fetch(tmp_path) == location
    (location / "1r").write_bytes(b"unrelated existing data")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        module.fetch(tmp_path)
    assert (location / "1r").read_bytes() == b"unrelated existing data"


@pytest.mark.parametrize("corruption", ["archive", "member"])
def test_bad_hash_never_writes_payloads(fetcher, tmp_path, corruption):
    module, _ = fetcher
    if corruption == "archive":
        module.SHA256 = "invalid"
    else:
        module.FILES["pdata/1/1r"] = "invalid"
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.fetch(tmp_path)
    assert not list(tmp_path.iterdir())
