# Building the SpectroChemPy NMR documentation

The build is independent of the SpectroChemPy monorepo documentation. From
this repository worktree:

```console
python -m venv /tmp/opencode/nmr-docs-venv
source /tmp/opencode/nmr-docs-venv/bin/activate
python -m pip install -r docs/requirements-docs.txt
python -m pip install -e /path/to/spectrochempy
python -m pip install -e /path/to/spectrochempy_hypercomplex
python -m pip install -e .
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

The editable installs intentionally use the three local checkouts for
development. Their exact versions and commits are recorded in
`docs/compatibility.toml`; a release build must replace them with the tagged
source revisions for that release.

For a tagged build, select the matching source links explicitly:

```console
NMR_DOCS_RELEASE=1 \
NMR_DOCS_GITHUB_REF=spectrochempy-nmr-v0.2.0 \
NMR_CORE_DOCS_URL=https://www.spectrochempy.fr/<compatible-core-version>/ \
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Replace ``<compatible-core-version>`` with the deployed core documentation
version selected for that release; release mode intentionally refuses to use
the moving ``latest`` default.

The gallery is non-executing by default for quick local documentation checks.
To execute the six public examples with the approved immutable fixtures:

```console
python docs/fetch_gallery_data.py
NMR_DOCS_EXECUTE=1 python -m sphinx -W --keep-going -b html docs docs/_build/html
```

RTD sets `READTHEDOCS=True`; the configuration then executes the gallery after
the pre-build fixture fetch. The excluded CP and 2D examples are never
selected. See `docs/fixtures.md` for provenance and licensing.

Record the tested environment with:

```console
python -m pip freeze > docs/build-manifest.txt
```

`docs/build-manifest.txt` is a local build record and is not required as a
source dependency. It must not be used as an installation requirements file.
