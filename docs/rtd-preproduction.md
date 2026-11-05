# RTD preproduction plan

This repository configuration is prepared for a separate Read the Docs
Community project, provisionally named `spectrochempy-nmr`. No RTD project,
custom domain, parent/subproject association, DNS record, or core project
setting is created or changed by this phase.

## Current build contract

`.readthedocs.yaml` uses Ubuntu 24.04, Python 3.13, Sphinx's native HTML build,
and `fail_on_warning`. The RTD pre-build job fetches only the verified gallery
fixtures documented in `docs/fixtures.md`. `docs/requirements-rtd.txt` pins
the published core (`spectrochempy==1.1.3`) and Hypercomplex (`0.2.0`); the NMR
package is installed from the checkout being built. Sphinx-Gallery executes the
six approved examples on RTD; local builds remain non-executing unless
`NMR_DOCS_EXECUTE=1` is set. No experimental 2D fixtures are fetched.

The RTD HTML job uses a serialized Sphinx invocation instead of RTD's default
`-j auto`: the parallel invocation reproduced a shared plugin-metadata failure
in the relaxation example, while the same clean environment passes serially.
This is a targeted reliability constraint, not a scientific-code change.

The local phase-1 validation used core `1.1.4.dev21` from a checkout. The RTD
resolution is intentionally recorded separately because that development core
revision is not a published installable dependency.

## RTD project setup, after explicit authorization

1. Verify that the `spectrochempy-nmr` project slug is available, or inspect
   the existing project before changing anything.
2. Create/connect only the standalone NMR repository.
3. Set the default branch to `master` (or the chosen development branch),
   enable pull-request builds, and keep the project public if appropriate.
4. Build the default branch and inspect the HTML, `objects.inv`, gallery and
   build log before activating any release version.
5. Do not associate the project with the core as a subproject in this phase.

RTD pull-request previews are not indexed by RTD search; link and page checks
must therefore be performed directly on the preview URL.

## Version policy

RTD creates `latest` from the configured default branch. The tag
`spectrochempy-nmr-v0.2.0` is a package release tag, but it predates the
standalone documentation and contains neither `docs/` nor `.readthedocs.yaml`.
It must not be activated or presented as the first stable documentation build.

RTD's automatic stable detection is based on semantic-version-compatible tag
names. The prefixed `spectrochempy-nmr-v0.2.0` tag must therefore be handled
explicitly in the dashboard, or a later documentation-bearing package tag
with the repository's approved naming convention must be used. Existing tags
must not be rewritten.

For historical builds, use a documentation-bearing revision with exact core,
NMR, Hypercomplex and documentation dependencies. Never substitute `latest`
or a mutable branch for an archived dependency or Intersphinx inventory.

## Local validation and measurements

Validate the RTD-equivalent dependency resolution with::

   python -m venv /tmp/opencode/nmr-rtd-venv
   . /tmp/opencode/nmr-rtd-venv/bin/activate
   python -m pip install -r docs/requirements-rtd.txt
   python -m pip install .
   python docs/fetch_gallery_data.py
   NMR_DOCS_EXECUTE=1 time python -m sphinx -W --keep-going -b html docs docs/_build/html

Record Python, installed package versions, wall time, six execution summaries,
gallery generation time, and peak resource information from the RTD build log.
Do not infer a cost reduction from a single local measurement.

The Phase 2b local evidence is recorded in `docs/phase2b-validation.md`.
