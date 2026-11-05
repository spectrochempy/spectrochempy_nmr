Compatibility and scope
=======================

This documentation documents the following contracts:

* NMR package version: ``0.2.0``;
* Hypercomplex package version: ``0.2.0``;
* core compatibility: ``spectrochempy>=1.1.0,<2``;
* public processing scope: validated 1D only;
* 2D reading/representation: documented separately from processing.

The exact tested Python, source revisions and documentation-tool versions are
recorded in ``docs/compatibility.toml`` and ``docs/build-manifest.txt``.
The latter can be refreshed with::

   python -m pip freeze > docs/build-manifest.txt

The build was produced from the local revisions recorded in that manifest.
They are not the source revisions listed in the 2026-10-28 architecture audit;
this difference is recorded explicitly and must be resolved by a release branch
before historical publication.

No core documentation source, ``docs/make.py``, RTD configuration, DNS, or
publication pipeline is used by this build.
