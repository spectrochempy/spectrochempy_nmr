# SpectroChemPy NMR agent instructions

Read [CONTRIBUTING.md](CONTRIBUTING.md) for repository conventions. Use the
public plugin API when validating a user workflow, and run the tests relevant
to the change. Report qualifications that require external data separately
from checks run with repository fixtures.

For maintainer-directed work in the shared SpectroChemPy workspace, also
read the workspace-level `AGENTS.md`. Before substantial changes or
qualification, consult the relevant decisions, RFCs and audits in the private
maintainer repository when available. In particular, check existing decisions
before changing Fourier conventions, coordinate referencing, processing
semantics or the boundary between the core, this plugin and nmrglue-ng.

Update the relevant maintainer audit as findings become useful for handoff.
Record the acquisition or fixture, the commands or tests used, and what the
results establish. After delivery, reconcile the note with the verified
outcome. For a standalone contributor checkout, the public documentation and
tests provide the project contract.

## User guides, notebooks, and gallery examples

Use the public SpectroChemPy and NMR APIs whenever they express the
operation clearly. Use `dataset.plot()`, `scp.plot_multiple()`, and
`scp.multiplot()` for plots and overlays. Use `NDDataset` and `Coord`
methods and arithmetic for calculations, with helpers such as
`scp.Coord.linspace()` for grids and `scp.gaussian()` for Gaussian
profiles. Keep direct Matplotlib or NumPy for features without a suitable
public equivalent, explicit interoperability, and low-level comparisons
such as decoding a raw vendor file or checking a Fourier convention.
Check existing imports when editing examples. Use versions supported by
the documentation build.

In notebooks, use `_ = ...` for unused results that IPython would otherwise
display, such as axes objects and tuples returned by plot setters. Keep
figures, useful diagnostics, and object displays that the guide explains.
