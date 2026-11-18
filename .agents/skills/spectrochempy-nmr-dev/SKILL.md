---
name: spectrochempy-nmr-dev
description: >-
  Write or edit SpectroChemPy NMR guides, notebooks, tutorials, and gallery
  examples, review local changes, or prepare, publish, review, or merge a PR.
---

## Tutorials and gallery examples

Use the public SpectroChemPy and plugin APIs wherever they express the
operation clearly. Check existing imports when editing an example, and use
APIs available in the versions supported by the documentation build.

- Use `dataset.plot()`, `scp.plot_multiple()`, or `scp.multiplot()` for
  plots and overlays.
- Use `NDDataset` and `Coord` methods and arithmetic for calculations, and
  `scp.Coord.linspace()` for regularly spaced axes.
- Generate line shapes with the public helper for the intended model:
  `scp.gaussian()`, `scp.lorentzian()`, `scp.voigt()`, or
  `scp.pseudovoigt()`. Match the width and normalization to the intended
  model when adapting an existing formula.
- Use direct NumPy or Matplotlib for features without a suitable public
  equivalent, explicit interoperability, or low-level comparisons such as
  checking a Fourier convention.
- In notebook-based tutorials and gallery examples, use `_ = expression`
  when an unused return value would appear as cell output. For example,
  `_ = dataset.plot()` hides the returned axes representation while keeping
  the figure visible. Keep useful diagnostics and intentionally displayed
  objects.

## PR publication

Keep PR preparation local. Push a branch only when the maintainer explicitly
asks to push it or to open a PR that needs it. Open a GitHub PR, including a
draft PR, only when the maintainer explicitly asks to open it. Write and
submit the title and description as part of opening the authorized PR.

## Task records and shared documents

Keep investigation, acquisitions or fixtures, commands, results, skips and
limits, review reports, and delivery outcome in the task's working record
under the workspace workflow, or in the local handoff/PR for a standalone
checkout. A routine PR does not require a maintainer audit file.

Update the existing RFC, architecture reference, or topic audit when a durable
decision, important scientific qualification, or reusable campaign conclusion
needs to be shared. Include enough evidence and limitations to assess shared
conclusions without access to local task tracking. Multi-PR coordination alone
does not require a new note. After delivery, reconcile shared documents and
indexes only when their content changes; record any pending shared update.

After an authorized push, verify the remote PR head and update the task record.
After a merge, verify it on GitHub and record the delivered result and explicit
follow-ups before closing the task. Inspect task-owned worktrees before
removing clean merged ones. Document edits and publication follow the existing
authorization rules.

## Independent review before merge

Every PR, including documentation, test-only and mechanical changes, needs
independent review. When the maintainer requests a review or delegates a
merge, start an independent review in a fresh temporary subagent session
with repository access. Give it the
original requirement, applicable repository instructions, exact base and
head commits, and any specific points to check. Omit the implementation
conversation and earlier reviews.

The reviewer checks the exact diff, verifies relevant claims and validation
against the sources, and returns findings, the reviewed commit, and one
verdict: `APPROVE`, `REQUEST_CHANGES`, or `NEEDS_DISCUSSION`.

For uncommitted work, provide the exact base commit and a snapshot of the
diff under review. Record which snapshot the verdict covers. The review
request authorizes the local review loop and corrections within task scope.
Git commits and publication follow the applicable authorization rules.

After each review, verify the findings, fix actionable issues, and run
targeted validation. Start a fresh independent subagent session for each
updated diff without asking again. Give each reviewer the exact revision
and keep the implementation conversation and earlier reviews out of their
context. Repeat until the current revision receives an explicit `APPROVE`.
If a finding needs a maintainer decision, surface it and resume the review
loop once it is resolved.

Return review reports to the parent session for the task record; a separate
maintainer Markdown report is not required. Record the verdict and reviewed
commit for a PR. Immediately before merging,
verify that GitHub still reports the approved head and that required checks
passed. Make the merge request conditional on that exact head. If the head
has changed, review the new head. If a guarded merge is unavailable,
leave the PR open. If an isolated reviewer is unavailable, report the
limitation, provide the handoff prompt, and leave any pending PR open.
