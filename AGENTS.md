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

## Development skill

Read [spectrochempy-nmr-dev](.agents/skills/spectrochempy-nmr-dev/SKILL.md)
when editing user guides, notebooks, tutorials, or gallery examples, when
reviewing local changes, or when preparing, publishing, reviewing, or merging
a PR.

Every PR requires independent review in a fresh temporary subagent session.
Merge only after an explicit `APPROVE` for the exact current head and passing
required checks. Follow the skill for the review handoff and follow-up.
