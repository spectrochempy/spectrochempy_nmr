# Contributing to SpectroChemPy NMR

## Commit and pull-request prefixes

Use the SpectroChemPy prefix vocabulary for commit subjects and pull-request
titles:

| Prefix   | Usage                               |
| -------- | ----------------------------------- |
| `ENH:`   | User-facing enhancement or feature |
| `FIX:`   | Bug fix                             |
| `DOC:`   | Documentation changes               |
| `TEST:`  | Test addition or modification       |
| `CI:`    | CI/CD and workflow changes          |
| `DEV:`   | Developer tooling                   |
| `PERF:`  | Performance improvement             |
| `MAINT:` | Refactoring, cleanup, maintenance   |
| `REL:`   | Release-related work                |

Use the uppercase prefix followed by a colon, for example:

```text
ENH: normalize Bruker complex 1D acquisitions
FIX: preserve the native axis order
```

`feat:` and other Conventional Commit prefixes are not part of this vocabulary.
Branch names are separate; a branch such as `feat/bruker-normalization` is
unaffected.

The `check-prefix` workflow validates pull-request titles. As in SpectroChemPy,
Dependabot pull requests are exempt, and the `non-standard-prefix` label
explicitly bypasses title validation for an intentional exception.

Install the regular pre-commit hooks from the repository root:

```bash
python -m pip install pre-commit
pre-commit install
```
