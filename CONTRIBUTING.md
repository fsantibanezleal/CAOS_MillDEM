# Contributing

Thanks for your interest in `milldem`. It is maintained by Felipe Santibanez-Leal. Contributions, issues,
and suggestions are welcome.

## Reporting issues

Please open a GitHub issue before starting non-trivial work, so intent is visible and effort is not
duplicated. A good issue states:

- What you observed (or want), with a minimal reproduction for a bug (a `MillConfig` and the observed vs
  expected result).
- The environment (OS, Python version, whether the numba `[jit]` extra is installed).
- One logical topic per issue.

## Development flow

1. Branch from `main` as `task/<short-slug>`.
2. Commit in small, focused units (one logical change per commit).
3. Open a pull request from your `task/<slug>` branch into `main`.
4. `main` is the released branch: every release is tagged `vX.XX.XXX` and published to PyPI by the
   `publish.yml` workflow (OIDC Trusted Publishing, no stored token).

## Physics changes carry a validation burden

`milldem` is a DEM engine whose value is that its numbers are trustworthy, so any change that touches the
contact model, the integrator, the placement, or the power route must keep the validation honest:

- Run the full suite (`pytest`), including the slow certifying tests in `tests/test_power3d.py`.
- The **size-consistency** of the net power (the DEM/Hogg-Fuerstenau ratio agreeing across mill sizes) is the
  load-bearing property, do not regress it.
- If a change shifts the validated numbers, update `docs/VALIDATION.md` and the `CHANGELOG.md` in the same
  commit, and say *why* the new numbers are correct. Never tune a constant to make a test pass; fix the
  physics or re-baseline the test with a documented reason.
- New behaviour needs a test that would fail without the change.

## Code conventions

- Code, identifiers, comments, and commit messages are written in **English**.
- Match the style of the surrounding code (naming, formatting, comment density).
- Keep the core importable with only `numpy` + `scipy`; `numba` and `torch` stay optional extras.
- No em-dash or emoji in code, comments, or docs.

## Local setup

See `README.md`. Install into an isolated environment (a project-local virtualenv), never globally:

```
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on POSIX
pip install -e ".[jit,dev]"
pytest -q
```

## License of contributions

By contributing, you agree that your contributions are licensed under the same MIT license as this
repository (see `LICENSE`).
