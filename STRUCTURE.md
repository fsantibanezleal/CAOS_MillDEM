# STRUCTURE, the actual layout of this repo

`milldem` is a cross-platform soft-sphere DEM for tumbling-mill charge motion and power, published to PyPI as
[`milldem`](https://pypi.org/project/milldem/). It is a **Python library** (not a web-app product), so it does
not carry the CAOS web archetype's `frontend/`, `data-pipeline/`, or `app/` lanes; it is the reusable engine
that products (e.g. ChargeCascade's DEM lane) depend on. The design rationale lives in `docs/VALIDATION.md`.

## Tree

| Path | What |
|---|---|
| `milldem/` | the package |
| `milldem/contact.py` | the soft-sphere contact model: linear-Hooke + Hertz normal force, Coulomb friction, restitution-to-damping mapping (Tsuji, Tanaka & Ishida 1992), effective mass and contact-time scaling |
| `milldem/engine.py` | the 2D rotating-drum engine (`MillConfig`, `MillDEM`): drum + lifters, gravity, spatial-hash O(N) neighbours, density-scaled stiffness, velocity ceiling, numba-JIT hot loop with a numpy fallback |
| `milldem/engine3d.py` | the **thin-3D-slab** engine (`MillDEM3D`): an axial slab with periodic axial boundaries, the route that makes the net power size-consistent; the placement tiles the periodic z to honor the charge fill |
| `milldem/metrics.py` | settled-state metrics: charge shape (toe/shoulder), motion regime, mean speed, the 2D CoM-arm power route |
| `milldem/cli.py`, `__main__.py` | the `milldem` CLI: `run` (2D charge shape / regime) and `power` (validated thin-3D-slab net power) |
| `milldem/__init__.py` | public API: `simulate`, `simulate_power`, `MillConfig`, `MillDEM`, `MillDEM3D`, `ContactModel`, `compute_metrics`, `__version__` |
| `tests/` | `test_contact.py` (contact-law correctness), `test_engine.py` (2D statics, settling, determinism, charge shape), `test_power3d.py` (the certifying thin-3D power: size-consistency + fill trend) |
| `docs/VALIDATION.md` | the honest validation status: what is verified, the power numbers, the scope statement |
| `examples/` | `validate.py` + rendered figures under `out/` |
| `.github/workflows/` | `ci.yml` (cross-platform matrix: ubuntu/windows/macos x py3.11/3.13) + `publish.yml` (OIDC Trusted Publishing on a `v*` tag) |
| `requirements.txt` | core runtime (numpy, scipy) |
| `requirements-jit.txt` | + numba (the 10-50x JIT lane) |
| `requirements-train.txt` | + torch (cu126) and torch-geometric for the optional GPU / GNS-surrogate path |
| `pyproject.toml` | setuptools backend, MIT, `[project.scripts] milldem`, optional-deps `jit` / `train` / `dev` (ADR-0061) |
| `VERSION`, `CHANGELOG.md` | `X.XX.XXX` display version + the release log; every release is tagged `vX.XX.XXX` |

## Install lanes

- **Core**: `pip install milldem` (numpy + scipy; pure-Python, runs anywhere, no C++/WSL).
- **JIT**: `pip install "milldem[jit]"` (adds numba, 10-50x on the hot loop).
- **Train**: `pip install "milldem[train]"` (adds torch-cu126 + torch-geometric for the GPU / surrogate path).

## Release + publish

`main` is the released branch. Tagging `vX.XX.XXX` triggers `publish.yml`, which builds the sdist + wheel and
publishes to PyPI via **OIDC Trusted Publishing** (no stored token; ADR-0061). The dist name equals the import
name (`milldem`), with no `caos-` prefix.

## What this repo is not

- Not a web app: there is no browser frontend, no FastAPI lane, no baked-artifact web contract here. Those
  live in the *products* that consume `milldem` (for example CAOS_ChargeCascade's `data-pipeline/cclab/dem`).
- Not a GPU-only engine: the GPU/torch path is an optional extra; the core and the validated physics run on
  numpy alone.
