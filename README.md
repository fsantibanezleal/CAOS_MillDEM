# milldem

[![PyPI](https://img.shields.io/pypi/v/milldem)](https://pypi.org/project/milldem/)
[![Python](https://img.shields.io/pypi/pyversions/milldem)](https://pypi.org/project/milldem/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A cross-platform soft-sphere **discrete element method (DEM)** engine for tumbling-mill charge motion and
power. **No C++ toolchain, no WSL**: pure NumPy with an optional Numba JIT and an optional Torch-CUDA path.
(Repo: `CAOS_MillDEM`; import name and PyPI dist are both `milldem`.)

It resolves every particle and contact with a soft-sphere contact law (linear Hookean or Hertzian, Coulomb
friction, Tsuji restitution damping) and offers two routes:

- **2D charge shape / regime** (`simulate`, `MillDEM`): a fast rotating disc-slice read of the DEM charge
  shape (toe/shoulder) and motion regime (cascading / cataracting / centrifuging).
- **Thin-3D-slab net power** (`simulate_power`, `MillDEM3D`): an axial slab with periodic axial boundaries
  that resolves the force chains carrying the charge lift, so the net power (van Nierop 2001 torque route,
  `P = 2*pi*T*N`) is **size-consistent and within ~10% of the classical Hogg-Fuerstenau model**. A single 2D
  disc slice cannot do this (its power/HF ratio drifts ~2x with mill size).

## Install

```bash
pip install milldem              # core (numpy + scipy), runs anywhere
pip install "milldem[jit]"       # + numba, 10-50x on the hot loop
pip install "milldem[train]"     # + torch (cu126) + torch-geometric, the GPU / surrogate path
```

## Quick start

```python
from milldem import simulate, simulate_power, MillConfig

# 2D charge shape + regime (fast, qualitative)
m = simulate(MillConfig(diameter_m=5.0, phi_c=0.75, fill=0.30), sim_time=2.0)
print(m.regime, m.toe_deg, m.shoulder_deg)

# validated thin-3D-slab net power [kW]
p = simulate_power(MillConfig(diameter_m=5.0, phi_c=0.75, fill=0.30, ball_diameter_m=0.10), sim_time=1.5)
print(p["net_power_kw"], p["arm_m"], p["n_particles"])
```

CLI:

```bash
milldem run   --D 5 --phi 0.75 --J 0.30 --time 2.0 --json shape.json   # 2D charge shape / regime
milldem power --D 5 --phi 0.75 --J 0.30 --ball 0.10 --L 6.0            # validated 3D net power
```

## Validation

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the honest validated-scope statement: verified
single-particle statics, charge settling, contact-model correctness, determinism, the size-consistent
thin-3D-slab power (tested in `tests/test_power3d.py`), and the fill handling. Nothing is fitted to hide a
gap. Sources: Cundall & Strack 1979, Tsuji et al. 1992, Govender et al. 2015, van Nierop et al. 2001.

## Contributing, conduct, security, license

- [`CONTRIBUTING.md`](CONTRIBUTING.md), the branch flow and the validation burden that physics changes carry.
- [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), the Contributor Covenant.
- [`SECURITY.md`](SECURITY.md), how to report a vulnerability privately.
- [`STRUCTURE.md`](STRUCTURE.md), the repo layout.
- [MIT](LICENSE) licensed.
