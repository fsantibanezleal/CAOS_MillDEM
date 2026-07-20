# CAOS_MillDEM

A cross-platform 2D soft-sphere **discrete element method (DEM)** engine for tumbling-mill charge motion and
power. No C++ toolchain, no WSL: pure NumPy with an optional Numba JIT and an optional Torch-CUDA path.

It simulates a rotating mill disc slice (the Govender et al. 2015 reduced setup, width = one particle
diameter) with a soft-sphere contact law (linear Hookean or Hertzian, Coulomb friction, restitution damping)
and reports the DEM charge shape (toe/shoulder), the motion regime (cascading / cataracting / centrifuging),
and the net power via the van Nierop (2001) torque route `P = 2*pi*T*N`.

```python
from milldem import simulate, MillConfig
m = simulate(MillConfig(diameter_m=5.0, phi_c=0.75, fill=0.30), sim_time=2.0)
print(m.net_power_kw, m.regime, m.toe_deg, m.shoulder_deg)
```

CLI: `milldem run --D 5 --phi 0.75 --J 0.30 --time 2.0 --json out.json`

See `docs/` for the contact-model equations, the power routes, and the validation against the classical
Hogg-Fuerstenau and Morrell power models. MIT licensed.
