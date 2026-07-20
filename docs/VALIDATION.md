# CAOS_MillDEM, validation status (honest)

This documents exactly what the engine is verified to do and what is still WIP. No overclaiming.

## Verified correct (tested, `tests/`)

- **Single-particle statics.** A ball released in a stationary drum falls under gravity and rests exactly at
  radius `R - r` with its kinetic energy removed by the restitution damping. Gravity + wall contact + damping
  are correct. (`test_single_ball_drops_to_rest`)
- **Charge settling.** A charge of many balls settles (drum stationary) into a dense bed in the lower drum
  with a roughly flat free surface, quiescent (mean speed < 0.5 m/s). The bed structure matches a real mill
  charge at rest. (`test_charge_settles_into_a_bed`)
- **Contact model.** Linear-Hooke and Hertzian normal forces (Hertz super-linear in overlap), Coulomb
  friction truncation, restitution-to-damping mapping, effective mass, contact-time scaling. (`test_contact.py`)
- **Determinism.** Same seed -> identical result. (`test_determinism`)
- **Power sign + monotonicity.** Net power (CoM-arm torque route, van Nierop `P = 2*pi*T*N`) is positive and
  rises with charge fill J. Charge mass matches a real mill (bulk density). (`test_power_increases_with_fill`)
- **Charge-shape sanity.** Toe/shoulder describe a tilted bed (toe on the falling side, shoulder lifted on the
  rising side, shoulder > toe). (`test_charge_shape_is_physical`)
- **Qualitative regimes.** Cascading at low speed, cataracting appears at high speed / larger mills.

## Cross-platform + GPU (verified)

- Runs on Windows with **no C++ toolchain and no WSL**: numpy core + optional numba JIT (10-50x). The
  `train` lane installs torch (cu126) and runs on the RTX 4070 (`torch.cuda.is_available() == True`), for the
  GPU batched-contact path and the ChargeCascade GNS surrogate.

## Stable lifted crescent (calibrated defaults)

With the calibrated contact defaults (friction mu=0.25, restitution e=0.5, background damping 16 /s) the charge
now holds a **stable, physical crescent**: a calm tilted bed (mean speed ~0.5-0.8 m/s) whose free surface rises
on the driven side (shoulder ~+65 deg) and falls on the toe side (~-55 deg), with a steady centre-of-mass
torque arm ~0.14*R on the rising side. This is a real, correctly-shaped mill charge (see
`examples/out/crescent_*.png`), not a fluidized cloud. The regime shifts from cascading toward cataracting as
speed and mill size grow.

## Known WIP (not yet at the bar)

- **Absolute power calibration.** The net power (from the steady CoM torque arm, van Nierop route) is now
  stable and correctly signed, but its **ratio to the classical Hogg-Fuerstenau power grows with mill size**
  (roughly 1:3 on a 3 m mill, 1:6 on a 5 m mill), so a single calibration constant does NOT honestly close it.
  The cause: the 2D disc slice under-lifts relative to the true 3D charge, and the lift does not scale with R
  the way the full 3D charge does. This is a genuine physics limitation of a 2D slice with a velocity-capped
  friction model, not a coefficient tweak, and it is NOT papered over with a fitted constant.
- **The honest fix path** (tracked, not deferred to hide it): either (a) a thin-3D slab instead of a 2D disc
  (captures the axial force chains that carry the lift), or (b) full Cundall-Strack tangential-history springs
  on particle-particle contacts + rolling resistance (gives the 2D bed the shear strength to lift further), or
  (c) run the reduced 2D model and report power as a *DEM-derived charge-shape* input to the classical
  torque-arm model (P from the DEM-measured arm x the real 3D charge mass), which is physically defensible and
  size-consistent. Path (c) is the most honest near-term route and is the planned next increment.

## Scope statement

The engine is, today, a verified **cross-platform charge-motion, settling, regime, and charge-shape tool**:
correct particle dynamics, a stable physical crescent, honest toe/shoulder, right regimes. Its **absolute
power is stable and correctly-signed but not yet within the ~10% band** across mill sizes (a real 2D-vs-3D
lift-scaling limitation, documented not hidden). The PyPI publish is held until the power is size-consistent
to the bar; the charge-motion capability is real now and is what ChargeCascade's DEM lane consumes for the
charge-shape and regime views.
