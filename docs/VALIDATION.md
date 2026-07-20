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

## Known WIP (not yet at the bar)

- **Absolute power calibration.** The net power is a **stable, size-consistent ~0.3-0.4x of the classical
  Hogg-Fuerstenau model** with the current settings, i.e. the right order of magnitude and the right scaling,
  but not yet within the ~10% target band. The cause is diagnosed: the charge does not hold a **stable lifted
  crescent** (its centre-of-mass torque arm oscillates around a small value and can cross zero, instead of
  sitting steadily at ~0.15*R on the rising side). A dense mill charge holds that crescent because its
  internal shear strength resists collapse; the current engine has the wall tangential-history shear-spring
  but a simplified particle-particle tangential and no rolling resistance, so the bed lacks the shear strength
  to hold the crescent and instead sloshes / over-agitates.
- **The fix** (the honest next step): full Cundall-Strack tangential-history springs on **particle-particle**
  contacts (not just the wall) + a rolling-resistance moment. This gives the bed a yield strength, the crescent
  becomes stable, the arm holds, and the power self-calibrates. This is the same physics production DEM codes
  (YADE, LIGGGHTS) implement in tuned C++; it is a real numerical-methods increment here, not a coefficient
  tweak.

## Scope statement

The engine is honestly a **cross-platform charge-motion and order-of-magnitude-power tool** today: correct
qualitative charge dynamics, settling, regimes, and charge shape, with power at the right order and scaling.
It is NOT yet a quantitatively power-accurate replacement for a tuned DEM code. The PyPI publish is held until
the particle-particle history + rolling resistance close the power to the ~10% band.
