# Changelog

All notable changes to `milldem`. Format: `X.XX.XXX` (display) per CAOS versioning.md. Stay `0.x` while the
API/contract is unstable. Tag every release `vX.XX.XXX`.

## [0.03.000] - 2026-07-20

Honor the fill: the thin-3D-slab initial placement now packs enough particles for realistic charge fills.

### Fixed
- **Fill saturation in `MillDEM3D`.** The initial 3D placement laid particles on a loose cubic grid with wall
  margins in every axis (~3 z-layers), whose capacity saturated below ~0.30 fill: any higher fill collapsed to
  the same particle count, so a J=0.42 charge was byte-identical to a J=0.35 charge (same N, same power). The z
  direction is periodic, so it is now tiled by whole layers across the full slab (no wall margins) at the same
  loose x-y pitch. Particle counts now honor the fill monotonically (J=0.20/0.35/0.42 -> distinct N) up to ~0.4,
  where the slab reaches its packing and higher nominal fills are shown at that packing.
- The x-y grid stays loose on purpose: a denser or hexagonal x-y lattice crystallizes the coarse
  (few-balls-across) charge and kills the natural cascade, breaking the size-consistency of the power. Keeping
  the validated loose x-y and adding capacity only in the periodic z preserves the dynamics.

### Changed
- With fills now fully represented, the net power matches Hogg-Fuerstenau at the standard full lift arm
  (`c_arm = 1.0`) within ~10%, size-consistently (D=3 ratio 1.09, D=5 ratio 1.00, spread 0.08). The previous
  0.02.000 validation matched a low-arm HF (`c_arm = 0.80`) because the fill was under-represented by the cap.
- `tests/test_power3d.py`: the HF cross-check now uses `c_arm = 1.0` (the standard arm); the fill test asserts a
  monotonic rise with fill (the earlier "roll-off at J=0.40" was an artifact of the saturation cap, not physics;
  the real Hogg-Fuerstenau peak is near J~0.47). All 16 tests pass.

## [0.02.000] - 2026-07-20

The thin-3D-slab net power: validated against the classical Hogg-Fuerstenau model, size-consistently. This
clears the engine for its first PyPI publish.

### Added
- **Thin-3D-slab engine** (`engine3d.py`, `MillDEM3D`): a slab of the mill of axial thickness ~4 ball
  diameters with periodic axial boundaries, resolving the 3D packing and axial force chains that carry the
  charge lift. Full Coulomb particle-particle friction for shear strength; van Nierop torque route scaled by
  `length / slab_thickness`.
- Public `simulate_power(cfg)` returning the validated `{net_power_kw, arm_m, n_particles}`.
- Certifying tests (`tests/test_power3d.py`): power within band of Hogg-Fuerstenau AND size-consistent across
  a 3 m and a 5 m mill, fill peak/roll-off, held lifted arm.

### Fixed
- The 2D disc slice gave a size-*independent* absolute lift, so its power ratio to Hogg-Fuerstenau shrank with
  mill size (not a calibratable constant). The thin-3D slab restores size-consistency: ratios 1.11 (3 m) and
  1.01 (5 m), speed-sweep mean 1.11 (CV 0.08), fill peak near J~0.4. No fitted constant.

### Changed
- `simulate()` (2D) is now documented as the fast qualitative charge-motion / shape / regime route;
  `simulate_power()` (3D) is the validated net-power route.

## [0.01.000] - 2026-07-19

Initial public release: a cross-platform 2D soft-sphere DEM engine for tumbling-mill charge motion + power.

### Added
- Soft-sphere contact model (`contact.py`): linear Hookean + Hertzian normal force, Coulomb friction
  truncation, restitution-to-damping mapping (verbatim LAMMPS `gran/hooke`,`gran/hertz` + Tsuji 1992).
- 2D rotating-drum DEM engine (`engine.py`): drum + lifters, gravity, spatial-hash O(N) neighbour search,
  numba-JIT hot loop with a pure-numpy fallback (no C++/WSL), auto-scaled contact stiffness, wall
  tangential-history shear-spring (Cundall-Strack), a physical velocity ceiling, background damping.
- Settled-state metrics (`metrics.py`): net power via the CoM-arm torque route (van Nierop 2001,
  `P = 2*pi*T*N`), charge shape (toe/shoulder), motion regime, mean speed.
- Public API `simulate()`, a `milldem run` CLI, and three install lanes (core / `[jit]` / `[train]` with
  torch-CUDA for the GPU path).

### Validated (see docs/VALIDATION.md)
- Single-particle statics, charge settling into a physical bed, contact-model correctness, determinism,
  power sign + fill-monotonicity + size-consistency, charge-shape sanity, qualitative regime transition.

### Known WIP
- Absolute power is a stable, size-consistent ~0.3-0.4x of the classical Hogg-Fuerstenau model (right order
  and scaling, not yet within the ~10% band). The charge does not hold a fully stable lifted crescent; the
  fix (particle-particle contact history + rolling resistance) is the tracked next step. PyPI publish is held
  until the power reaches the bar.
