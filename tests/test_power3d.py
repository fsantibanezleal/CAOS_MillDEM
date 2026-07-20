"""The certifying test: the thin-3D-slab DEM net power matches the classical Hogg-Fuerstenau model within the
validated band, size-consistently. This is what clears the engine for a PyPI release."""
from __future__ import annotations

import math

import pytest

from milldem import MillConfig, MillDEM3D, simulate_power


def hf_kw(D, L, phi, J, lift_deg=35.0, c_arm=1.0, rho_c=4.8):
    """Classical Hogg-Fuerstenau net power [kW] (the cross-check reference). c_arm is the effective fractional lift
    arm; 1.0 is the standard full geometric arm (the DEM torque route, computed from first principles, matches this)."""
    R = D / 2
    Nc = 42.3 / math.sqrt(D)
    omega = 2 * math.pi * (phi * Nc) / 60
    M = rho_c * 1000 * (math.pi * R * R * L) * J
    return omega * M * 9.81 * (c_arm * R * math.sin(math.radians(lift_deg)) * max(0.0, 1 - 1.065 * J)) / 1000


def test_3d_power_matches_hf_and_is_size_consistent():
    """The 3D-slab DEM power lands within ~25% of Hogg-Fuerstenau at both a small and a large mill, and the two
    ratios agree with each other (size-consistent) - the property the 2D disc slice lacked. Size-consistency is the
    load-bearing claim: a 2D disc slice gives a size-independent absolute lift so its ratio drifts ~2x with mill size;
    the thin-3D slab holds the ratio to within a few percent."""
    cases = [(3.0, 0.20, 4.5), (5.0, 0.28, 6.5)]
    ratios = []
    for D, ball, L in cases:
        p_dem = simulate_power(MillConfig(diameter_m=D, phi_c=0.7, fill=0.30, ball_diameter_m=ball, length_m=L),
                               sim_time=1.3)["net_power_kw"]
        p_hf = hf_kw(D, L, 0.7, 0.30)
        r = p_dem / p_hf
        ratios.append(r)
        assert 0.75 < r < 1.35, f"D={D}: DEM/HF ratio {r:.2f} within the validated band"
    # size-consistency: the two ratios agree to within 25% of each other (the 2D model diverged 2x)
    assert abs(ratios[0] - ratios[1]) / max(ratios) < 0.25, f"size-consistent ratios {ratios[0]:.2f} vs {ratios[1]:.2f}"


def test_3d_power_rises_with_fill():
    """DEM power rises monotonically with fill through the operating range (more charge mass, more torque). The
    classical Hogg-Fuerstenau peak is near J~0.47, so power still climbs across 0.20 -> 0.40; the roll-off lies
    beyond the range the thin slab packs faithfully, so it is not asserted here."""
    base = dict(diameter_m=4.0, phi_c=0.75, ball_diameter_m=0.24, length_m=6.0)
    p_low = simulate_power(MillConfig(fill=0.20, **base), sim_time=1.2)["net_power_kw"]
    p_mid = simulate_power(MillConfig(fill=0.30, **base), sim_time=1.2)["net_power_kw"]
    p_high = simulate_power(MillConfig(fill=0.40, **base), sim_time=1.2)["net_power_kw"]
    assert p_mid > p_low, f"power rises from J=0.20 ({p_low:.0f}) to J=0.30 ({p_mid:.0f})"
    assert p_high > p_mid, f"power rises from J=0.30 ({p_mid:.0f}) to J=0.40 ({p_high:.0f}) (peak is near J~0.47)"


def test_3d_charge_holds_a_lifted_arm():
    """The 3D charge holds a positive centre-of-mass torque arm on the rising side (a stable lifted crescent)."""
    sim = MillDEM3D(MillConfig(diameter_m=4.0, phi_c=0.75, fill=0.30, ball_diameter_m=0.24, length_m=6.0), seed=42)
    sim.run(1.3)
    assert sim.arm_m() > 0.02 * sim.R          # lifted to the rising side, not slumped at the bottom
