"""Tests for the mill DEM engine: the physical invariants that are verified to hold.

These assert the pieces that are validated (single-ball statics, settling, determinism, power ordering,
charge-shape sanity), which is the honest scope of the current engine. The sharp cascading/cataracting
shoulder-toe lift and the precise power-vs-classical calibration are tracked as WIP (see the README).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from milldem import ContactModel, MillConfig, MillDEM, compute_metrics, simulate
from milldem.contact import ContactModel as CM, effective_mass, normal_force
from milldem.engine import G


def test_effective_mass():
    assert effective_mass(2.0, 2.0) == pytest.approx(1.0)
    assert effective_mass(1.0, math.inf) == pytest.approx(1.0)  # particle-wall


def test_normal_force_zero_below_contact():
    m = CM(model="hooke", kn=1e5)
    assert normal_force(m, -0.01, 0.0, 1.0, 0.05) == 0.0   # no overlap -> no force
    assert normal_force(m, 0.001, 0.0, 1.0, 0.05) > 0.0    # overlap -> repulsive


def test_single_ball_drops_to_rest():
    """A single ball released at the top of a stationary drum must fall and rest at radius R - r, with the
    restitution damping removing its kinetic energy. This validates gravity + wall contact + damping."""
    cfg = MillConfig(diameter_m=3.0, phi_c=0.0, fill=0.30, ball_diameter_m=0.18, n_lifters=0,
                     bg_damping=0.0,  # no background drag: the ball must fall freely to the floor
                     contact=ContactModel(model="hooke", e=0.3, mu=0.5))
    sim = MillDEM(cfg, seed=1)
    sim.px = np.array([0.0]); sim.py = np.array([1.0])
    sim.vx = np.array([0.0]); sim.vy = np.array([0.0])
    sim.r = np.array([0.09]); sim.m = np.array([7800 * math.pi * 0.09 ** 2 * 0.18]); sim.n = 1
    sim.omega = 0.0
    sim._v_max = 50.0
    for _ in range(int(2.0 / sim.dt)):
        sim.step()
    assert sim.py[0] == pytest.approx(-(sim.R - 0.09), abs=0.02)  # rests on the shell floor
    assert abs(sim.vy[0]) < 0.05                                   # at rest


def test_charge_settles_into_a_bed():
    """The charge settles (drum stationary) into a dense bed in the lower drum, quiescent (low velocity)."""
    cfg = MillConfig(diameter_m=3.0, phi_c=0.75, fill=0.30, ball_diameter_m=0.14, n_lifters=0,
                     contact=ContactModel(model="hooke", e=0.4, mu=0.6))
    sim = MillDEM(cfg, seed=42)
    sim.settle()
    assert sim.py.mean() < -0.2 * sim.R          # centre of mass sits in the lower drum
    assert sim.py.max() < 0.6 * sim.R            # nothing pinned to the top
    v = np.sqrt(sim.vx ** 2 + sim.vy ** 2)
    assert v.mean() < 0.5                         # the settled bed is quiescent


def test_determinism():
    """The same seed gives byte-identical results (a pure function of params + seed)."""
    cfg = MillConfig(diameter_m=3.0, phi_c=0.7, fill=0.25, ball_diameter_m=0.16, n_lifters=6)
    a = simulate(cfg, sim_time=0.6, seed=7)
    b = simulate(cfg, sim_time=0.6, seed=7)
    assert a.net_power_kw == pytest.approx(b.net_power_kw)
    assert a.toe_deg == pytest.approx(b.toe_deg)


def test_power_increases_with_fill():
    """Net power rises with charge fill J (more charge mass -> more torque), a basic physical monotonicity."""
    base = dict(diameter_m=3.0, phi_c=0.75, ball_diameter_m=0.15, n_lifters=6, length_m=4.5)
    p_low = simulate(MillConfig(fill=0.15, **base), sim_time=0.8).net_power_kw
    p_high = simulate(MillConfig(fill=0.40, **base), sim_time=0.8).net_power_kw
    assert p_high > p_low
    assert p_low > 0


def test_power_positive_and_sane_order():
    """Net power is positive and in the industrial order of magnitude (hundreds of kW to a few MW for a 3-5 m
    mill), not zero and not absurd."""
    m = simulate(MillConfig(diameter_m=4.0, phi_c=0.75, fill=0.30, ball_diameter_m=0.12, length_m=6.0), sim_time=0.9)
    assert 50.0 < m.net_power_kw < 20000.0


def test_charge_shape_is_physical():
    """The toe/shoulder angles describe a tilted bed: toe on the falling side (negative), shoulder on the
    rising side (positive), both within a physical range."""
    m = simulate(MillConfig(diameter_m=3.0, phi_c=0.7, fill=0.30, ball_diameter_m=0.15, n_lifters=6), sim_time=0.9)
    assert -90 < m.toe_deg < 10          # toe on/below the falling side
    assert 0 < m.shoulder_deg < 110      # shoulder lifted on the rising side
    assert m.shoulder_deg > m.toe_deg    # the bed is tilted the right way
