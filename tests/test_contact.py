"""Unit tests for the soft-sphere contact model (contact.py)."""
from __future__ import annotations

import math

import pytest

from milldem.contact import ContactModel, effective_mass, normal_force, tangential_force


def test_restitution_damping_monotonic():
    """Lower restitution e -> more damping (a more dissipative contact)."""
    m_soft = ContactModel(e=0.9, kn=1e5)
    m_hard = ContactModel(e=0.3, kn=1e5)
    assert m_hard.normal_damping(1.0) > m_soft.normal_damping(1.0)
    # e -> 1 gives near-zero damping (elastic)
    assert ContactModel(e=0.999, kn=1e5).normal_damping(1.0) < 1.0


def test_contact_time_scaling():
    """Contact time t_c = pi*sqrt(m/kn): stiffer -> shorter, heavier -> longer."""
    m = ContactModel(kn=1e5)
    t1 = m.contact_time(1.0)
    assert m.contact_time(4.0) == pytest.approx(2 * t1)          # 4x mass -> 2x time
    assert ContactModel(kn=4e5).contact_time(1.0) == pytest.approx(t1 / 2)  # 4x stiffness -> half time


def test_coulomb_truncation():
    """The tangential force never exceeds mu * |F_n| (Coulomb friction limit)."""
    m = ContactModel(kn=1e6, mu=0.4)
    fn = 1000.0
    # a large tangential displacement/velocity must saturate at mu*fn
    ft = tangential_force(m, ds_t=10.0, v_t=100.0, fn=fn, m_eff=1.0)
    assert abs(ft) <= m.mu * fn + 1e-9


def test_hertz_scales_with_overlap():
    """The Hertzian normal force grows super-linearly with overlap (delta^1.5), unlike the linear Hooke."""
    hooke = ContactModel(model="hooke", kn=1e6)
    hertz = ContactModel(model="hertz", kn=1e6)
    f_h1 = normal_force(hooke, 0.001, 0.0, 1.0, 0.05)
    f_h2 = normal_force(hooke, 0.002, 0.0, 1.0, 0.05)
    assert f_h2 / f_h1 == pytest.approx(2.0, rel=0.01)          # linear: doubles with overlap
    fz1 = normal_force(hertz, 0.001, 0.0, 1.0, 0.05)
    fz2 = normal_force(hertz, 0.002, 0.0, 1.0, 0.05)
    assert fz2 / fz1 > 2.5                                       # super-linear (delta^1.5)


def test_effective_mass_wall():
    assert effective_mass(3.0, math.inf) == 3.0                 # particle-wall -> particle mass
