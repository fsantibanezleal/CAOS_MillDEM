"""Soft-sphere contact force models for DEM.

Two force laws behind one interface, both verified verbatim against the LAMMPS granular pair-style docs
(docs.lammps.org/pair_gran.html) and the Golpayegani & Rezai (2022) review (Eqs 23-25):

Linear Hookean (``gran/hooke``)::

    F_n = k_n * delta       - m_eff * gamma_n * v_n      (normal:   spring - dashpot)
    F_t = -(k_t * ds_t)     - m_eff * gamma_t * v_t      (tangent:  shear spring - dashpot)
    |F_t| <= mu * |F_n|                                  (Coulomb friction truncation)

Hertzian (``gran/hertz/history``): the Hookean force scaled by ``sqrt(delta) * sqrt(R_i R_j / (R_i + R_j))``.

Damping is not free: the dashpot constant ``gamma_n`` is set from a target coefficient of restitution ``e``.
For the linear model the closed form is exact::

    gamma_n = -2 * ln(e) * sqrt(k_n / m_eff) / sqrt(pi**2 + ln(e)**2)

(the standard linear-spring-dashpot restitution relation). For Hertz the Tsuji (1992) velocity-independent
form is used. References: Cundall & Strack 1979 (DOI 10.1680/geot.1979.29.1.47); Tsuji, Tanaka & Ishida 1992
(DOI 10.1016/0032-5910(92)88030-L); LAMMPS granular docs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContactModel:
    """Contact parameters. ``model`` is ``"hooke"`` (linear) or ``"hertz"``.

    ``kn``/``kt`` are the normal/tangential stiffnesses (N/m for Hooke; the Hertz prefactor for hertz),
    ``e`` the coefficient of restitution (0..1), ``mu`` the sliding-friction coefficient, ``mu_r`` the
    rolling-friction coefficient (a resistive moment, 0 disables it).
    """

    model: str = "hooke"
    kn: float = 1.0e5          # normal stiffness FLOOR [N/m] (the engine auto-scales up for stable integration)
    kt_ratio: float = 2.0 / 7  # kt = kt_ratio * kn (the standard 2/7 shear-to-normal ratio)
    e: float = 0.5             # coefficient of restitution (0.45-0.90 typical for mill media)
    mu: float = 0.25           # sliding friction (calibrated so the charge holds a stable lifted crescent;
                               # too high fluidizes the bed, too low lets it slump)
    mu_r: float = 0.05         # rolling friction (resistive-moment coefficient)

    def kt(self) -> float:
        return self.kt_ratio * self.kn

    def normal_damping(self, m_eff: float) -> float:
        """The normal dashpot constant gamma_n [1/s] giving the target restitution e, linear model."""
        e = min(max(self.e, 1e-4), 0.999)
        lne = math.log(e)
        # gamma_n such that the linear spring-dashpot yields restitution e (Antypov & Elliott 2011 form)
        return -2.0 * lne * math.sqrt(self.kn / m_eff) / math.sqrt(math.pi ** 2 + lne ** 2)

    def contact_time(self, m_eff: float) -> float:
        """The linear contact half-period t_c = pi * sqrt(m_eff / kn) [s]; dt must be a fraction of this."""
        return math.pi * math.sqrt(m_eff / self.kn)


def effective_mass(m_i: float, m_j: float) -> float:
    """m_eff = m_i m_j / (m_i + m_j); for a particle-wall contact pass m_j = inf (returns m_i)."""
    if math.isinf(m_j):
        return m_i
    return m_i * m_j / (m_i + m_j)


def normal_force(model: ContactModel, delta: float, v_n: float, m_eff: float, r_eff: float) -> float:
    """Scalar normal contact force [N] (positive = repulsive) for overlap ``delta`` >= 0 and normal
    approach velocity ``v_n`` (positive when the bodies approach). ``r_eff`` = R_i R_j /(R_i + R_j)."""
    if delta <= 0.0:
        return 0.0
    gamma_n = model.normal_damping(m_eff)
    if model.model == "hertz":
        scale = math.sqrt(delta) * math.sqrt(max(r_eff, 1e-12))
        return scale * (model.kn * delta + m_eff * gamma_n * v_n)
    # linear Hooke: spring + dashpot (damping opposes approach; v_n>0 approaching)
    return model.kn * delta + m_eff * gamma_n * v_n


def tangential_force(model: ContactModel, ds_t: float, v_t: float, fn: float, m_eff: float) -> float:
    """Scalar tangential force [N], Coulomb-truncated to |F_t| <= mu*|F_n|. ``ds_t`` is the accumulated
    elastic tangential displacement, ``v_t`` the tangential relative velocity."""
    gamma_t = model.normal_damping(m_eff) * 0.5  # tangential damping ~ half normal (common choice)
    ft = -(model.kt() * ds_t) - m_eff * gamma_t * v_t
    cap = model.mu * abs(fn)
    if abs(ft) > cap:
        ft = math.copysign(cap, ft)
    return ft
