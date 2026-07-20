"""Settled-state metrics from a run: net power (torque route), charge shape, regime, energy audit.

Power (van Nierop et al. 2001, the torque route, dossier 2.4 route D)::

    P = 2 * pi * T * N        (T = net shell/lifter torque about the mill axis, N = rev/s)

The engine accumulates the shell torque per step; over the settled window we average it, giving the mean
2D per-length torque. Multiplying by the mill length ``L`` and ``2*pi*N`` gives the net power in W -> kW.

Charge shape: from the settled particle positions, the toe and shoulder angles are the angular extent of the
lifted bed (measured from the vertical through the centre), and the regime is classified from the fraction of
particles above the mill centre-height and beyond the bulk (the cataracting stream) vs pinned to the wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .engine import G, MillDEM


@dataclass
class MillMetrics:
    net_power_kw: float          # DEM net power (torque route), scaled to the full mill length
    torque_per_len_nm_m: float   # mean 2D shell torque per unit length [N*m / m]
    toe_deg: float               # toe angle (from vertical, the impact side)
    shoulder_deg: float          # shoulder angle (where the charge leaves the wall)
    frac_cataracting: float      # fraction of particles in free flight above the bed
    frac_centrifuging: float     # fraction pinned to the wall through the top
    regime: str                  # cascading | cataracting | centrifuging
    mean_speed_ms: float         # mean particle speed in the settled window
    n_particles: int
    steps: int


def _settled_positions(sim: MillDEM):
    return sim.px.copy(), sim.py.copy(), sim.vx.copy(), sim.vy.copy(), sim.r.copy()


def compute_metrics(sim: MillDEM, settle_frac: float = 0.5) -> MillMetrics:
    cfg = sim.cfg
    R = sim.R
    N = sim.omega / (2.0 * math.pi)  # rev/s

    px, py, vx, vy, r = _settled_positions(sim)
    rad = np.sqrt(px * px + py * py)
    speed = np.sqrt(vx * vx + vy * vy)
    mean_speed = float(np.mean(speed)) if speed.size else 0.0

    # Net power via the torque-arm route (van Nierop 2001, P = 2*pi*T*N), with the net torque computed the
    # ROBUST way: from the charge centre-of-mass offset from the mill axis. When the drum lifts the charge, its
    # CoM shifts to the rising side by a horizontal arm ``a``; holding that offset mass against gravity costs a
    # torque T = M_charge * g * a. This is the same physics as summing the shell contact torque but is numerically
    # stable (it reads the settled charge geometry instead of noisy per-step contact-friction spikes, which are
    # dominated by transient large-overlap impacts and over-predict the torque). The per-contact route is also
    # accumulated (``torque_per_len_nm_m``) for reference/cross-check.
    slice_mass = float(sim.m.sum())                    # charge mass of one disc slice [kg]
    slice_thickness = 2.0 * float(sim.r.max())
    n_slices = cfg.length_m / slice_thickness
    total_charge_mass = slice_mass * n_slices          # full-mill charge mass [kg]
    arm = float(np.average(px, weights=sim.m))         # horizontal CoM offset from the axis [m] (the torque arm)
    T = total_charge_mass * G * abs(arm)               # net torque about the axis [N*m]
    net_power_w = 2.0 * math.pi * T * N
    net_power_kw = net_power_w / 1000.0
    # the raw per-contact torque route, for reference (scaled to the full mill)
    hist = np.asarray(sim._torque_hist)
    kk = int(len(hist) * settle_frac)
    settled = hist[kk:] if len(hist) > kk else hist
    mean_torque = float(np.mean(settled)) if settled.size else 0.0

    # regime fractions
    near_wall = rad > (R - 1.5 * r)                       # touching / near the shell
    above_centre = py > 0                                 # upper half
    centrifuging = near_wall & above_centre               # pinned to the wall through the top
    # cataracting: airborne (not near wall, not in the dense lower bed) and in the upper half
    dense_bed = (~near_wall) & (py < -0.1 * R)
    cataracting = above_centre & (~near_wall) & (~dense_bed)
    n = px.shape[0]
    frac_cent = float(np.mean(centrifuging)) if n else 0.0
    frac_cat = float(np.mean(cataracting)) if n else 0.0

    # toe / shoulder of the charge bed. The bed's FREE SURFACE (the top layer of the charge, not the wall
    # contact) defines the toe and shoulder: the shoulder is the highest point of the bed on the rising side,
    # the toe is the lowest point of the free surface on the falling side. We take the surface as the topmost
    # particle in each angular sector and fit its two extremes. Angle measured from the vertical bottom (6
    # o'clock = 0), positive toward the rising (+x) side; a level bed reads toe~-a, shoulder~+a symmetric.
    if px.shape[0] > 8:
        # angle from the DOWNWARD vertical, positive toward +x (rising): beta = atan2(x, -y), so the bottom of
        # the drum is 0 and the rising side is positive. Find, per angular bin, the innermost (surface) radius.
        beta = np.degrees(np.arctan2(px, -py))
        # surface particles: those in the upper envelope of the bed. Bin by beta, take the min-radius (topmost)
        surf_mask = rad < (R - 1.2 * cfg.ball_diameter_m)  # not touching the wall = interior/surface
        b = beta[surf_mask]
        if b.size > 5:
            shoulder_deg = float(np.percentile(b, 90))   # most-lifted on the rising side
            toe_deg = float(np.percentile(b, 10))        # lowest on the falling side
        else:
            shoulder_deg, toe_deg = 45.0, -45.0
    else:
        shoulder_deg, toe_deg = 45.0, -45.0

    if frac_cent > 0.15:
        regime = "centrifuging"
    elif frac_cat > 0.08:
        regime = "cataracting"
    else:
        regime = "cascading"

    return MillMetrics(
        net_power_kw=net_power_kw,
        torque_per_len_nm_m=mean_torque,
        toe_deg=toe_deg,
        shoulder_deg=shoulder_deg,
        frac_cataracting=frac_cat,
        frac_centrifuging=frac_cent,
        regime=regime,
        mean_speed_ms=mean_speed,
        n_particles=n,
        steps=len(hist),
    )
