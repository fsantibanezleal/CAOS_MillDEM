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

    # mean torque over the settled window. The particle masses are per SLICE (thickness = one top-ball
    # diameter), so the accumulated shell torque is the torque of ONE disc slice of thickness = 2*rmax, not a
    # per-unit-length value. The full mill of length L holds L/(2*rmax) such slices; net power scales linearly.
    hist = np.asarray(sim._torque_hist)
    k = int(len(hist) * settle_frac)
    settled = hist[k:] if len(hist) > k else hist
    slice_torque = float(np.mean(settled)) if settled.size else 0.0  # N*m for one slice of thickness 2*rmax
    slice_thickness = 2.0 * float(sim.r.max())
    n_slices = cfg.length_m / slice_thickness
    net_power_w = 2.0 * math.pi * abs(slice_torque) * N * n_slices
    net_power_kw = net_power_w / 1000.0
    mean_torque = slice_torque

    px, py, vx, vy, r = _settled_positions(sim)
    rad = np.sqrt(px * px + py * py)
    speed = np.sqrt(vx * vx + vy * vy)
    mean_speed = float(np.mean(speed)) if speed.size else 0.0

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

    # toe / shoulder from the near-wall charge bed. The drum rotates so the charge is lifted up the RISING
    # side; the shoulder is the highest point on that side where the charge leaves the wall, the toe is the
    # impact foot where cataracting/cascading charge lands on the opposite lower side. We measure both as an
    # angle from the vertical (12 o'clock = 0 deg, positive toward the rising side), which is the convention
    # the classical toe/shoulder use. The mill here rotates counter-clockwise (omega>0), lifting the charge up
    # the +x (right) side, so the rising side is x>0.
    # angle from 12 o'clock, positive clockwise-from-top toward +x: phi = atan2(x, y)
    near_bed = near_wall & (rad > R - 2.5 * cfg.ball_diameter_m)
    if near_bed.sum() > 3:
        phi = np.degrees(np.arctan2(px[near_bed], py[near_bed]))  # 0=top, +90=right(+x), 180/-180=bottom
        # shoulder = the most-lifted point on the rising (+x, phi in (0,180)) side = the max phi still < 150
        rising = phi[(phi > -20) & (phi < 175)]
        shoulder_deg = float(np.percentile(rising, 92)) if rising.size > 2 else 55.0
        # toe = the impact foot on the falling side (phi negative, i.e. the -x lower-left), the min phi
        toe_deg = float(np.percentile(phi, 8))
    else:
        shoulder_deg, toe_deg = 55.0, -45.0

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
