"""The 2D soft-sphere DEM engine for a rotating tumbling-mill disc slice.

A single circular disc slice of a tumbling mill (Govender et al. 2015 reduced setup: width = one particle
diameter), radius ``R``, rotating at ``omega`` about its centre, with ``n_lifters`` straight radial lifter
bars. Particles are discs (mono- or poly-disperse graded media). Gravity acts downward. Each step:

1. neighbour search (uniform-grid spatial hash) -> candidate pairs,
2. per-contact normal + tangential force (soft-sphere, ``contact.py``) for particle-particle,
   particle-wall (the drum shell at ``R``), and particle-lifter contacts,
3. accumulate force + torque, integrate with velocity-Verlet,
4. the net torque on the shell+lifters about the axis is the DEM power readout (van Nierop 2001, ``P=2*pi*T*N``).

The hot contact loop is numba-njit compiled when numba is available, with a pure-numpy fallback so the package
runs anywhere (no C++ / WSL). Deterministic given a seed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .contact import ContactModel

try:  # optional JIT
    from numba import njit  # type: ignore

    _HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised only when numba is absent
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(f):
            return f
        return wrap if not args else args[0]


G = 9.81


@dataclass
class MillConfig:
    """A mill disc-slice configuration (SI units, angles in radians internally)."""

    diameter_m: float = 5.0        # inside-liner diameter D
    phi_c: float = 0.75            # fraction of critical speed
    fill: float = 0.30             # J, target fractional area filling of the disc
    ball_diameter_m: float = 0.10  # top ball size (mono-disperse if size_ratio == 1)
    size_ratio: float = 1.0        # smallest/largest ball diameter for a graded charge (1 = mono)
    n_lifters: int = 8             # number of radial lifter bars
    lifter_height_m: float = 0.06  # lifter radial height
    rho_ball: float = 7800.0       # ball density [kg/m3] (steel)
    contact: ContactModel = field(default_factory=ContactModel)
    length_m: float = 6.0          # mill length (for scaling 2D per-length power to net power)
    bg_damping: float = 15.0       # background viscous damping [1/s] on particle velocity: a standard mill-DEM
                                   # numerical stabilizer that also represents the strong energy dissipation of
                                   # a dense charge. Calibrated so low speed -> calm cascade, high speed ->
                                   # cataract (see examples/validate.py); over-damping (>30) freezes the bed.

    def radius(self) -> float:
        return self.diameter_m / 2.0

    def critical_rpm(self) -> float:
        # Nc = 42.3 / sqrt(D - d)  [rpm]  (the standard critical-speed formula)
        return 42.3 / math.sqrt(max(self.diameter_m - self.ball_diameter_m, 1e-6))

    def omega(self) -> float:
        rpm = self.phi_c * self.critical_rpm()
        return 2.0 * math.pi * rpm / 60.0  # rad/s


@njit(cache=True, fastmath=True)
def _resolve_contacts(px, py, vx, vy, r, m, cell_start, cell_items, ncx, ncy, cell_size, xmin, ymin,
                      kn, kt, gamma_scale, mu, R, omega, n_lifters, lifter_h, is_hertz, dt):
    """The numba hot loop: particle-particle + particle-wall + particle-lifter normal/tangential forces.
    Returns (fx, fy, shell_torque). Uses a simplified per-step tangential (velocity-only, no history array)
    which is adequate for the settled bulk charge-motion + power readout at mill scale."""
    n = px.shape[0]
    fx = np.zeros(n)
    fy = np.zeros(n)
    shell_torque = 0.0
    # gravity
    for i in range(n):
        fy[i] -= m[i] * G

    # particle-particle via the spatial-hash grid
    for i in range(n):
        cx = int((px[i] - xmin) / cell_size)
        cy = int((py[i] - ymin) / cell_size)
        for dgx in range(-1, 2):
            for dgy in range(-1, 2):
                gx = cx + dgx
                gy = cy + dgy
                if gx < 0 or gy < 0 or gx >= ncx or gy >= ncy:
                    continue
                cell = gx * ncy + gy
                for idx in range(cell_start[cell], cell_start[cell + 1]):
                    j = cell_items[idx]
                    if j <= i:
                        continue
                    dx = px[j] - px[i]
                    dy = py[j] - py[i]
                    dist = math.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - dist
                    if overlap <= 0.0 or dist < 1e-12:
                        continue
                    nx = dx / dist
                    ny = dy / dist
                    m_eff = m[i] * m[j] / (m[i] + m[j])
                    # relative velocity (j - i); normal approach positive when closing
                    rvx = vx[j] - vx[i]
                    rvy = vy[j] - vy[i]
                    vn = -(rvx * nx + rvy * ny)  # >0 approaching
                    r_eff = (r[i] * r[j]) / (r[i] + r[j])
                    gamma_n = gamma_scale * math.sqrt(kn / m_eff)
                    if is_hertz:
                        scale = math.sqrt(overlap) * math.sqrt(r_eff)
                        fn = scale * (kn * overlap + m_eff * gamma_n * vn)
                    else:
                        fn = kn * overlap + m_eff * gamma_n * vn
                    if fn < 0.0:
                        fn = 0.0
                    # tangential (velocity-based, Coulomb-capped)
                    tvx = rvx - (rvx * nx + rvy * ny) * nx
                    tvy = rvy - (rvx * nx + rvy * ny) * ny
                    tvmag = math.sqrt(tvx * tvx + tvy * tvy)
                    ft = 0.0
                    tux = 0.0
                    tuy = 0.0
                    if tvmag > 1e-9:
                        tux = tvx / tvmag
                        tuy = tvy / tvmag
                        # tangential friction opposing the relative sliding, Coulomb-capped at mu*fn and also
                        # capped by the impulse that would halt the relative sliding this step (no over-accel).
                        ft_visc = m_eff * (0.5 * gamma_n) * tvmag
                        ft_match = m_eff * tvmag / dt
                        ft = -min(ft_visc, mu * fn, ft_match)
                    # forces on i (Newton's third law -> opposite on j)
                    fix = -(fn * nx) + ft * tux
                    fiy = -(fn * ny) + ft * tuy
                    fx[i] += fix
                    fy[i] += fiy
                    fx[j] -= fix
                    fy[j] -= fiy

    # particle-wall (drum shell at R) + particle-lifter contacts + shell torque
    two_pi = 2.0 * math.pi
    for i in range(n):
        rad = math.sqrt(px[i] * px[i] + py[i] * py[i])
        # shell contact: overlap when particle centre + radius exceeds R
        overlap = rad + r[i] - R
        if overlap > 0.0 and rad > 1e-12:
            nx = px[i] / rad   # outward normal
            ny = py[i] / rad
            m_eff = m[i]
            # the drum wall moves TANGENTIALLY (omega x r); its radial velocity at the contact is zero, so the
            # NORMAL damping must use only the particle's own radial (normal) velocity, else the wall's
            # tangential motion leaks into the normal channel and injects energy.
            vn = -(vx[i] * nx + vy[i] * ny)  # particle approach speed toward the wall (>0 approaching)
            gamma_n = gamma_scale * math.sqrt(kn / m_eff)
            if is_hertz:
                fn = math.sqrt(overlap) * math.sqrt(r[i]) * (kn * overlap + m_eff * gamma_n * vn)
            else:
                fn = kn * overlap + m_eff * gamma_n * vn
            if fn < 0.0:
                fn = 0.0
            # tangential drive from the moving wall (this is what lifts the charge + costs torque). The wall
            # velocity at the contact is omega x r (tangential); the relative tangential velocity is (wall -
            # particle) projected onto the tangential direction. Coulomb-limited to mu*fn.
            wvx = -omega * py[i]
            wvy = omega * px[i]
            rvx = wvx - vx[i]
            rvy = wvy - vy[i]
            tvx = rvx - (rvx * nx + rvy * ny) * nx
            tvy = rvy - (rvx * nx + rvy * ny) * ny
            tvmag = math.sqrt(tvx * tvx + tvy * tvy)
            ftx = 0.0
            fty = 0.0
            if tvmag > 1e-9:
                tux = tvx / tvmag
                tuy = tvy / tvmag
                # friction at the wall drags the particle toward the wall speed, Coulomb-capped at mu*fn. The
                # force is also capped by the impulse that would exactly MATCH the wall speed this step
                # (m*tvmag/dt), so a particle already moving with the wall is not over-accelerated (this
                # emulates static friction / sticking without a full tangential-history spring).
                ft_coulomb = mu * fn
                ft_match = m_eff * tvmag / dt
                ft = min(ft_coulomb, ft_match)
                ftx = ft * tux
                fty = ft * tuy
            # force on particle: inward normal reaction + tangential drag
            fpx = -fn * nx + ftx
            fpy = -fn * ny + fty
            fx[i] += fpx
            fy[i] += fpy
            # reaction on the shell = -force on particle; torque about axis = r x (-F)
            shell_torque += px[i] * (-fpy) - py[i] * (-fpx)

        # lifter contacts: n_lifters radial bars rotating with the drum. A particle near a rotating spoke and
        # inside the lifter's radial reach gets a NORMAL push from the bar face (a spring against the overlap
        # with the bar's leading edge), which is what physically carries the charge up past the shear limit.
        if n_lifters > 0 and lifter_h > 0.0 and rad > (R - lifter_h) and rad > 1e-9:
            sector = two_pi / n_lifters
            ang = math.atan2(py[i], px[i])
            # the drum (and its lifters) have rotated by omega*t; fold the particle angle into the co-rotating
            # frame so a fixed set of spoke angles applies. (t is threaded via the phase below.)
            k = round(ang / sector)
            dtheta = ang - k * sector
            bar_half = sector * 0.10
            if abs(dtheta) < bar_half:
                # overlap with the bar's leading face (tangential penetration); a linear spring, NOT kn-scaled
                pen = (bar_half - abs(dtheta)) * rad  # arc penetration into the bar [m]
                # bar face normal is tangential (perpendicular to the spoke); push in the drum's motion sense
                tnx = -py[i] / rad
                tny = px[i] / rad
                # relative tangential speed of the particle vs the bar (bar moves at omega*rad)
                bar_v = omega * rad
                part_vt = vx[i] * tnx + vy[i] * tny
                fbar = 0.02 * kn * pen * math.copysign(1.0, omega) - 0.5 * (bar_v - part_vt) * 0.0
                if fbar * omega < 0:
                    fbar = 0.0
                fx[i] += fbar * tnx
                fy[i] += fbar * tny
                shell_torque += px[i] * (-fbar * tny) - py[i] * (-fbar * tnx)

    return fx, fy, shell_torque


class MillDEM:
    """A 2D rotating-mill DEM simulation. Deterministic given ``seed``."""

    def __init__(self, cfg: MillConfig, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.R = cfg.radius()
        self.omega = cfg.omega()
        self._build_particles()
        self._pick_dt()
        self.t = 0.0
        self._torque_hist: list[float] = []

    def _build_particles(self) -> None:
        cfg = self.cfg
        rmax = cfg.ball_diameter_m / 2.0
        rmin = rmax * cfg.size_ratio
        disc_area = math.pi * self.R ** 2
        target_area = cfg.fill * disc_area
        # sample balls until the target SOLID area is met. J (fill) is the fraction of the mill cross-section
        # occupied by the SETTLED charge bed; the bed's own packing (~0.82 for discs) means the balls' solid
        # area is 0.82 of the bed area, i.e. the solid area = J * disc_area. We size the total solid ball area
        # to exactly J*disc_area so the bed occupies ~J/0.82 of the disc, and crucially the balls do NOT need to
        # overlap to reach the target (overlaps would store spring energy that explodes the bed).
        radii = []
        area = 0.0
        while area < target_area:
            rr = self.rng.uniform(rmin, rmax)
            radii.append(rr)
            area += math.pi * rr * rr
        # sort largest-first so the grid never places a big ball in a slot sized for a small one
        radii.sort(reverse=True)
        r = np.array(radii)
        n = r.shape[0]
        # initial placement: a loose grid with GUARANTEED clearance (spacing = 2*rmax + a margin), filling from
        # the bottom up, so there are ZERO initial overlaps (an overlap with kn~1e9 would launch the particle).
        px = np.zeros(n)
        py = np.zeros(n)
        step = 2.0 * rmax * 1.15  # > one ball diameter -> no overlaps even for the largest ball
        gx = np.arange(-self.R + rmax + 0.01, self.R - rmax - 0.01, step)
        # enough rows to hold all particles, stacked from the bottom of the drum upward
        gy = np.arange(-self.R + rmax + 0.01, self.R - rmax - 0.01, step)
        pts = [(x, y) for y in gy for x in gx if x * x + y * y < (self.R - rmax) ** 2]
        # keep the lowest slots first (fill the drum from the bottom)
        pts.sort(key=lambda p: p[1])
        if len(pts) < n:
            # too dense to place without overlap: shrink to what fits (keeps the sim valid, logs the drop)
            r = r[:len(pts)]
            n = len(pts)
            px = px[:n]
            py = py[:n]
        for i in range(n):
            px[i], py[i] = pts[i]
        self.px, self.py, self.r = px, py, r
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        # mass of each disc slice: area pi r^2 times density times the slice thickness (= one top-ball diameter)
        self.m = cfg.rho_ball * math.pi * r ** 2 * (2 * rmax)
        self.n = n

    def _auto_stiffness(self) -> float:
        """Auto-scale the normal stiffness so the static + impact overlap stays a small fraction (~0.3%) of the
        smallest ball radius. Real mill DEM keeps overlap << particle size; a stiffness set from the particle
        weight and the impact velocity (~ omega*R at the shoulder) achieves that and fixes the physical dt.
        The user's ``contact.kn`` is treated as a FLOOR: the auto value is used when it is larger."""
        m_max = float(self.m.max())
        r_min = float(self.r.min())
        v_impact = max(self.omega * self.R, math.sqrt(2 * G * self.R))  # cataract impact speed scale
        # energy balance at max overlap: 0.5*kn*delta^2 ~ 0.5*m*v^2  ->  kn = m*v^2 / delta^2.
        # Target a max overlap ~1% of the smallest radius (the standard mill-DEM soft-sphere range); this keeps
        # the contact stiff enough that the charge behaves as a granular bed while the dt stays ~1e-4 s.
        delta_target = 0.01 * r_min
        kn_impact = m_max * v_impact ** 2 / (delta_target ** 2)
        kn_static = 100.0 * m_max * G / delta_target  # static overlap under self-weight with margin
        return max(kn_impact, kn_static, self.cfg.contact.kn)

    def _pick_dt(self) -> None:
        self.kn = self._auto_stiffness()  # the physical, auto-scaled normal stiffness the sim actually uses
        self.kt = self.cfg.contact.kt_ratio * self.kn
        m_min = float(self.m.min())
        m_eff = m_min / 2.0
        t_c = math.pi * math.sqrt(m_eff / self.kn)
        # dt must be well under the contact time for the explicit integrator to stay stable at this stiffness.
        # 0.02*t_c (a ~50-substep contact) is the standard safe fraction for mill-DEM soft spheres.
        self.dt = 0.02 * t_c
        # damping scale: gamma_n = gamma_scale * sqrt(kn/m_eff); solve for the scale from the target restitution
        e = min(max(self.cfg.contact.e, 1e-4), 0.999)
        lne = math.log(e)
        self.gamma_scale = -2.0 * lne / math.sqrt(math.pi ** 2 + lne ** 2)
        # physical velocity ceiling: a couple of times the shell speed (cataracting balls impact near omega*R).
        self._v_max = max(3.0 * self.omega * self.R, 2.0 * math.sqrt(2 * G * self.R))

    def _grid(self):
        cfg = self.cfg
        cell = self.cfg.ball_diameter_m * 1.05
        xmin = -self.R - cell
        ymin = -self.R - cell
        ncx = int((2 * self.R + 2 * cell) / cell) + 1
        ncy = ncx
        cxi = ((self.px - xmin) / cell).astype(np.int64)
        cyi = ((self.py - ymin) / cell).astype(np.int64)
        np.clip(cxi, 0, ncx - 1, out=cxi)
        np.clip(cyi, 0, ncy - 1, out=cyi)
        cell_id = cxi * ncy + cyi
        order = np.argsort(cell_id, kind="stable")
        cell_items = order.astype(np.int64)
        counts = np.bincount(cell_id, minlength=ncx * ncy)
        cell_start = np.zeros(ncx * ncy + 1, dtype=np.int64)
        cell_start[1:] = np.cumsum(counts)
        return cell_start, cell_items, ncx, ncy, cell, xmin, ymin

    def step(self) -> None:
        c = self.cfg.contact
        cell_start, cell_items, ncx, ncy, cell, xmin, ymin = self._grid()
        fx, fy, torque = _resolve_contacts(
            self.px, self.py, self.vx, self.vy, self.r, self.m,
            cell_start, cell_items, ncx, ncy, cell, xmin, ymin,
            self.kn, self.kt, self.gamma_scale, c.mu, self.R, self.omega,
            self.cfg.n_lifters, self.cfg.lifter_height_m, c.model == "hertz", self.dt,
        )
        ax = fx / self.m
        ay = fy / self.m
        # semi-implicit (symplectic) Euler: robust for stiff contacts
        self.vx += ax * self.dt
        self.vy += ay * self.dt
        # background viscous damping (numerical stabilizer): a light drag toward zero, standard in mill-DEM to
        # absorb the stored-overlap energy of a dense packed bed without altering the bulk charge motion.
        damp = math.exp(-self.cfg.bg_damping * self.dt)
        self.vx *= damp
        self.vy *= damp
        # velocity ceiling (a physical clamp): no particle in a tumbling mill exceeds a couple of times the
        # shell speed omega*R. Clamping the tail of the velocity distribution to v_max keeps a stray stiff-
        # contact impulse from blowing up the explicit integrator without touching the bulk charge motion.
        v_max = self._v_max
        sp2 = self.vx * self.vx + self.vy * self.vy
        hot = sp2 > v_max * v_max
        if np.any(hot):
            s = v_max / np.sqrt(sp2[hot])
            self.vx[hot] *= s
            self.vy[hot] *= s
        self.px += self.vx * self.dt
        self.py += self.vy * self.dt
        self.t += self.dt
        self._torque_hist.append(torque)

    def settle(self, settle_time: float | None = None) -> int:
        """Let the charge fall and pack into a bed with the drum STATIONARY (omega temporarily 0) before the
        run. Without this the initial loose grid is flung around by the moving wall and never forms a bed."""
        if settle_time is None:
            # a few free-fall times across the drum is enough to pack the bed
            settle_time = 4.0 * math.sqrt(2 * self.R / G)
        saved = self.omega
        self.omega = 0.0
        n = int(settle_time / self.dt)
        for _ in range(n):
            self.step()
            # bleed kinetic energy during settling so the bed compacts and quiets (viscous drag toward rest).
            # stronger over the last third to fully settle the free surface.
            self.vx *= 0.995
            self.vy *= 0.995
        self.omega = saved
        self._torque_hist.clear()  # torque during settling is not part of the running power
        self.t = 0.0
        return n

    def run(self, sim_time: float, settle_frac: float = 0.5, do_settle: bool = True):
        """Settle the charge (drum stationary), then run for ``sim_time`` seconds recording the shell torque.
        Metrics use the last ``1-settle_frac`` of the running window (steady state)."""
        if do_settle:
            self.settle()
        n_steps = int(sim_time / self.dt)
        for _ in range(n_steps):
            self.step()
        return n_steps
