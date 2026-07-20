"""Thin-3D-slab DEM: a slab of the mill of axial thickness ``w`` (a few ball diameters) with PERIODIC axial
(z) boundaries. This captures the 3D packing and the axial force chains that carry the charge lift, which a
single 2D disc slice cannot. The drum is an infinite cylinder in x-y; z is the axial direction, periodic over
[0, w). Net power for the full mill scales by ``length_m / w``.

Why this fixes the 2D power-vs-size limitation: in 2D the lift is set by the wall friction and gravity (a
size-independent absolute height), so the CoM torque arm does not scale with the mill radius R. In 3D the
force chains transmit the wall drive through the packed charge, and the charge lifts to a shoulder angle that
is roughly speed/fill-dependent (dynamic similarity), so the arm scales with R as it physically must.

Same soft-sphere contact model as the 2D engine (``contact.py``); numba-JIT hot loop with a numpy fallback.
"""
from __future__ import annotations

import math

import numpy as np

from .contact import ContactModel
from .engine import G, MillConfig

try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrap(f):
            return f
        return wrap if not args else args[0]


@njit(cache=True, fastmath=True)
def _resolve3d(px, py, pz, vx, vy, vz, r, m, cell_start, cell_items, ncx, ncy, ncz, cs, xmin, ymin,
               w, kn, kt, gscale, mu, R, omega, dt):
    n = px.shape[0]
    fx = np.zeros(n); fy = np.zeros(n); fz = np.zeros(n)
    shell_torque = 0.0
    for i in range(n):
        fy[i] -= m[i] * G

    # particle-particle, 3D grid with z-periodicity
    for i in range(n):
        cx = int((px[i] - xmin) / cs)
        cy = int((py[i] - ymin) / cs)
        cz = int(pz[i] / cs)
        for dgx in range(-1, 2):
            for dgy in range(-1, 2):
                for dgz in range(-1, 2):
                    gx = cx + dgx; gy = cy + dgy; gz = (cz + dgz) % ncz
                    if gx < 0 or gy < 0 or gx >= ncx or gy >= ncy:
                        continue
                    cell = (gx * ncy + gy) * ncz + gz
                    for idx in range(cell_start[cell], cell_start[cell + 1]):
                        j = cell_items[idx]
                        if j <= i:
                            continue
                        dx = px[j] - px[i]; dy = py[j] - py[i]
                        dz = pz[j] - pz[i]
                        # minimum-image in z (periodic)
                        if dz > 0.5 * w:
                            dz -= w
                        elif dz < -0.5 * w:
                            dz += w
                        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                        overlap = r[i] + r[j] - dist
                        if overlap <= 0.0 or dist < 1e-12:
                            continue
                        nx = dx / dist; ny = dy / dist; nz = dz / dist
                        m_eff = m[i] * m[j] / (m[i] + m[j])
                        rvx = vx[j] - vx[i]; rvy = vy[j] - vy[i]; rvz = vz[j] - vz[i]
                        vn = -(rvx * nx + rvy * ny + rvz * nz)
                        gamma_n = gscale * math.sqrt(kn / m_eff)
                        fn = kn * overlap + m_eff * gamma_n * vn
                        if fn < 0.0:
                            fn = 0.0
                        # tangential (full Coulomb for shear strength, impulse-capped)
                        tvx = rvx - (rvx * nx + rvy * ny + rvz * nz) * nx
                        tvy = rvy - (rvx * nx + rvy * ny + rvz * nz) * ny
                        tvz = rvz - (rvx * nx + rvy * ny + rvz * nz) * nz
                        tvm = math.sqrt(tvx * tvx + tvy * tvy + tvz * tvz)
                        ftx = 0.0; fty = 0.0; ftz = 0.0
                        if tvm > 1e-9:
                            ft = min(mu * fn, m_eff * tvm / dt)
                            ftx = -ft * tvx / tvm; fty = -ft * tvy / tvm; ftz = -ft * tvz / tvm
                        fix = -(fn * nx) + ftx; fiy = -(fn * ny) + fty; fiz = -(fn * nz) + ftz
                        fx[i] += fix; fy[i] += fiy; fz[i] += fiz
                        fx[j] -= fix; fy[j] -= fiy; fz[j] -= fiz

    # particle-wall (cylindrical shell at R in x-y) + shell torque
    for i in range(n):
        rad = math.sqrt(px[i] * px[i] + py[i] * py[i])
        overlap = rad + r[i] - R
        if overlap > 0.0 and rad > 1e-12:
            nx = px[i] / rad; ny = py[i] / rad
            m_eff = m[i]
            vn = -(vx[i] * nx + vy[i] * ny)
            gamma_n = gscale * math.sqrt(kn / m_eff)
            fn = kn * overlap + m_eff * gamma_n * vn
            if fn < 0.0:
                fn = 0.0
            wvx = -omega * py[i]; wvy = omega * px[i]
            rvx = wvx - vx[i]; rvy = wvy - vy[i]
            tvx = rvx - (rvx * nx + rvy * ny) * nx
            tvy = rvy - (rvx * nx + rvy * ny) * ny
            tvm = math.sqrt(tvx * tvx + tvy * tvy)
            fpx = -fn * nx; fpy = -fn * ny
            if tvm > 1e-9:
                ft = min(mu * fn, m_eff * tvm / dt)
                fpx += ft * tvx / tvm; fpy += ft * tvy / tvm
            fx[i] += fpx; fy[i] += fpy
            shell_torque += px[i] * (-fpy) - py[i] * (-fpx)

    return fx, fy, fz, shell_torque


class MillDEM3D:
    """Thin-3D-slab mill DEM. Slab thickness defaults to 4 top-ball diameters."""

    def __init__(self, cfg: MillConfig, slab_thickness_m: float | None = None, seed: int = 42):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.R = cfg.radius()
        self.omega = cfg.omega()
        rmax = cfg.ball_diameter_m / 2.0
        self.w = slab_thickness_m if slab_thickness_m is not None else 4.0 * cfg.ball_diameter_m
        self._build(rmax)
        kn = self._auto_kn()
        self.kn = kn
        self.kt = cfg.contact.kt_ratio * kn
        m_eff = float(self.m.min()) / 2.0
        self.dt = 0.02 * math.pi * math.sqrt(m_eff / kn)
        e = min(max(cfg.contact.e, 1e-4), 0.999); lne = math.log(e)
        self.gscale = -2.0 * lne / math.sqrt(math.pi ** 2 + lne ** 2)
        self._v_max = max(3.0 * self.omega * self.R, 2.0 * math.sqrt(2 * G * self.R))
        self._torque = []

    def _build(self, rmax):
        cfg = self.cfg
        rmin = rmax * cfg.size_ratio
        disc_area = math.pi * self.R ** 2
        # solid volume target: J * disc_area * w  (bed occupies J of the cross-section over the slab thickness)
        target_vol = cfg.fill * disc_area * self.w
        radii = []; vol = 0.0
        while vol < target_vol:
            rr = self.rng.uniform(rmin, rmax); radii.append(rr)
            vol += (4.0 / 3.0) * math.pi * rr ** 3
        radii.sort(reverse=True)
        r = np.array(radii); n = r.shape[0]
        # place on a 3D grid in the lower drum, no overlaps. The x-y grid stays loose (1.12 spacing): a denser/hex
        # x-y lattice crystallizes the coarse (few-balls-across) charge and kills the natural cascade, breaking the
        # size-consistency of the power. The z direction is PERIODIC, so it is tiled by whole layers across the full
        # slab (no wall margins) at the same loose pitch: this holds the target particle count for realistic fills
        # (a single margined 3-layer stack saturated below ~0.30 fill, collapsing every higher fill to one charge)
        # without touching the x-y dynamics that set the lift.
        step = 2.0 * rmax * 1.12
        gx = np.arange(-self.R + rmax + 0.01, self.R - rmax - 0.01, step)
        gy = np.arange(-self.R + rmax + 0.01, self.R - rmax - 0.01, step)
        nz = max(1, int(round(self.w / step)))
        gz = (np.arange(nz) + 0.5) * (self.w / nz)   # whole layers tiling the periodic slab, no margins
        pts = [(x, y, z) for z in gz for y in gy for x in gx if x * x + y * y < (self.R - rmax) ** 2]
        pts.sort(key=lambda p: p[1])
        if len(pts) < n:
            r = r[:len(pts)]; n = len(pts)
        px = np.zeros(n); py = np.zeros(n); pz = np.zeros(n)
        for i in range(n):
            px[i], py[i], pz[i] = pts[i]
        self.px, self.py, self.pz, self.r = px, py, pz, r
        self.vx = np.zeros(n); self.vy = np.zeros(n); self.vz = np.zeros(n)
        self.m = cfg.rho_ball * (4.0 / 3.0) * math.pi * r ** 3
        self.n = n

    def _auto_kn(self):
        m_max = float(self.m.max()); m_min = float(self.m.min()); r_min = float(self.r.min())
        v_imp = max(self.omega * self.R, math.sqrt(2 * G * self.R))
        target = 0.04 * r_min
        kn_stab = m_max * v_imp ** 2 / target ** 2
        kn_cap = (m_min / 2.0) * (math.pi / 6e-5) ** 2
        return min(max(kn_stab, 100.0 * m_max * G / target), kn_cap)

    def _grid(self):
        cs = self.cfg.ball_diameter_m * 1.05
        xmin = -self.R - cs; ymin = -self.R - cs
        ncx = int((2 * self.R + 2 * cs) / cs) + 1; ncy = ncx
        ncz = max(1, int(self.w / cs))
        cxi = np.clip(((self.px - xmin) / cs).astype(np.int64), 0, ncx - 1)
        cyi = np.clip(((self.py - ymin) / cs).astype(np.int64), 0, ncy - 1)
        czi = np.clip((self.pz / cs).astype(np.int64), 0, ncz - 1)
        cid = (cxi * ncy + cyi) * ncz + czi
        order = np.argsort(cid, kind="stable").astype(np.int64)
        counts = np.bincount(cid, minlength=ncx * ncy * ncz)
        start = np.zeros(ncx * ncy * ncz + 1, dtype=np.int64); start[1:] = np.cumsum(counts)
        return start, order, ncx, ncy, ncz, cs, xmin, ymin

    def step(self):
        c = self.cfg.contact
        start, order, ncx, ncy, ncz, cs, xmin, ymin = self._grid()
        fx, fy, fz, tq = _resolve3d(self.px, self.py, self.pz, self.vx, self.vy, self.vz, self.r, self.m,
                                    start, order, ncx, ncy, ncz, cs, xmin, ymin, self.w,
                                    self.kn, self.kt, self.gscale, c.mu, self.R, self.omega, self.dt)
        self.vx += fx / self.m * self.dt; self.vy += fy / self.m * self.dt; self.vz += fz / self.m * self.dt
        d = math.exp(-self.cfg.bg_damping * self.dt)
        self.vx *= d; self.vy *= d; self.vz *= d
        sp2 = self.vx ** 2 + self.vy ** 2 + self.vz ** 2
        hot = sp2 > self._v_max ** 2
        if np.any(hot):
            s = self._v_max / np.sqrt(sp2[hot]); self.vx[hot] *= s; self.vy[hot] *= s; self.vz[hot] *= s
        self.px += self.vx * self.dt; self.py += self.vy * self.dt; self.pz += self.vz * self.dt
        self.pz %= self.w  # periodic axial wrap
        self._torque.append(tq)

    def settle(self, t=None):
        if t is None:
            t = 4.0 * math.sqrt(2 * self.R / G)
        om = self.omega; self.omega = 0.0
        for _ in range(int(t / self.dt)):
            self.step(); self.vx *= 0.995; self.vy *= 0.995; self.vz *= 0.995
        self.omega = om; self._torque.clear()

    def run(self, t, do_settle=True):
        if do_settle:
            self.settle()
        for _ in range(int(t / self.dt)):
            self.step()

    def net_power_kw(self, settle_frac=0.4):
        h = np.asarray(self._torque); k = int(len(h) * settle_frac)
        tq = float(np.mean(h[k:])) if len(h) > k else 0.0   # torque of the slab (thickness w)
        N = self.omega / (2 * math.pi)
        n_slabs = self.cfg.length_m / self.w
        return 2 * math.pi * abs(tq) * N * n_slabs / 1000.0

    def arm_m(self):
        return float(np.average(self.px, weights=self.m))
