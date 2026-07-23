#!/usr/bin/env python3
"""Reproducible DEM-vs-Hogg-Fuerstenau validation sweep for the milldem software note.

Runs the validated thin-3D-slab net-power route (`milldem.simulate_power`) across a mill-size sweep and a fill
sweep, cross-checks each against the classical Hogg-Fuerstenau model, and writes a committed artifact
(`../data/validation_sweep.json`) that the figure script reads. Deterministic (fixed seed inside the engine).

Run:  python validation_sweep.py         (writes ../data/validation_sweep.json)
Deps: milldem[jit], numpy.
The Hogg-Fuerstenau reference is the exact form used by the certifying test tests/test_power3d.py.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from milldem import MillConfig, simulate_power

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(exist_ok=True)

SIM_TIME = 1.3          # simulated seconds per run (matches the certifying test)
PHI = 0.72              # fraction of critical speed for the sweeps
RHO_C = 4.8             # charge bulk density [t/m^3]
LIFT_DEG = 35.0         # HF effective lift angle


def hf_kw(D, L, phi, J, lift_deg=LIFT_DEG, c_arm=1.0, rho_c=RHO_C):
    """Classical Hogg-Fuerstenau net mill power [kW] (the cross-check reference; c_arm=1.0 = full geometric arm)."""
    R = D / 2.0
    Nc = 42.3 / math.sqrt(D)                       # critical speed [rpm]
    omega = 2 * math.pi * (phi * Nc) / 60.0        # angular velocity [rad/s]
    M = rho_c * 1000.0 * (math.pi * R * R * L) * J  # charge mass [kg]
    arm = c_arm * R * math.sin(math.radians(lift_deg)) * max(0.0, 1 - 1.065 * J)
    return omega * M * 9.81 * arm / 1000.0


def run(D, phi, J, ball, L):
    t0 = time.time()
    p = simulate_power(MillConfig(diameter_m=D, phi_c=phi, fill=J, ball_diameter_m=ball, length_m=L),
                       sim_time=SIM_TIME)
    dem = p["net_power_kw"]
    hf = hf_kw(D, L, phi, J)
    return {"D": D, "phi_c": phi, "J": J, "ball": ball, "L": L,
            "dem_kw": round(dem, 1), "hf_kw": round(hf, 1), "ratio": round(dem / hf, 3),
            "arm_m": round(p["arm_m"], 4), "n_particles": int(p["n_particles"]),
            "seconds": round(time.time() - t0, 1)}


def main():
    out = {"meta": {"sim_time_s": SIM_TIME, "phi_c": PHI, "rho_c_tpm3": RHO_C, "lift_deg": LIFT_DEG,
                    "hf_c_arm": 1.0, "note": "milldem thin-3D-slab net power vs classical Hogg-Fuerstenau"},
           "size_sweep": [], "fill_sweep": []}

    # size sweep: ball scaled with D (~0.066 D) to bound particle counts, as in the certifying test
    for D in (3.0, 4.0, 5.0, 6.0):
        ball = round(0.066 * D, 3)
        L = round(1.3 * D, 1)
        r = run(D, PHI, 0.30, ball, L)
        out["size_sweep"].append(r)
        print("size", r)

    # fill sweep at D=4 m
    for J in (0.20, 0.28, 0.35, 0.42):
        r = run(4.0, PHI, J, 0.264, 6.0)
        out["fill_sweep"].append(r)
        print("fill", r)

    # summary: the two load-bearing numbers
    ratios = [r["ratio"] for r in out["size_sweep"]]
    out["meta"]["size_ratio_min"] = min(ratios)
    out["meta"]["size_ratio_max"] = max(ratios)
    out["meta"]["size_ratio_spread"] = round(max(ratios) - min(ratios), 3)

    (DATA / "validation_sweep.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", DATA / "validation_sweep.json",
          "size-ratio spread", out["meta"]["size_ratio_spread"])


if __name__ == "__main__":
    main()
