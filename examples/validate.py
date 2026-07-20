"""Validation harness for the mill DEM engine: a charge-shape snapshot + a speed/fill sweep vs the classical
Hogg-Fuerstenau power model. Writes PNGs to examples/out/ and prints a comparison table.

Run: .venv/Scripts/python examples/validate.py
"""
from __future__ import annotations

import math
import os

import numpy as np

from milldem import ContactModel, MillConfig, MillDEM, compute_metrics, simulate

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def hogg_fuerstenau_kw(D, L, rho_c, phi_c, J, lift_deg=35.0, c_arm=0.80):
    """The classical Hogg-Fuerstenau net power [kW] (the same form ChargeCascade ships), for cross-check."""
    if J <= 0:
        return 0.0
    R = D / 2
    Nc = 42.3 / math.sqrt(D)
    omega = 2 * math.pi * (phi_c * Nc) / 60
    M = rho_c * 1000 * (math.pi * R * R * L) * J
    arm = c_arm * R * math.sin(math.radians(lift_deg)) * max(0.0, 1 - 1.065 * J)
    return omega * M * 9.81 * arm / 1000


def snapshot(cfg: MillConfig, sim_time: float, fname: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = MillDEM(cfg, seed=42)
    sim.run(sim_time)
    m = compute_metrics(sim)
    fig, ax = plt.subplots(figsize=(5, 5))
    th = np.linspace(0, 2 * math.pi, 200)
    ax.plot(sim.R * np.cos(th), sim.R * np.sin(th), "k-", lw=1)
    ax.scatter(sim.px, sim.py, s=(sim.r * 300) ** 1.4, c="#0969da", alpha=0.75, edgecolors="none")
    ax.set_aspect("equal")
    ax.set_title(f"phi_c={cfg.phi_c} J={cfg.fill}  {m.regime}\nP={m.net_power_kw:.0f} kW toe={m.toe_deg:.0f} shoulder={m.shoulder_deg:.0f}")
    ax.axis("off")
    fig.savefig(os.path.join(OUT, fname), dpi=90, bbox_inches="tight")
    plt.close(fig)
    return m


def main():
    # a 3 m mill, coarse media for speed. rho_c ~ 4.8 for a steel ball charge.
    base = dict(diameter_m=3.0, fill=0.30, ball_diameter_m=0.18, n_lifters=8, length_m=4.5,
                contact=ContactModel(model="hooke", e=0.6, mu=0.5))
    rho_c = 4.8

    print("=== charge-shape snapshots across speed (fixed J=0.30) ===")
    for phi in (0.55, 0.75, 1.05):
        cfg = MillConfig(phi_c=phi, **base)
        m = snapshot(cfg, 1.2, f"charge_phi{int(phi*100)}.png")
        print(f"phi_c={phi}: regime={m.regime:12s} P_dem={m.net_power_kw:6.1f} kW  cat={m.frac_cataracting:.2f} cent={m.frac_centrifuging:.2f} toe={m.toe_deg:5.0f} shoulder={m.shoulder_deg:5.0f}")

    print("\n=== DEM net power vs Hogg-Fuerstenau, speed sweep (J=0.30) ===")
    print(f"{'phi_c':>6} {'P_DEM kW':>9} {'P_HF kW':>9} {'ratio':>7} {'regime':>13}")
    for phi in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        cfg = MillConfig(phi_c=phi, **base)
        m = simulate(cfg, sim_time=1.2)
        p_hf = hogg_fuerstenau_kw(cfg.diameter_m, cfg.length_m, rho_c, phi, cfg.fill)
        ratio = m.net_power_kw / p_hf if p_hf > 0 else float("nan")
        print(f"{phi:6.2f} {m.net_power_kw:9.1f} {p_hf:9.1f} {ratio:7.2f} {m.regime:>13}")

    print("\n=== power vs fill (fixed phi_c=0.75), expect a peak near J~0.4-0.5 ===")
    print(f"{'J':>5} {'P_DEM kW':>9} {'P_HF kW':>9}")
    for J in (0.15, 0.25, 0.35, 0.45):
        cfg = MillConfig(phi_c=0.75, **{**base, "fill": J})
        m = simulate(cfg, sim_time=1.2)
        p_hf = hogg_fuerstenau_kw(cfg.diameter_m, cfg.length_m, rho_c, 0.75, J)
        print(f"{J:5.2f} {m.net_power_kw:9.1f} {p_hf:9.1f}")


if __name__ == "__main__":
    main()
