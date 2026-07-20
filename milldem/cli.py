"""``milldem`` command-line interface.

    milldem run   --D 5.0 --phi 0.75 --J 0.30 --ball 0.10 --time 2.0 [--json out.json] [--frames frames.npz]
    milldem power --D 5.0 --phi 0.75 --J 0.30 --ball 0.10 --L 6.0 [--time 1.5] [--json out.json]

``run`` gives the fast 2D qualitative charge-shape / regime read; ``power`` runs the validated thin-3D slab and
reports the size-consistent net power (within ~10-20% of Hogg-Fuerstenau, see docs/VALIDATION.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

import numpy as np

from . import ContactModel, MillConfig, MillDEM, __version__, compute_metrics, simulate_power


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="milldem", description="Soft-sphere tumbling-mill DEM (2D charge shape + "
                                                            "validated thin-3D-slab power).")
    p.add_argument("--version", action="version", version=f"milldem {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a 2D simulation and report the settled charge-shape / regime metrics")
    r.add_argument("--D", type=float, default=5.0, help="mill diameter [m]")
    r.add_argument("--L", type=float, default=6.0, help="mill length [m] (for net-power scaling)")
    r.add_argument("--phi", type=float, default=0.75, help="fraction of critical speed")
    r.add_argument("--J", type=float, default=0.30, help="fractional filling")
    r.add_argument("--ball", type=float, default=0.10, help="top ball diameter [m]")
    r.add_argument("--size-ratio", type=float, default=1.0, help="min/max ball diameter (1 = mono)")
    r.add_argument("--lifters", type=int, default=8, help="number of lifter bars")
    r.add_argument("--e", type=float, default=0.6, help="coefficient of restitution")
    r.add_argument("--mu", type=float, default=0.5, help="sliding friction")
    r.add_argument("--model", choices=["hooke", "hertz"], default="hooke")
    r.add_argument("--time", type=float, default=2.0, help="simulated time [s]")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--json", type=str, default=None, help="write the metrics to this JSON path")
    r.add_argument("--frames", type=str, default=None, help="write a compact frame series (npz) to this path")
    r.add_argument("--frame-stride", type=int, default=200, help="record one frame every N steps")

    pw = sub.add_parser("power", help="run the validated thin-3D slab and report the net power [kW]")
    pw.add_argument("--D", type=float, default=5.0, help="mill diameter [m]")
    pw.add_argument("--L", type=float, default=6.0, help="mill length [m]")
    pw.add_argument("--phi", type=float, default=0.75, help="fraction of critical speed")
    pw.add_argument("--J", type=float, default=0.30, help="fractional filling")
    pw.add_argument("--ball", type=float, default=0.24, help="top ball diameter [m]")
    pw.add_argument("--size-ratio", type=float, default=1.0, help="min/max ball diameter (1 = mono)")
    pw.add_argument("--slab", type=float, default=None, help="slab thickness [m] (default 4 ball diameters)")
    pw.add_argument("--time", type=float, default=1.5, help="simulated time [s] after settling")
    pw.add_argument("--seed", type=int, default=42)
    pw.add_argument("--json", type=str, default=None, help="write the result to this JSON path")

    a = p.parse_args(argv)
    if a.cmd == "power":
        cfg = MillConfig(diameter_m=a.D, length_m=a.L, phi_c=a.phi, fill=a.J, ball_diameter_m=a.ball,
                         size_ratio=a.size_ratio)
        res = simulate_power(cfg, sim_time=a.time, slab_thickness_m=a.slab, seed=a.seed)
        out = {"config": {"D": a.D, "L": a.L, "phi_c": a.phi, "J": a.J, "ball": a.ball, "seed": a.seed},
               "net_power_kw": res["net_power_kw"], "arm_m": res["arm_m"], "n_particles": res["n_particles"]}
        print(json.dumps(out, indent=2))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        return 0
    if a.cmd == "run":
        cfg = MillConfig(
            diameter_m=a.D, length_m=a.L, phi_c=a.phi, fill=a.J, ball_diameter_m=a.ball,
            size_ratio=a.size_ratio, n_lifters=a.lifters,
            contact=ContactModel(model=a.model, e=a.e, mu=a.mu),
        )
        sim = MillDEM(cfg, seed=a.seed)
        n_steps = int(a.time / sim.dt)
        frames = []
        for i in range(n_steps):
            sim.step()
            if a.frames and (i % a.frame_stride == 0):
                frames.append(np.stack([sim.px, sim.py], axis=1).astype(np.float32))
        m = compute_metrics(sim)
        md = asdict(m)
        print(json.dumps(md, indent=2))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump({"config": {"D": a.D, "L": a.L, "phi_c": a.phi, "J": a.J, "ball": a.ball,
                                       "model": a.model, "e": a.e, "mu": a.mu, "seed": a.seed},
                           "metrics": md}, f, indent=2)
        if a.frames and frames:
            np.savez_compressed(a.frames, frames=np.stack(frames), radii=sim.r.astype(np.float32),
                                R=sim.R, n_frames=len(frames))
            print(f"wrote {len(frames)} frames to {a.frames}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
