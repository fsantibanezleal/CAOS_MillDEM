"""CAOS_MillDEM, a cross-platform 2D soft-sphere DEM engine for tumbling-mill charge motion + power.

No C++ toolchain, no WSL: pure numpy with an optional numba JIT and an optional torch-CUDA path. The engine
simulates a rotating mill disc slice (Govender et al. 2015 reduced setup) and reports the DEM charge shape,
motion regime, and net power via the van Nierop (2001) torque route.

Quick start::

    from milldem import simulate, MillConfig, ContactModel
    m = simulate(MillConfig(diameter_m=5.0, phi_c=0.75, fill=0.30), sim_time=2.0)
    print(m.net_power_kw, m.regime, m.toe_deg, m.shoulder_deg)
"""
from __future__ import annotations

from .contact import ContactModel
from .engine import G, MillConfig, MillDEM
from .metrics import MillMetrics, compute_metrics

__version__ = "0.01.000"  # X.XX.XXX display form (versioning.md); PEP 440 normalizes to 0.1.0
__all__ = ["ContactModel", "MillConfig", "MillDEM", "MillMetrics", "compute_metrics", "simulate", "G"]


def simulate(cfg: MillConfig, sim_time: float = 2.0, seed: int = 42, settle_frac: float = 0.5) -> MillMetrics:
    """Run a full mill DEM simulation and return the settled-state metrics.

    ``sim_time`` seconds of simulated mill operation; the last ``1 - settle_frac`` of the run is used for the
    metrics (the first half lets the charge settle from its initial placement into steady motion).
    """
    sim = MillDEM(cfg, seed=seed)
    sim.run(sim_time, settle_frac=settle_frac)
    return compute_metrics(sim, settle_frac=settle_frac)
