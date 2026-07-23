#!/usr/bin/env python3
"""Regenerate the validation figure for the milldem software note from the COMMITTED sweep artifact
(../data/validation_sweep.json, produced by validation_sweep.py running the real DEM engine). One figure:

  fig-validation.pdf  - DEM thin-3D-slab net power vs the classical Hogg-Fuerstenau model, across a mill-size
                        sweep (a) and a fill sweep (b), with the DEM/HF ratio annotated on each pair.

The two hand-authored schematics (fig-contact.svg, fig-slab.svg) are converted to PDF separately via svglib.

Run:  python make_figs.py
Deps: matplotlib.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

INK = "#1a1a2e"
DEMC = "#1b6ca8"
HFC = "#e07a3f"
BAND = "#eef3f0"
GRID = "#d8d8e0"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9.4, "axes.edgecolor": INK,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.8, "figure.dpi": 200,
})


def _grouped(ax, xlabels, dem, hf, ratios, xtitle, panel_title):
    x = np.arange(len(xlabels))
    w = 0.38
    b1 = ax.bar(x - w / 2, dem, w, color=DEMC, edgecolor=INK, linewidth=0.6, label="milldem (3D slab)", zorder=3)
    b2 = ax.bar(x + w / 2, hf, w, color=HFC, edgecolor=INK, linewidth=0.6, label="Hogg-Fuerstenau", zorder=3)
    top = max(max(dem), max(hf))
    for xi, r in zip(x, ratios):
        ax.text(xi, top * 1.02, f"{r:.2f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold",
                color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel(xtitle)
    ax.set_ylabel("net power [kW]")
    ax.set_ylim(0, top * 1.16)
    ax.set_title(panel_title, fontsize=9.2)
    ax.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return b1, b2


def main():
    d = json.loads((DATA / "validation_sweep.json").read_text(encoding="utf-8"))
    ss, fs = d["size_sweep"], d["fill_sweep"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    b1, b2 = _grouped(
        axa, [f"{r['D']:.0f} m" for r in ss], [r["dem_kw"] for r in ss], [r["hf_kw"] for r in ss],
        [r["ratio"] for r in ss], "mill diameter", f"(a) size sweep (J={ss[0]['J']:.2f})")
    _grouped(
        axb, [f"{r['J']:.2f}" for r in fs], [r["dem_kw"] for r in fs], [r["hf_kw"] for r in fs],
        [r["ratio"] for r in fs], "fill fraction J", f"(b) fill sweep (D={fs[0]['D']:.0f} m)")

    # the ratio-labels legend note
    axa.text(0.02, 0.97, "numbers = DEM/HF ratio", transform=axa.transAxes, fontsize=7.4,
             va="top", ha="left", style="italic", color="#555")
    fig.legend(handles=[b1, b2], loc="upper center", ncol=2, fontsize=8.4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(HERE / "fig-validation.pdf", bbox_inches="tight")
    plt.close(fig)
    ratios = [r["ratio"] for r in ss + fs]
    print(f"wrote fig-validation.pdf; {len(ratios)} configs, ratio band "
          f"[{min(ratios):.2f}, {max(ratios):.2f}], mean {sum(ratios)/len(ratios):.2f}")


if __name__ == "__main__":
    main()
