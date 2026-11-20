"""Step 3 elbow plot: PySR's Pareto front on synthetic data made from the step-2 formula.

    python plot_synthetic.py clean   ->  figures/step_3_synthetic_clean_elbow.png   (reads runs/synthetic_clean_front.csv)

For comparison, the planted formula built term by term (truncated to 1..5 terms and refitted to the same data), at its
complexity in PySR's node count with the generic operators of sr.py (powers as x^c):
    a A (3);  + a A^c -> 9;  + a Z^2 A^c -> 19;  + a (Nn - Z)^2 / A -> 29;  + a P A^c -> 37 (the full law, exact).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ame import load_ame
from semf import TERMS, design, fit, fit_sample
from synthetic import wide_grid

HERE = Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "xtick.color": INK2,
                     "ytick.color": INK2})
tag = sys.argv[1]
front = pd.read_csv(HERE / "runs" / f"synthetic_{tag}_front.csv")
cx, rms, held = front["complexity"].to_numpy(), front["rms_truth"].to_numpy(), front["rms_held_out"].to_numpy()

# The elbow: the largest drop between neighbouring formulas after the trivial first step (constant -> a*A),
# per unit of complexity. Its lower end is the "base of the elbow".
i = np.argmax(-np.diff(np.log(rms))[1:] / np.diff(cx)[1:]) + 2

Z, N, B = fit_sample(load_ame())
coef = fit(Z, N, B)[0]
if tag == "wide":
    Z, N = wide_grid(len(B))
y = design(Z, N) @ coef
ladder_cx = np.array([3, 9, 19, 29, 37])
ladder_rms = np.array([np.sqrt(np.mean(fit(Z, N, y, n)[1] ** 2)) for n in range(1, 6)])
FLOOR = 0.3

fig, ax = plt.subplots(figsize=(11.5, 7))
ax.plot(cx, rms, "o-", color=BLUE, lw=2, ms=6, label="PySR: best formula at each complexity (error on the fitted nuclei)")
ax.plot(cx, held, "o--", color=AQUA, lw=1.3, ms=4, label="same formulas on 1,003 nuclei PySR never saw")
if tag == "wide":
    ax.plot(cx, front["rms_valley"], "o:", color="#6250d6", lw=1.3, ms=4,
            label="same formulas on the 2,484 real (valley) nuclei")
ax.plot(ladder_cx[:4], ladder_rms[:4], "s--", color=ORANGE, lw=1.5, ms=7, label="planted formula, built term by term")
ax.plot(ladder_cx[4], FLOOR, "v", color=ORANGE, ms=11, clip_on=False)
ax.annotate("full planted formula:\nexact everywhere\n(complexity 37)", (37, FLOOR), (38.5, 0.42), fontsize=9,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
for c, r, t in zip(ladder_cx[1:4], ladder_rms[1:4], TERMS[1:4]):
    ax.annotate(f"+ {t}", (c, r), (6, 6 if t != "asymmetry" else -14), textcoords="offset points", fontsize=8.5,
                color=ORANGE)

ax.annotate(f"elbow: complexity {cx[i - 1]} → {cx[i]}, {rms[i - 1]:.1f} → {rms[i]:.1f} MeV\n"
            f"base of the elbow:  {front['equation'][i]}\n"
            f"on unseen nuclei: {held[i]:.1f} MeV, so a lookalike, not the planted law",
            (cx[i], rms[i]), (1, 0.42), fontsize=8.5, color=INK, ha="left",
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
best_upto = lambda c: rms[cx <= c].min()  # PySR's best with at most c nodes
ax.annotate(f"at the planted law's complexities PySR is worse:\n≤ 29 nodes: {best_upto(29):.2f} vs {ladder_rms[3]:.2f} "
            f"MeV;  ≤ 37: {best_upto(37):.2f} vs 0", (29, best_upto(29)), (27, 9), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.set(yscale="log", ylim=(FLOOR, 1000), xlim=(0, 46), xlabel="complexity (number of nodes in the formula)",
       ylabel="RMS error vs the planted law (MeV, log scale)")
ax.set_yticks([0.3, 1, 3, 10, 30, 100, 300, 1000])
ax.yaxis.set_major_formatter("{x:g}")
ax.yaxis.set_minor_formatter("")
ax.legend(loc="upper right", fontsize=9, frameon=False)
if tag == "wide":
    title = ("Step 3c: the same blind search on a WIDE grid of nuclei (N/Z = 0.6-2.0, vs 1.0-1.55 in the valley)\n"
             "Spreading the data out did not help: still lookalikes, and the largest formulas blow up on unseen nuclei")
else:
    title = (f"Step 3: the complexity elbow on {tag} synthetic data (generic operators, no hints; 3,000 iterations, "
             "~11 min on 10 cores)\nPySR finds an elbow, but the formulas past it are lookalikes: they do worse on "
             "nuclei they never saw")
ax.set_title(title, fontsize=11, loc="left")
fig.tight_layout()
out = f"step_3c_synthetic_{tag}_elbow.png" if tag == "wide" else f"step_3_synthetic_{tag}_elbow.png"
fig.savefig(HERE / "figures" / out, dpi=140)
print(f"saved figures/{out};  elbow {cx[i - 1]} -> {cx[i]};  ladder {np.round(ladder_rms, 3)}")
