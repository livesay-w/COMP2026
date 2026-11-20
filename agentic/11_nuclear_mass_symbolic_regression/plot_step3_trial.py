"""The first complexity elbow: PySR's Pareto front on clean synthetic data (step-3 trial, target B).

Reads runs/step3_trial_B_front.csv (the hall of fame written by `python search_trial.py 1000 B`).
For comparison, the planted formula's own ladder: the step-2 formula truncated to 1..5 terms, refitted to the same
synthetic data, placed at its complexity in PySR's node count with the same operators:
    a A (3);  + a cbrt(A)^2 -> 9;  + a Z^2/cbrt(A) -> 17;  + a (Nn - Z)^2/A -> 26;  + a P/sqrt(A) -> 33 (exact).

python plot_step3_trial.py  ->  figures/step_3_trial_elbow.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ame import load_ame
from semf import TERMS, design, fit, fit_sample

HERE = Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "xtick.color": INK2,
                     "ytick.color": INK2})

front = pd.read_csv(HERE / "runs" / "step3_trial_B_front.csv")
cx, rms = front["Complexity"].to_numpy(), np.sqrt(front["Loss"].to_numpy())
# Elbow = the knee: the point furthest below the straight line from the first to the last point, in
# (complexity, log RMS). (The steepest single drop is always the trivial first step, constant -> a*A.)
line = np.interp(cx, [cx[0], cx[-1]], np.log([rms[0], rms[-1]]))
k = np.argmax(line - np.log(rms))

Z, N, B = fit_sample(load_ame())
y = design(Z, N) @ fit(Z, N, B)[0]  # the planted law
ladder_cx = [3, 9, 17, 26, 33]
ladder_rms = [np.sqrt(np.mean(fit(Z, N, y, n)[1] ** 2)) for n in range(1, 6)]
FLOOR = 0.3  # bottom of the log axis; the full planted formula fits exactly (RMS ~1e-11) and is drawn there

fig, ax = plt.subplots(figsize=(11, 6.5))
ax.plot(cx, rms, "o-", color=BLUE, lw=2, ms=6, label="PySR: best formula found at each complexity")
ax.plot(ladder_cx[:4], ladder_rms[:4], "s--", color=ORANGE, lw=1.5, ms=7, label="planted formula, built term by term")
ax.plot(33, FLOOR, "v", color=ORANGE, ms=11, clip_on=False)
ax.annotate("full planted formula:\nexact (RMS 0) at complexity 33", (33, FLOOR), (34, 0.4), fontsize=9,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
for c, r, t in zip(ladder_cx[1:4], ladder_rms[1:4], TERMS[1:4]):
    ax.annotate(f"+ {t}", (c, r), (6, -14 if t == "asymmetry" else 8), textcoords="offset points", fontsize=8.5,
                color=ORANGE)

# The second bend: the largest drop between neighbours after the trivial first step, and where it levels off.
drop = np.argmax(-np.diff(np.log(rms))[1:] / np.diff(cx)[1:]) + 1  # index of the point just before the cliff
end = drop + 2  # two points later, where the curve flattens again
ax.annotate(f"knee 1, complexity {cx[k]}: {rms[k]:.1f} MeV\n{front['Equation'][k]}\n≈ volume + surface",
            (cx[k], rms[k]), (cx[k] + 1.5, 60), fontsize=8.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.annotate(f"cliff, complexity {cx[drop]} → {cx[end]}: {rms[drop]:.1f} → {rms[end]:.1f} MeV\n"
            f"terms in Z² enter (≈ Coulomb + asymmetry together)", (cx[drop + 1], rms[drop + 1]), (19, 5.5),
            fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.annotate(f"beyond {cx[end]}: slow gains,\n{rms[end]:.1f} → {rms[-1]:.2f} MeV over {cx[-1] - cx[end]} more nodes",
            (cx[-1], rms[-1]), (31, 2.4), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.set(yscale="log", ylim=(FLOOR, 800), xlim=(0, 41), xlabel="complexity (number of nodes in the formula)",
       ylabel="RMS error vs the planted law (MeV, log scale)")
ax.set_yticks([0.3, 1, 3, 10, 30, 100, 300])
ax.yaxis.set_major_formatter("{x:g}")
ax.yaxis.set_minor_formatter("")
ax.legend(loc="upper right", fontsize=9, frameon=False)
ax.set_title("Step 3 trial: the first complexity elbow (clean synthetic data, 1000 iterations, ~5.5 min)\n"
             "PySR's curve stays above the planted formula: it found lookalike formulas, not the planted one",
             fontsize=11, loc="left")
fig.tight_layout()
fig.savefig(HERE / "figures" / "step_3_trial_elbow.png", dpi=140)
print(f"saved figures/step_3_trial_elbow.png;  elbow at complexity {cx[k]};  planted ladder RMS {np.round(ladder_rms, 3)}")
