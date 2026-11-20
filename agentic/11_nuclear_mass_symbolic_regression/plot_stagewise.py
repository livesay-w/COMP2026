"""Step 3b figure: stagewise symbolic regression vs the plain search, on clean synthetic data.

    python plot_stagewise.py clean  ->  figures/step_3b_stagewise_clean.png
Reads runs/stagewise_<tag>.csv and runs/synthetic_<tag>_front.csv.
Complexity of a stagewise model ~ the sum of its terms' complexities + 3 per extra term (coefficient, x, +).
"""
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "xtick.color": INK2,
                     "ytick.color": INK2})
tag = sys.argv[1]
st = pd.read_csv(HERE / "runs" / f"stagewise_{tag}.csv")
plain = pd.read_csv(HERE / "runs" / f"synthetic_{tag}_front.csv")
cum = np.cumsum(st["complexity"]) + 3 * np.arange(len(st))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={"width_ratios": [1.05, 1]})
ax1.plot(plain["complexity"], plain["rms_held_out"], "o--", color=BLUE, lw=1.3, ms=4,
         label="plain search (step 3), on unseen nuclei")
ax1.plot(cum, st["rms_held_out"], "D-", color=MAGENTA, lw=2, ms=8, label="stagewise (step 3b), on unseen nuclei")
ax1.plot(cum, st["rms_fit"], "D:", color=MAGENTA, lw=1.2, ms=5, alpha=0.7, label="stagewise, on the fitted nuclei")
ax1.plot(37, 0.5, "v", color=ORANGE, ms=11, clip_on=False, label="planted law: exact (complexity 37)")
for s, c, r in zip(st["stage"], cum, st["rms_held_out"]):
    ax1.annotate(f"stage {s}", (c, r), (5, 6), textcoords="offset points", fontsize=8.5, color=INK)
ax1.set(yscale="log", ylim=(0.5, 700), xlim=(0, 70), xlabel="complexity (stagewise: sum over kept terms, approx.)",
        ylabel="RMS error vs the planted law (MeV, log scale)",
        title="Stagewise vs plain search: error on nuclei never fitted")
ax1.set_yticks([0.5, 1, 3, 10, 30, 100, 300])
ax1.yaxis.set_major_formatter("{x:g}")
ax1.yaxis.set_minor_formatter("")
ax1.legend(loc="upper right", fontsize=8.5, frameon=False)

ax2.axis("off")
lines = ["What each stage kept (the shape of the new term), and the held-out error after a joint re-fit:", ""]
for _, r in st.iterrows():
    status = "kept" if r["kept"] else "REJECTED (held-out error got worse)"
    lines += [f"stage {r['stage']}  ({status})"] + textwrap.wrap(r["term"], 70, initial_indent="    ",
                                                                 subsequent_indent="      ") + [
              f"    held-out {r['rms_held_out']:.1f} MeV ({100 * r['held_gain']:+.0f}%),  fitted {r['rms_fit']:.1f} MeV", ""]
lines += ["The true terms would be:  a·A,  a·A^(2/3),  a·Z²/A^(1/3),  a·(N−Z)²/A,  a·P/√A",
          "Stage 1 found volume (a·A). After that every stage kept a lookalike",
          "(log Z, 1.19^Z, polynomials in A and N): in the narrow valley they fit",
          "the leftover as well as the true terms would."]
ax2.text(0, 1, "\n".join(lines), va="top", fontsize=8.8, family="monospace", color=INK, transform=ax2.transAxes)
fig.suptitle(f"Step 3b: building the law term by term ({tag} synthetic data, generic operators, ≤ 20 nodes per stage)",
             fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(HERE / "figures" / f"step_3b_stagewise_{tag}.png", dpi=140)
print(f"saved figures/step_3b_stagewise_{tag}.png")
