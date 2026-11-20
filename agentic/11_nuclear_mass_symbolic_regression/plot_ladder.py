"""Step 3d figure: rediscovery of noise-free planted laws of increasing size, for two generic operator sets.

Reads whichever runs/synthetic_ladder_<ops>_k<k>_front.csv exist (synthetic_ladder.py), so a stopped run still plots.
python plot_ladder.py  ->  figures/step_3d_size_ladder.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).parent
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "xtick.color": INK2,
                     "ytick.color": INK2})
# Same as synthetic_ladder.PLANTED_SIZE (not imported: importing it would start a search).
PLANTED_SIZE = {"power": {1: 3, 2: 9, 3: 19, 4: 29}, "calculator": {1: 3, 2: 9, 3: 17, 4: 26, 5: 33}}
LAWS = ["volume", "+ surface", "+ Coulomb", "+ asymmetry", "+ pairing"]
TITLES = {"power": "x^c (exponent tuned as a number), exp, log",
          "calculator": "calculator set: sqrt, cbrt, square, cube, 1/x, exp, log"}
EXACT = 1e-3

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
for ax, ops in zip(axes, ("power", "calculator")):
    done = 0
    for k, size in PLANTED_SIZE[ops].items():
        path = HERE / "runs" / f"synthetic_ladder_{ops}_k{k}_front.csv"
        if not path.exists():
            continue
        done += 1
        f = pd.read_csv(path)
        hit = f[f["rms"] < EXACT]
        found = f"found exactly at {hit['complexity'].iloc[0]}" if len(hit) else f"not found, best {f['rms'].min():.2g} MeV"
        ax.plot(f["complexity"], f["rms"], "o-", color=COLORS[k - 1], lw=1.8, ms=4.5,
                label=f"{LAWS[k - 1]} ({size} nodes): {found}")
        ax.axvline(size, color=COLORS[k - 1], lw=1, ls=":")
    ax.axhline(EXACT, color=INK2, lw=1, ls="--")
    ax.text(1, 1.4e-3, "'exact': 0.001 MeV", fontsize=8.5, color=INK2)
    stopped = "" if done == len(PLANTED_SIZE[ops]) else f"  (run stopped after {done} of {len(PLANTED_SIZE[ops])} laws)"
    ax.set(yscale="log", ylim=(1e-6, 3e3), xlim=(0, 41), xlabel="complexity (number of nodes in the formula)",
           title=f"{TITLES[ops]}{stopped}")
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, title="planted law (dotted: its size)",
              title_fontsize=8.5)
axes[0].set_ylabel("RMS error vs the planted law (MeV, log scale)")
fig.suptitle("Step 3d: how big a law can the blind search rediscover? Noise-free data at the real nuclei, 1,000 "
             "iterations per law.\nWith either operator set only the 1-term law is found: log-based lookalikes "
             "(A·log A) imitate A^(2/3)", fontsize=11.5, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(HERE / "figures" / "step_3d_size_ladder.png", dpi=140)
print("saved figures/step_3d_size_ladder.png")
