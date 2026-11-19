"""Step 6a/6b figures from runs/scan_<name>.npz.

    figures/step_<N>_scan_<name>.png   summary: U4(m2), chi / L^(7/4), chi(2L)/chi(L) for each doubling pair,
                                       and M histograms on both sides of the transition and at it
    figures/step_<N>_scan_<name>.gif   the scan as a movie, one frame per m2: field snapshot (largest L),
                                       M histogram, and the U4 and chi-ratio curves drawn up to the current m2

Run: python plot_scan.py coarse|fine
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from scan import U4_ISING, doubling_pairs, load_scan, pair_crossing

FIGS = Path(__file__).parent / "figures"
STEP = {"coarse": "6a", "fine": "6b"}  # figure names are prefixed step_<N>_ so they sort in project order
HIST_M2 = {"coarse": (-2.75, -4.75), "fine": (-3.65, -4.0)}  # histogram panels: one each side of m2_c
COLORS = {8: "tab:blue", 16: "tab:orange", 12: "tab:green", 24: "tab:red", 32: "tab:purple"}
BINS = np.linspace(-1.6, 1.6, 61)

name = sys.argv[1]
prefix = f"step_{STEP[name]}_scan_{name}"
d, U4, dU4, chi, dchi = load_scan(name)
Ls, m2, lam = list(d["Ls"]), d["m2"], float(d["lam"])
pairs = doubling_pairs(Ls)
ratios = {(L1, L2): chi[Ls.index(L2)] / chi[Ls.index(L1)] for L1, L2 in pairs}
dratios = {p: r * np.hypot(dchi[Ls.index(p[1])] / chi[Ls.index(p[1])], dchi[Ls.index(p[0])] / chi[Ls.index(p[0])])
           for p, r in ratios.items()}
m2c, _ = pair_crossing(m2, U4, dU4, Ls, *pairs[-1])  # the largest pair has the smallest finite-size shift
L_snap = Ls[-1]
pad = 0.03 * (m2[0] - m2[-1])
XLIM = (m2[0] + pad, m2[-1] - pad)  # m2 decreases to the right: symmetric phase left, broken phase right


def draw_U4(ax, upto):
    for a, L in enumerate(Ls):
        ax.errorbar(m2[:upto], U4[a, :upto], dU4[a, :upto], fmt="o-", ms=3, color=COLORS[L], label=f"L = {L}")
    ax.axhline(2 / 3, color="gray", ls=":", lw=1)
    ax.axhline(U4_ISING, color="k", ls="--", lw=1, label=f"Ising $U_4^*$ = {U4_ISING}")
    ax.set(xlim=XLIM, ylim=(U4.min() - 0.05, 0.72), xlabel="$m^2$", ylabel="Binder $U_4$")
    ax.legend(loc="lower right", fontsize=8)


def draw_ratio(ax, upto):
    for (L1, L2), r in ratios.items():
        ax.errorbar(m2[:upto], r[:upto], dratios[L1, L2][:upto], fmt="o-", ms=3, color=COLORS[L1],
                    label=rf"$\chi_{{{L2}}} / \chi_{{{L1}}}$")
    # Labels sit on the side of the plot the data is NOT on: the ratio is ~1 at small |m2|, ~4 at large.
    for y, lab, x, ha in [(1, "symmetric: 1", m2[-1], "right"), (2 ** 1.75, "critical: $2^{7/4}$", m2[0], "left"),
                          (4, "broken: 4", m2[0], "left")]:
        ax.axhline(y, color="gray", ls=":", lw=1)
        ax.text(x, y + 0.05, lab, fontsize=8, color="gray", ha=ha)
    ax.set(xlim=XLIM, ylim=(0.5, 4.5), xlabel="$m^2$", ylabel=r"$\chi_{2L} / \chi_L$")
    ax.legend(loc="upper center", fontsize=8)  # lower right would cover the "symmetric: 1" label


def draw_hist(ax, i):
    for L in Ls:
        ax.hist(d[f"M_{L}"][i], bins=BINS, density=True, histtype="step", lw=1.5, color=COLORS[L], label=f"L = {L}")
    ax.set(xlabel="magnetization $M$", ylabel="density", yticks=[])
    ax.legend(loc="upper right", fontsize=8)


# ---------------------------------------------------------------- static summary
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
draw_U4(axes[0, 0], len(m2))
axes[0, 0].axvline(m2c, color="gray", lw=1)
axes[0, 0].set_title(f"Binder cumulant: L = {pairs[-1][0]}, {pairs[-1][1]} cross at $m^2 \\approx$ {m2c:.3f}")
# chi ~ L^(gamma/nu) f((m2 - m2c) L^(1/nu)): at m2c, chi / L^(7/4) is the same for every L if gamma/nu = 7/4.
for a, L in enumerate(Ls):
    axes[0, 1].errorbar(m2, chi[a] / L**1.75, dchi[a] / L**1.75, fmt="o-", ms=3, color=COLORS[L], label=f"L = {L}")
axes[0, 1].axvline(m2c, color="gray", lw=1)
axes[0, 1].set(yscale="log", xlim=XLIM, xlabel="$m^2$", ylabel=r"$\chi / L^{7/4}$",
               title=r"$\chi = V\langle M^2\rangle$ scaled by $L^{7/4}$: curves meet at $m^2_c$")
axes[0, 1].legend(fontsize=8)
draw_ratio(axes[0, 2], len(m2))
axes[0, 2].axvline(m2c, color="gray", lw=1)
axes[0, 2].set_title("finite-size growth of $\\chi$ when L doubles")
for ax, target in zip(axes[1], [HIST_M2[name][0], m2c, HIST_M2[name][1]]):
    i = np.argmin(np.abs(m2 - target))
    draw_hist(ax, i)
    ax.set_title(f"$m^2$ = {m2[i]:+.3f}")
sweeps = ", ".join(f"{d[f'M_{L}'].shape[1] // 1000}k" for L in Ls)
fig.suptitle(f"Step {STEP[name]}: {name} Metropolis scan of 2D $\\phi^4$ at $\\lambda$ = {lam:g} "
             f"(sweeps per point for L = {', '.join(map(str, Ls))}: {sweeps}; global flip on)")
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / f"{prefix}.png", dpi=130)
print(f"saved figures/{prefix}.png")

# ---------------------------------------------------------------- movie
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
im = axes[0, 0].imshow(d[f"snap_{L_snap}"][0], cmap="RdBu_r", vmin=-2, vmax=2)
fig.colorbar(im, ax=axes[0, 0], fraction=0.046, label=r"$\phi_x$")


def frame(i):
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.clear()
    im.set_data(d[f"snap_{L_snap}"][i])
    axes[0, 0].set(title=f"field snapshot, L = {L_snap}", xticks=[], yticks=[])
    draw_hist(axes[0, 1], i)
    axes[0, 1].set_title("distribution of $M$ over the chain")
    draw_U4(axes[1, 0], i + 1)
    draw_ratio(axes[1, 1], i + 1)
    fig.suptitle(f"$\\lambda$ = {lam:g},   $m^2$ = {m2[i]:+.3f}", fontsize=14)
    return []


frame(0)  # draw once so tight_layout sees the real labels
fig.tight_layout(rect=(0, 0, 1, 0.96))
anim = FuncAnimation(fig, frame, frames=len(m2))
anim.save(FIGS / f"{prefix}.gif", writer=PillowWriter(fps=3))
print(f"saved figures/{prefix}.gif")
