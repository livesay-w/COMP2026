"""Figures for steps 1 and 2 (no fitting beyond the step-2 least squares).

    figures/step_1_ame2020_data.png   the AME2020 table: chart of nuclides (B/A, valley of stability) and B/A vs A
    figures/step_2_semf_fit.png       the semi-empirical formula: what each term contributes, the error as each
                                      textbook term is added (a term-by-term ladder, not a complexity axis), residuals,
                                      pairing

python plot_data_and_semf.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ame import load_ame
from semf import A_MIN, NAMES, design, fit, fit_sample, parity

FIGS = Path(__file__).parent / "figures"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MAGIC = [8, 20, 28, 50, 82, 126]
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})

d = load_ame()
meas = d["measured"]
BA = d["B"] / d["A"]
# Valley of stability: for each A, the measured isobar with the lowest atomic mass (check_ame.py [5]).
valley = []
for A in range(A_MIN, 241):  # beyond A ~ 240 few isobars are measured and the line jumps
    iso = np.flatnonzero((d["A"] == A) & meas)
    if len(iso):
        valley.append(iso[np.argmin(d["mass_excess"][iso])])
valley = np.array(valley)

# ---------------------------------------------------------------- step 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.15, 1]})
ax1.scatter(d["N"][~meas], d["Z"][~meas], s=5, marker="s", lw=0, color="#d6d5ce", label="AME estimate (not measured)")
sc = ax1.scatter(d["N"][meas], d["Z"][meas], c=BA[meas], s=5, marker="s", lw=0, cmap=SEQ, vmin=7.0, vmax=8.8)
ax1.plot(d["N"][valley], d["Z"][valley], color=ORANGE, lw=1.6, label="valley of stability (lightest isobar)")
ax1.plot([0, 120], [0, 120], color=MUTED, ls="--", lw=1, label="N = Z")
cb = fig.colorbar(sc, ax=ax1, extend="min", pad=0.02, shrink=0.8)
cb.set_label("binding energy per nucleon B/A (MeV)")
ax1.set(xlabel="neutron number N", ylabel="proton number Z", xlim=(0, 180), ylim=(0, 120), aspect="equal",
        title=f"Chart of nuclides: {meas.sum()} measured, {(~meas).sum()} estimated")
ax1.legend(loc="upper left", fontsize=8, frameon=False, markerscale=2.5)

ax2.axvspan(0, A_MIN - 0.5, color=GRID, alpha=0.6, lw=0)
ax2.text(A_MIN + 3, 1.0, f"A < {A_MIN}: left out of the\nstep-2 fit", fontsize=8, color=INK2)
ax2.scatter(d["A"][meas], BA[meas], s=4, lw=0, color=BLUE, alpha=0.5, label="measured nuclei")
for Z, A, name, dx, dy in [(2, 4, "⁴He", 8, 0.1), (28, 62, "⁶²Ni (peak, 8.795 MeV)", 10, 0.35),
                           (82, 208, "²⁰⁸Pb", 5, 0.45), (92, 238, "²³⁸U", 0, -0.75)]:
    i = np.flatnonzero((d["Z"] == Z) & (d["A"] == A))[0]
    ax2.annotate(name, (A, BA[i]), (A + dx, BA[i] + dy), fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax2.plot(A, BA[i], "o", ms=5, color=INK)
ax2.set(xlabel="mass number A", ylabel="B/A (MeV)", xlim=(0, 300), ylim=(0, 9.5),
        title="Binding energy per nucleon: rises fast, peaks at iron/nickel, falls slowly")
ax2.legend(loc="lower right", fontsize=8, frameon=False, markerscale=3)
fig.suptitle("Step 1: the AME2020 data (2,550 measured binding energies)", fontsize=12, x=0.01, ha="left")
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "step_1_ame2020_data.png", dpi=140)
print("saved figures/step_1_ame2020_data.png")

# ---------------------------------------------------------------- step 2
Z, N, B = fit_sample(d)
coef, res = fit(Z, N, B)
rms = lambda r: np.sqrt(np.mean(r**2))
fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

# (a) What each term contributes per nucleon, along the valley of stability.
ax = axes[0, 0]
Zv, Nv, Av = d["Z"][valley], d["N"][valley], d["A"][valley]
per_nucleon = design(Zv, Nv) * coef / Av[:, None]  # each term's contribution to B/A
cum = np.cumsum(per_nucleon[:, :4], axis=1)  # volume, then minus surface, Coulomb, asymmetry
total = per_nucleon.sum(axis=1)
ax.fill_between(Av, cum[:, 1], cum[:, 0], color=BLUE, alpha=0.22, lw=0)
ax.fill_between(Av, cum[:, 2], cum[:, 1], color=ORANGE, alpha=0.25, lw=0)
ax.fill_between(Av, cum[:, 3], cum[:, 2], color=AQUA, alpha=0.25, lw=0)
ax.plot(Av, cum[:, 0], color=INK2, lw=1.2)
ax.plot(Av, total, color=INK, lw=2, label="full formula")
ax.scatter(Av, BA[valley], s=7, color=BLUE, lw=0, label="data (valley of stability)", zorder=3)
ax.text(150, coef[0] + 0.25, f"volume: a_V = {coef[0]:.2f} MeV per nucleon", fontsize=8.5, color=INK2, ha="center")
ax.text(40, 12.6, "− surface", fontsize=9, color=INK)
ax.text(190, 10.9, "− Coulomb", fontsize=9, color=INK)
ax.text(225, 8.2, "− asymmetry", fontsize=9, color=INK)
ax.set(xlabel="mass number A", ylabel="contribution to B/A (MeV)", ylim=(6.5, 16.5), xlim=(A_MIN, 260),
       title="(a) Building B/A: volume binding minus three losses")
ax.legend(loc="lower left", fontsize=8, frameon=False)

# (b) RMS error as each term is added (the hand-made complexity ladder).
ax = axes[0, 1]
ladder = [rms(fit(Z, N, B, k)[1]) for k in range(1, 6)]
x = np.arange(1, 6)
ax.plot(x, ladder, "o-", color=BLUE, lw=2, ms=8)
for xi, r in zip(x, ladder):
    ax.annotate(f"{r:.1f} MeV", (xi, r), (6, 6), textcoords="offset points", fontsize=9, color=INK)
ax.annotate("biggest gain: Coulomb and asymmetry\nonly pay off together", (4, ladder[3]), (1.0, 5),
            fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
ax.set(yscale="log", ylim=(2, 90), xticks=x, xlim=(0.7, 5.5), ylabel="RMS error over 2,484 nuclei (MeV, log scale)",
       title="(b) Error as each textbook term is added (not PySR complexity)")
ax.set_yticks([2, 3, 5, 10, 20, 30, 50])
ax.yaxis.set_major_formatter("{x:g}")
ax.yaxis.set_minor_formatter("")
ax.set_xticklabels(["volume", "+ surface", "+ Coulomb", "+ asymmetry", "+ pairing"])

# (c) Residuals vs N: 3.3 MeV scatter with structure at the magic numbers (studied in step 7).
ax = axes[1, 0]
for m in MAGIC[2:]:
    ax.axvline(m, color=YELLOW, lw=1.2, alpha=0.8)
    ax.text(m, 25.5, str(m), ha="center", fontsize=8, color=INK2)
ax.axhline(0, color=MUTED, lw=1)
ax.scatter(N, res, s=5, lw=0, color=BLUE, alpha=0.55)
ax.annotate("largest misses: light neutron-rich\nB and C isotopes (A = 16-22)", (16, 24), (30, 18), fontsize=8.5,
            color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.text(0.98, 0.04, "vertical lines: magic neutron numbers (step 7)", transform=ax.transAxes, ha="right",
        fontsize=8, color=INK2)
ax.set(xlabel="neutron number N", ylabel="data − formula (MeV)", ylim=(-9, 28),
       title=f"(c) What the formula misses: RMS {rms(res):.2f} MeV, with structure")

# (d) Pairing: without the pairing term, even-even nuclei sit above the fit and odd-odd below.
ax = axes[1, 1]
_, res4 = fit(Z, N, B, n_terms=4)
p = parity(Z, N)
bins = np.linspace(-12, 12, 49)
for k, color, label in [(1, BLUE, "even Z, even N"), (-1, ORANGE, "odd Z, odd N")]:
    ax.hist(res4[p == k], bins=bins, histtype="step", lw=2, color=color, density=True,
            label=f"{label}: mean {res4[p == k].mean():+.2f} MeV")
    ax.axvline(res4[p == k].mean(), color=color, lw=1.2, ls="--")
ax.text(0.02, 0.97, f"with the pairing\nterm (a_P = {coef[4]:.1f} MeV/√A)\nthe means become\n"
        f"{res[p == 1].mean():+.2f} and {res[p == -1].mean():+.2f} MeV", transform=ax.transAxes, va="top",
        fontsize=8.5, color=INK2)
ax.set(xlabel="data − formula without pairing (MeV)", ylabel="fraction of nuclei (density)",
       title="(d) Why the pairing term is there")
ax.legend(loc="center right", fontsize=8, frameon=False)

coeffs = ",  ".join(f"{n} = {c:.2f}" for n, c in zip(NAMES, coef))
fig.suptitle(f"Step 2: least-squares fit of the semi-empirical mass formula to {len(B)} nuclei (A ≥ {A_MIN})\n"
             f"{coeffs} MeV", fontsize=12, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(FIGS / "step_2_semf_fit.png", dpi=140)
print("saved figures/step_2_semf_fit.png")
