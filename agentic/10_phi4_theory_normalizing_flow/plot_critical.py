"""Step 6c and 7 figures, from saved runs only (no training, no sampling).

    figures/step_6c_flow_at_criticality.png   the flow at m2 = -3.80 (next to m2_c), L = 8 and 16:
        left   distribution of M: Metropolis vs the plain flow's proposals (mode collapse) vs the Z2-symmetrized
               flow's proposals (both wells, but a gap at M ~ 0 for L = 16)
        right  flow-MCMC U4 from the first N proposals of the chain vs the Metropolis value
    figures/step_7_delta_sigma.png            Delta_sigma from chi at fixed U4 (6b chains), and the flow at L = 8

Run after: critical.py (both versions' outputs), delta_sigma.py, flow_fixed_u4.py.  python plot_critical.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from delta_sigma import OMEGAS, jk_err, replicas, target_scan
from scan import U4_ISING, binder_chi

RUNS, FIGS = Path(__file__).parent / "runs", Path(__file__).parent / "figures"
COLORS = {8: "tab:blue", 16: "tab:orange", 12: "tab:green", 24: "tab:red", 32: "tab:purple"}  # as plot_scan.py
METRO, PLAIN, SYM = "0.55", "tab:red", "tab:blue"  # Metropolis is the gray reference; flows get line styles too
scan = np.load(RUNS / "scan_fine.npz")
Ls, m2 = [int(L) for L in scan["Ls"]], scan["m2"]
I_C = 6  # 6b grid index of m2 = -3.80, where the 6c flows were run
assert np.isclose(m2[I_C], -3.80)

# ---------------------------------------------------------------- step 6c
BINS = np.linspace(-1.2, 1.2, 97)
CHECKPOINTS = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
for row, L in zip(axes, (8, 16)):
    M_met = scan[f"M_{L}"][I_C]
    plain = np.load(RUNS / f"crit_samples_L{L}.npz")
    sym = np.load(RUNS / f"crit_sym_samples_L{L}_N500000.npz")
    ax = row[0]
    ax.hist(M_met, bins=BINS, density=True, color=METRO, alpha=0.45, label="Metropolis (6b chain)")
    ax.hist(plain["M"], bins=BINS, density=True, histtype="step", lw=1.8, ls="--", color=PLAIN,
            label="plain flow proposals")
    ax.hist(sym["M"], bins=BINS, density=True, histtype="step", lw=1.8, color=SYM, label="symmetrized flow proposals")
    gap = np.abs(sym["M"]).min()
    if gap > 0.05:  # L = 16 has a hard gap; L = 8 only a dip (proposals reach M ~ 0, just too rarely)
        ax.axvspan(-gap, gap, color=SYM, alpha=0.08, lw=0)
        ax.text(0, 2e-3, f"no symmetrized\nproposal (of 500k)\nwith |M| < {gap:.2f}", ha="center", va="bottom",
                fontsize=8)
    ax.set(yscale="log", ylim=(1e-3, 6), xlabel="magnetization $M$", ylabel="density (log scale)",
           title=f"L = {L}: where the flow proposes vs where $e^{{-S}}$ puts weight")
    ax.legend(loc="upper left", fontsize=8)

    ax = row[1]
    (u, du), _ = binder_chi(M_met, L)
    ax.axhspan(u - du, u + du, color=METRO, alpha=0.35, lw=0)
    ax.axhline(u, color=METRO, lw=1.5, label=f"Metropolis: {u:.3f} ± {du:.3f}")
    est = np.array([binder_chi(sym["M"][sym["idx"][:n]], L)[0] for n in CHECKPOINTS])
    ax.errorbar(CHECKPOINTS, est[:, 0], est[:, 1], fmt="o-", ms=7, lw=2, color=SYM, capsize=3,
                label="flow MCMC (symmetrized), first N proposals")
    ax.set(xscale="log", xlabel="number of proposals N", ylabel="Binder $U_4$",
           title=f"L = {L}: " + ("converges to Metropolis" if L == 8 else "jumps toward Metropolis; error bars too small"))
    ax.legend(loc="lower right" if L == 8 else "center left", fontsize=8)
fig.suptitle("Step 6c: the flow at $m^2$ = −3.80, $\\lambda$ = 1 (next to $m^2_c$ = −3.806)\n"
             "Reverse-KL training collapses onto one well (red); symmetrizing covers both (blue) "
             "but leaves M ≈ 0 uncovered at L = 16")
fig.tight_layout()
FIGS.mkdir(exist_ok=True)
fig.savefig(FIGS / "step_6c_flow_at_criticality.png", dpi=130)
print("saved figures/step_6c_flow_at_criticality.png")

# ---------------------------------------------------------------- step 7
ds = np.load(RUNS / "delta_sigma.npz")
val = {k[4:]: ds[k][0] for k in ds.files if k.startswith("rep_")}
err = {k[4:]: jk_err(ds[k]) for k in ds.files if k.startswith("rep_")}
pairs = [tuple(p) for p in ds["pairs"]]
reps = {L: replicas(scan[f"M_{L}"], L) for L in Ls}
k = list(OMEGAS).index(1.75)
flow = np.load(RUNS / "fixed_u4_L8.npz")
(fu, dfu), (fchi, dfchi) = binder_chi(flow["M"][flow["idx"]], 8)

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ax = axes[0, 0]
for j, L in enumerate(Ls):
    ax.errorbar(m2, reps[L][0][0], jk_err(reps[L][0]), fmt="o-", ms=3, lw=1.2, color=COLORS[L], label=f"L = {L}")
    ax.plot(val["m2_star"][j], U4_ISING, "X", ms=10, color=COLORS[L], mec="white", mew=1.5)
ax.axhline(U4_ISING, color="k", ls="--", lw=1, label=f"$U_4^*$ = {U4_ISING} (2D Ising)")
ax.set(xlim=(m2[0] + 0.01, m2[-1] - 0.01), xlabel="$m^2$", ylabel="Binder $U_4$",
       title="1. each L is evaluated where $U_4 = U_4^*$ (crosses: $m^2_*(L)$)")
ax.legend(loc="lower right", fontsize=8)

ax = axes[0, 1]
Larr = np.array(Ls, dtype=float)
ax.errorbar(Larr, val["chi_star"] / Larr**1.75, err["chi_star"] / Larr**1.75, fmt="o-", ms=7, lw=2, color="k",
            capsize=3, label=r"Metropolis $\chi_*(L)$")
ax.errorbar(8 * 1.04, fchi / 8**1.75, dfchi / 8**1.75, fmt="s", ms=8, color=SYM, capsize=3,
            label=f"flow MCMC at $m^2_*(8)$ (U4 = {fu:.3f})")
ax.set(xscale="log", xticks=Ls, xticklabels=Ls, xlabel="L", ylabel=r"$\chi_* / L^{7/4}$",
       title=r"2. $\chi_*/L^{7/4}$ would be flat if $\Delta_\sigma = 1/8$ with no corrections")
ax.minorticks_off()
ax.legend(fontsize=8)

ax = axes[1, 0]
Lp = np.array([a for a, _ in pairs], dtype=float)
x = Lp**-1.75
for j, (a, b) in enumerate(pairs):
    ax.errorbar(x[j], val["dse"][j], err["dse"][j], fmt="o", ms=8, capsize=3, color=COLORS[a], label=f"L = {a}, {b}")
xs = np.linspace(0, x.max() * 1.1, 50)
ax.plot(xs, val["delta_sigma"][k] + val["a"][k] * xs, color="k", lw=1.5)
ax.errorbar(0, val["delta_sigma"][k], err["delta_sigma"][k], fmt="D", ms=8, color="k", capsize=4,
            label=rf"extrapolated: {val['delta_sigma'][k]:.3f} ± {err['delta_sigma'][k]:.3f}")
ax.axhline(0.125, color="tab:green", ls="--", lw=1.5, label=r"exact $\Delta_\sigma = 1/8$")
ax.set(xlabel=r"$L^{-7/4}$", ylabel=r"$\Delta_\sigma^{\rm eff}(L) = 1 - \frac{1}{2}\log_2[\chi_*(2L)/\chi_*(L)]$",
       ylim=(0.11, 0.215), title=r"3. extrapolate $\Delta_\sigma^{\rm eff} = \Delta_\sigma + a\,L^{-7/4}$: 2.7σ above 1/8")
ax.legend(loc="center right", fontsize=8)

ax = axes[1, 1]
targets = [0.58, 0.59, 0.60, U4_ISING]
scanned = [target_scan(m2, reps, Ls, t) for t in targets]
ext = np.array([[s[0, -1], jk_err(s)[-1]] for s in scanned])
ax.axvspan(0.590, 0.596, color=METRO, alpha=0.3, lw=0)
ax.text(0.593, 0.205, "where finite-L\n$U_4$ curves cross", ha="center", fontsize=8)
ax.errorbar(targets, ext[:, 0], ext[:, 1], fmt="o", ms=8, capsize=3, color="k")
ax.plot(U4_ISING, ext[-1, 0], "o", ms=13, mfc="none", mec="k", mew=1.5)
ax.axhline(0.125, color="tab:green", ls="--", lw=1.5, label=r"exact $\Delta_\sigma = 1/8$")
ax.set(xlabel="fixed $U_4$ target", ylabel=r"extrapolated $\Delta_\sigma$ ($\omega$ = 7/4)", ylim=(0.08, 0.22),
       title="4. systematic: dependence on the $U_4$ target (circled = main)")
ax.legend(loc="lower left", fontsize=8)
fig.suptitle(r"Step 7: $\Delta_\sigma$ from $\chi$ at fixed Binder cumulant, 6b Metropolis chains L = 8–32, $\lambda$ = 1."
             rf"  Result: $\Delta_\sigma$ = {val['delta_sigma'][k]:.3f} ± {err['delta_sigma'][k]:.3f} (exact 1/8), "
             "limited by L ≤ 32")
fig.tight_layout()
fig.savefig(FIGS / "step_7_delta_sigma.png", dpi=130)
print("saved figures/step_7_delta_sigma.png")
