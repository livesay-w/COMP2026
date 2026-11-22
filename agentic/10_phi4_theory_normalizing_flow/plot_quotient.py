"""Step 7c-7f figures, from saved runs only (no new simulations).

    figures/step_7c_quotient.png     why the quotient method: on the 6b Metropolis data the fixed target U4* = 0.611 lies
                                     above every pair's own crossing (left); Delta_sigma_eff at the fixed target (7a) vs at
                                     the pair's own crossing (7c), same chains (right)
    figures/step_7d_cluster.png      the cluster algorithm: tau_int(M^2) vs L, Metropolis vs cluster (left); cluster and
                                     Metropolis U4 at the 14 shared (L, m2) points, independent chains (right)
    figures/step_7e_reweighting.png  multi-histogram reweighting: the E histograms of the L = 128 chains overlap (left);
                                     at (32,64), linear interpolation vs the reweighted curves (middle); Delta_sigma_eff of
                                     every cluster pair by method (right)
    figures/step_7f_delta_sigma.png  the result: Delta_sigma_eff(L) for every stage of step 7 (left); extrapolation of the
                                     cluster pairs in L^(-7/4) (middle); U4 and m2 at the pair crossings (right)

Colors: one fixed color per lattice size L in every panel (the 8 hues of the reference palette in its validated order,
L = 8 ... 256), each L also with its own marker. Method comparisons use ink tones and marker shapes, not hues.
Errors: jackknife over 50 blocks, as in the analysis. Lattice units, lam = 1.

Run after: quotient.py (with CLUSTER_RUNS including "prod256") and delta_sigma.py.  python plot_quotient.py  (~1 min)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from delta_sigma import jk_err, replicas
from phi4 import tau_int
from quotient import COLS, THIN, fit_replicas, fit_set, load_quotient, load_sources, pairs_of
from reweight import Reweighter, u4_chi
from scan import U4_ISING, binder_chi

FIGS = Path(__file__).parent / "figures"
RUNS = Path(__file__).parent / "runs"
# One color and marker per L, in the reference palette's validated categorical order.
LCOL = {8: "#2a78d6", 12: "#eb6834", 16: "#1baf7a", 24: "#eda100", 32: "#e87ba4", 64: "#008300", 128: "#6250d6",
        256: "#e34948"}
LMARK = {8: "o", 12: "s", 16: "^", 24: "v", 32: "D", 64: "P", 128: "X", 256: "*"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
EXACT = 0.125
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "legend.frameon": False,
                     "errorbar.capsize": 2.5, "figure.facecolor": "white"})
col = {name: i for i, name in enumerate(COLS)}

sources = load_sources()
pairs = pairs_of(sources)
q = load_quotient()
assert ("cluster", 128) in q, "runs/quotient.npz has no (128,256) pair: run quotient.py with prod256 first"
val = {k: v[0] for k, v in q.items()}
err = {k: jk_err(v) for k, v in q.items()}
met = sources["metropolis"]
m2_6b = met[8][0]
reps_met = {L: replicas(met[L][1], L) for L in sorted(met)}
ds7a = np.load(RUNS / "delta_sigma.npz")


def exact_line(ax, label=r"exact $\Delta_\sigma = 1/8$"):
    ax.axhline(EXACT, color=INK, ls="--", lw=1, label=label, zorder=1)


# ---------------------------------------------------------------- 7c: fixed target vs own crossing
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
for L in sorted(met):
    U = reps_met[L][0]
    a1.errorbar(m2_6b, U[0], jk_err(U), fmt=LMARK[L] + "-", ms=4, lw=1.2, color=LCOL[L], label=f"L = {L}")
a1.axhline(U4_ISING, color=INK, ls="--", lw=1, label=f"fixed target $U_4^*$ = {U4_ISING} (7a)")
for j, L in enumerate(ds7a["Ls"]):  # 7a: where each L meets the fixed target
    a1.plot(ds7a["rep_m2_star"][0, j], U4_ISING, LMARK[L], ms=8, color=LCOL[L], mec=INK, mew=0.8)
label_offset = {8: (8, -14), 12: (10, -18), 16: (-48, 10)}  # (12,24) and (16,32) cross close together
for LA, LB in pairs["metropolis"]:  # 7c: each pair's own crossing
    v = val[("metropolis", LA)]
    a1.plot(v[col["m2_x"]], v[col["U4_x"]], "o", ms=11, mfc="none", mec=INK, mew=1.4)
    a1.annotate(f"({LA},{LB})", (v[col["m2_x"]], v[col["U4_x"]]), xytext=label_offset[LA], textcoords="offset points",
                fontsize=8, color=INK2)
a1.set(xlabel=r"$m^2$", ylabel=r"Binder $U_4$", xlim=(-3.66, -3.99), ylim=(0.45, 0.67),
       title="6b Metropolis: pair crossings (open circles) sit below the 7a target")
a1.legend(fontsize=8, loc="lower right", ncol=2)

Lp = np.array([8, 12, 16])
dse7a, e7a = ds7a["rep_dse"][0], jk_err(ds7a["rep_dse"])
dse7c = np.array([val[("metropolis", L)][col["dse"]] for L in Lp])
e7c = np.array([err[("metropolis", L)][col["dse"]] for L in Lp])
a2.errorbar(Lp * 0.97, dse7a, e7a, fmt="s", ms=7, mfc="white", color=MUTED, label=r"7a: at fixed $U_4^*$ = 0.611")
a2.errorbar(Lp * 1.03, dse7c, e7c, fmt="o", ms=7, color=INK, label="7c: at each pair's own crossing")
exact_line(a2)
a2.set(xscale="log", xticks=Lp, xticklabels=[f"({L},{2 * L})" for L in Lp], xlabel="pair (L, 2L)",
       ylabel=r"$\Delta_\sigma^{\rm eff}(L) = 1 - \frac{1}{2}\log_2[\chi(2L)/\chi(L)]$",
       title="Same chains, no target: every pair moves toward 1/8")
a2.minorticks_off()
a2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIGS / "step_7c_quotient.png", dpi=130)
print("saved figures/step_7c_quotient.png")


# ---------------------------------------------------------------- 7d: cluster algorithm
cl = sources["cluster"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
Lm = np.array(sorted(met))
i_met = int(np.argmin(np.abs(m2_6b + 3.806)))  # 6b point nearest the 6b crossing
tau_m = np.array([tau_int(met[L][1][i_met].astype(float) ** 2) for L in Lm])
m2x_pair = {LA: val[("cluster", LA)][col["m2_x"]] for LA, _ in pairs["cluster"]}
m2x_pair[256] = m2x_pair[128]
Lc = np.array(sorted(cl))
tau_c = np.array([tau_int(cl[L][1][int(np.argmin(np.abs(cl[L][0] - m2x_pair[L])))] ** 2) for L in Lc])
zm, cm = np.polyfit(np.log(Lm[Lm >= 12]), np.log(tau_m[Lm >= 12]), 1)
zc, cc = np.polyfit(np.log(Lc), np.log(tau_c), 1)
xs = np.array([6, 300])
a1.plot(xs, np.exp(cm) * xs**zm, color=MUTED, lw=1, ls=":", zorder=1)
a1.plot(xs, np.exp(cc) * xs**zc, color=INK, lw=1, ls=":", zorder=1)
a1.plot(Lm, tau_m, "s", ms=7, mfc="white", color=MUTED, label=f"local Metropolis (6b), per sweep: $\\tau \\sim L^{{{zm:.1f}}}$")
a1.plot(Lc, tau_c, "o", ms=7, color=INK, label=f"cluster (Metropolis + SW), per update: $\\tau \\sim L^{{{zc:.2f}}}$")
a1.annotate(f"L = 256: τ = {tau_c[-1]:.0f}", (256, tau_c[-1]), xytext=(-20, 18), textcoords="offset points",
            ha="right", fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a1.set(xscale="log", yscale="log", xlabel="L", ylabel=r"$\tau_{\rm int}(M^2)$ at the crossing", xlim=(6, 320),
       title="Critical slowing down is gone (one update ≈ 2 sweeps of cost)")
a1.set_xticks([8, 16, 32, 64, 128, 256], ["8", "16", "32", "64", "128", "256"])
a1.minorticks_off()
a1.legend(fontsize=8.5, loc="upper left")

zs = []
for L in (16, 32):
    g, M = cl[L][0], cl[L][1]
    Uc = np.array([binder_chi(M[j], L)[0] for j in range(len(g))])
    idx = [int(np.flatnonzero(np.isclose(m2_6b, m2))[0]) for m2 in g]
    Um = np.array([binder_chi(met[L][1][i], L)[0] for i in idx])
    zs += list((Uc[:, 0] - Um[:, 0]) / np.hypot(Uc[:, 1], Um[:, 1]))
    a2.errorbar(g + 0.002, Um[:, 0], Um[:, 1], fmt=LMARK[L], ms=7, mfc="white", color=LCOL[L], lw=1,
                label=f"L = {L}, Metropolis (6b)")
    a2.errorbar(g - 0.002, Uc[:, 0], Uc[:, 1], fmt=LMARK[L], ms=6, color=LCOL[L], lw=1, label=f"L = {L}, cluster")
a2.text(0.03, 0.95, f"{len(zs)} independent comparisons of U4: max |z| = {np.max(np.abs(zs)):.2f}",
        transform=a2.transAxes, fontsize=8.5, color=INK2, va="top")
a2.set(xlabel=r"$m^2$", ylabel=r"Binder $U_4$", xlim=(-3.715, -3.885),
       title="Two algorithms, independent chains, same U4 (open: Metropolis)")
a2.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIGS / "step_7d_cluster.png", dpi=130)
print("saved figures/step_7d_cluster.png")


# ---------------------------------------------------------------- 7e: reweighting
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
g128, E128 = cl[128][0], cl[128][2]
blues = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#0d366b"]  # ramp 250-700
order = np.argsort(-g128)  # least negative m2 (lightest) first
bins = np.linspace(E128.min(), E128.max(), 80)
for c_, j in zip(blues, order):
    a1.hist(E128[j], bins=bins, density=True, histtype="step", lw=1.4, color=c_)
gap = np.max(np.abs(np.diff(E128.mean(axis=1))) / E128.std(axis=1)[:-1])
a1.text(0.02, 0.97, f"9 chains, light → dark:\n$m^2$ = {g128[order[0]]:.3f} … {g128[order[-1]]:.3f}\n"
        f"neighbouring means ≤ {gap:.2f} σ apart", transform=a1.transAxes, va="top", fontsize=8.5, color=INK2)
a1.set(xlabel=r"$E = \frac{1}{2}\sum_x \phi_x^2$", ylabel="density", yticks=[],
       ylim=(0, a1.get_ylim()[1] * 1.3), title="L = 128: neighbouring E histograms overlap")

LA, LB = 32, 64
x = np.linspace(-3.8, -3.84, 161)
for L in (LA, LB):
    g, M, E = cl[L]
    rw = Reweighter([e[::THIN] for e in E], g)
    Mall = np.concatenate([m[::THIN] for m in M])
    U_rw = np.array([u4_chi(rw, Mall, L, m2)[0] for m2 in x if g.min() <= m2 <= g.max()])
    xin = np.array([m2 for m2 in x if g.min() <= m2 <= g.max()])
    U_pts = reps_U = replicas(M, L)[0]
    a2.errorbar(g, U_pts[0], jk_err(U_pts), fmt=LMARK[L], ms=7, color=LCOL[L], lw=0, elinewidth=1, label=f"L = {L} chains")
    a2.plot(g, U_pts[0], color=LCOL[L], lw=1, ls="--")
    a2.plot(xin, U_rw, color=LCOL[L], lw=2)
v = val[("cluster", LA)]
a2.axvline(v[col["m2_x_lin"]], color=MUTED, lw=1, ls="--", label=f"linear crossing {v[col['m2_x_lin']]:.4f}")
a2.axvline(v[col["m2_x"]], color=INK, lw=1, label=f"reweighted crossing {v[col['m2_x']]:.4f}")
a2.set(xlabel=r"$m^2$", ylabel=r"Binder $U_4$", xlim=(-3.80, -3.84), ylim=(0.56, 0.66),
       title="(32,64): dashed = linear interpolation, solid = reweighted")
a2.legend(fontsize=8, loc="lower right")

Lc_pairs = np.array([L for L, _ in pairs["cluster"]])
for k, (name, mk, fc, c_, dx) in enumerate((("dse_lin", "s", "white", MUTED, 0.94), ("dse_quad", "^", "white", INK2, 1.0),
                                             ("dse", "o", INK, INK, 1.06))):
    y = np.array([val[("cluster", L)][col[name]] for L in Lc_pairs])
    e = np.array([err[("cluster", L)][col[name]] for L in Lc_pairs])
    label = {"dse_lin": "linear interpolation (7c)", "dse_quad": "quadratic interpolation",
             "dse": "multi-histogram reweighting (7e, main)"}[name]
    a3.errorbar(Lc_pairs * dx, y, e, fmt=mk, ms=7, mfc=fc, color=c_, label=label)
exact_line(a3)
a3.set(xscale="log", xticks=Lc_pairs, xticklabels=[f"({L},{2 * L})" for L in Lc_pairs], xlabel="pair (L, 2L)",
       ylabel=r"$\Delta_\sigma^{\rm eff}(L)$", title="Same chains, three ways to read off the crossing")
a3.minorticks_off()
a3.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGS / "step_7e_reweighting.png", dpi=130)
print("saved figures/step_7e_reweighting.png")


# ---------------------------------------------------------------- 7f: the result
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.15, 1, 0.9]})
Lpm = np.array([8, 12, 16])
a1.errorbar(Lpm * 0.95, dse7a, e7a, fmt="s", ms=7, mfc="white", color=MUTED, label="7a  fixed $U_4^*$, Metropolis")
a1.errorbar(Lpm * 1.0, dse7c, e7c, fmt="D", ms=6, mfc="white", color=INK2, label="7c  own crossing, Metropolis")
ylin = np.array([val[("cluster", L)][col["dse_lin"]] for L in Lc_pairs if L <= 64])
elin = np.array([err[("cluster", L)][col["dse_lin"]] for L in Lc_pairs if L <= 64])
a1.errorbar(Lc_pairs[Lc_pairs <= 64] * 0.95, ylin, elin, fmt="^", ms=7, mfc="white", color=INK2,
            label="7c/d  own crossing, cluster, linear interp.")
yrw = np.array([val[("cluster", L)][col["dse"]] for L in Lc_pairs])
erw = np.array([err[("cluster", L)][col["dse"]] for L in Lc_pairs])
a1.errorbar(Lc_pairs[:-1] * 1.05, yrw[:-1], erw[:-1], fmt="o", ms=7, color=INK, label="7e  cluster, reweighted")
a1.errorbar(Lc_pairs[-1:] * 1.05, yrw[-1:], erw[-1:], fmt="*", ms=13, color=LCOL[256],
            label=f"7f  (128,256): {yrw[-1]:.4f} ± {erw[-1]:.4f}")
exact_line(a1)
a1.set(xscale="log", xticks=[8, 12, 16, 32, 64, 128], xticklabels=["8", "12", "16", "32", "64", "128"],
       xlabel="L of the pair (L, 2L)", ylabel=r"$\Delta_\sigma^{\rm eff}(L)$", ylim=(0.115, 0.215),
       title="Step 7, stage by stage")
a1.minorticks_off()
a1.legend(fontsize=8, loc="upper right")

fset = [k for k in fit_set(pairs) if k[0] == "cluster"]  # cluster pairs only, L >= 16
Lf = np.array([L for _, L in fset], dtype=float)
dse_reps = np.stack([q[k][:, col["dse"]] for k in fset], axis=1)
fr, chi2, dof = fit_replicas(dse_reps, Lf, 1.75)
fe = jk_err(fr)
xx = np.linspace(0, 16**-1.75 * 1.05, 100)
band = np.array([r[0] + r[1] * xx for r in fr[1:]])
band_err = np.sqrt((len(band) - 1) * np.mean((band - band.mean(axis=0)) ** 2, axis=0))
a2.fill_between(xx, fr[0, 0] + fr[0, 1] * xx - band_err, fr[0, 0] + fr[0, 1] * xx + band_err, color=GRID, lw=0)
a2.plot(xx, fr[0, 0] + fr[0, 1] * xx, color=INK2, lw=1.2,
        label=f"fit $\\Delta_\\sigma + a L^{{-7/4}}$: χ²/dof = {chi2:.1f}/{dof}")
for L, y, e in zip(Lc_pairs, yrw, erw):
    a2.errorbar(L**-1.75, y, e, fmt=LMARK[L], ms=8 if L != 128 else 9, color=LCOL[L], label=f"({L},{2 * L})")
a2.errorbar(0, fr[0, 0], fe[0], fmt="o", ms=8, mfc="white", color=INK, mew=1.5,
            label=f"extrapolated: {fr[0, 0]:.4f} ± {fe[0]:.4f} (stat)")
exact_line(a2)
a2.set(xlabel=r"$L^{-7/4}$", ylabel=r"$\Delta_\sigma^{\rm eff}(L)$", xlim=(-0.0004, 16**-1.75 * 1.08),
       title="Cluster pairs, extrapolated in $L^{-7/4}$")
a2.legend(fontsize=7.5, loc="upper left")
ins = a2.inset_axes([0.55, 0.17, 0.42, 0.38])  # zoom on the two largest pairs and the extrapolation
ins.fill_between(xx, fr[0, 0] + fr[0, 1] * xx - band_err, fr[0, 0] + fr[0, 1] * xx + band_err, color=GRID, lw=0)
ins.plot(xx, fr[0, 0] + fr[0, 1] * xx, color=INK2, lw=1.2)
for L, y, e in zip(Lc_pairs, yrw, erw):
    ins.errorbar(L**-1.75, y, e, fmt=LMARK[L], ms=7, color=LCOL[L])
ins.errorbar(0, fr[0, 0], fe[0], fmt="o", ms=7, mfc="white", color=INK, mew=1.5)
ins.axhline(EXACT, color=INK, ls="--", lw=1)
ins.set(xlim=(-0.0001, 0.0009), ylim=(0.1235, 0.1310), xticks=[0, 0.0004, 0.0008], yticks=[0.124, 0.126, 0.128, 0.130])
ins.tick_params(labelsize=7)
ins.set_facecolor("white")

invL = 1 / Lc_pairs
U4x = np.array([val[("cluster", L)][col["U4_x"]] for L in Lc_pairs])
eU4x = np.array([err[("cluster", L)][col["U4_x"]] for L in Lc_pairs])
for L, xv, y, e in zip(Lc_pairs, invL, U4x, eU4x):
    a3.errorbar(xv, y, e, fmt=LMARK[L], ms=8, color=LCOL[L])
    a3.annotate(f"({L},{2 * L})", (xv, y), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK2)
a3.axhline(U4_ISING, color=INK, ls="--", lw=1, label=f"2D Ising $U_4^*$ = {U4_ISING}")
a3.set(xlabel="1 / L", ylabel=r"$U_4$ at the pair crossing", xlim=(0, 0.068), ylim=(0.5945, 0.6125),
       title="Binder crossing converges to the Ising value")
a3.legend(fontsize=8, loc="lower left")
m2x = [val[("cluster", L)][col["m2_x"]] for L in Lc_pairs]
a3.text(0.97, 0.55, "crossing $m^2_x$:\n" + "\n".join(f"({L},{2 * L}): {m:.4f}" for L, m in zip(Lc_pairs, m2x)),
        transform=a3.transAxes, ha="right", va="top", fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(FIGS / "step_7f_delta_sigma.png", dpi=130)
print("saved figures/step_7f_delta_sigma.png")
