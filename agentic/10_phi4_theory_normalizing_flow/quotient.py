"""Step 7c: Delta_sigma by the quotient method, from the 6b Metropolis chains (L = 8 ... 32) and the 7d cluster chains
(cluster_scan.py configs listed in CLUSTER_RUNS).

For each pair (L, 2L), at that pair's own Binder crossing m2_x, where U4(L) = U4(2L):
    Delta_sigma_eff(L) = 1 - 1/2 log2[ chi(2L, m2_x) / chi(L, m2_x) ].
Why: near criticality chi = L^(2 - 2 Delta_sigma) f(t L^(1/nu)) and U4 = g(t L^(1/nu)), t = m2 - m2_c, up to corrections.
Where the two U4 agree, both lattices sit at the same t L^(1/nu), so f cancels in the ratio and only corrections to scaling
remain, O(L^-omega). Step 7a instead fixed U4 at the infinite-volume value 0.611. The finite-L crossings sit at 0.59-0.60, so
that put small L deeper into the broken phase than large L and biased Delta_sigma_eff upward. Here no target value enters.

How U4(m2) and chi(m2) between grid points are obtained:
  * Cluster pairs (main): multi-histogram reweighting in m2 (reweight.py, step 7e) of all chains of each L, which gives smooth
    curves. The crossing is found by Brent's method inside the union-grid interval where the linear interpolants cross.
    Chains are thinned by THIN for speed; tau_int(M^2) ~ 20 updates, so this costs ~0.6% in error (check_reweight.py [4]).
  * Metropolis pairs (the 6b chains stored only M, so no reweighting), and every pair as a cross-check: U4 and log chi
    interpolated linearly in m2. The crossing is the sign change of U4(2L) - U4(L) on the union of the two grids inside
    their overlap; both interpolants are linear between union points, so this is the exact crossing of the interpolants.
    A quadratic fit to the 4 grid points nearest the crossing, per L, is a further check.

Pairs: Metropolis (8,16), (12,24), (16,32); cluster: every (L, 2L) in the cluster runs. The extrapolation uses the cluster
value for every pair the cluster chains cover (tau_int ~10x smaller, so more precise), and Metropolis for (8,16), (12,24).
Metropolis (16,32) is an independent cross-check of cluster (16,32).

Errors: joint jackknife as in 7a. Replica b drops block b from every chain at once (all chains are independent; 50 blocks
each), and the whole pipeline (reweighting fit, crossing, interpolation, extrapolation) is rerun per replica.

Extrapolation: Delta_sigma_eff(L) = Delta_sigma + a L^(-omega), weighted least squares (weights from the jackknife errors,
fixed across replicas). omega = 7/4 is the main fit (analytic background in chi); 1 and 2 show the sensitivity.

Lattice units, lam = 1.  Run: python quotient.py   (a few minutes; writes runs/quotient.npz)
"""
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from cluster_scan import load
from delta_sigma import N_BLOCKS, OMEGAS, delta_sigma_eff, jk_err, replicas
from reweight import Reweighter, u4_chi
from scan import crossing, doubling_pairs

RUNS = Path(__file__).parent / "runs"
CLUSTER_RUNS = ("prod", "prod256")  # L = 16-128, then L = 256
THIN = 2  # keep every 2nd cluster sample for the reweighting (check_reweight.py [4])
COLS = ("m2_x", "U4_x", "chi_L", "chi_2L", "dse",           # main: reweighted (cluster) or linear (Metropolis)
        "m2_x_lin", "dse_lin", "m2_x_quad", "dse_quad")    # cross-checks: linear and quadratic interpolation


def load_sources():
    """{"metropolis": {L: (m2 grid, M, None)}, "cluster": {L: (m2 grid, M, E = 1/2 sum phi^2)}}, M and E (n_m2, n_t)."""
    d = np.load(RUNS / "scan_fine.npz")
    met = {int(L): (d["m2"], d[f"M_{L}"], None) for L in d["Ls"]}
    cl = {}
    for name in CLUSTER_RUNS:
        for L, (g, M, phi2) in load(name).items():
            assert L not in cl, f"L = {L} appears in two cluster runs"
            cl[L] = (g, M, 0.5 * phi2)
    return {"metropolis": met, "cluster": cl}


def pairs_of(sources):
    return {"metropolis": [(8, 16), (12, 24), (16, 32)], "cluster": doubling_pairs(sorted(sources["cluster"]))}


def fit_set(pairs):
    """(source, L) of the pairs used in the extrapolation: Metropolis below the cluster range, cluster for the rest."""
    return [("metropolis", 8), ("metropolis", 12)] + [("cluster", L) for L, _ in pairs["cluster"]]


def interp(grid, y, x):
    """Linear interpolation of y(grid) at x, for x inside the grid (np.interp would silently clamp outside it)."""
    assert np.all((x >= grid.min()) & (x <= grid.max())), "interpolation point outside the m2 grid"
    order = np.argsort(grid)
    return np.interp(x, grid[order], y[order])


def union_grid(gA, gB):
    lo, hi = max(gA.min(), gB.min()), min(gA.max(), gB.max())
    x = np.unique(np.r_[gA, gB])
    return x[(x >= lo) & (x <= hi)]


def pair_linear(gA, UA, chiA, gB, UB, chiB):
    """Crossing of the piecewise-linear U4 interpolants of L (A) and 2L (B). Returns (m2_x, U4_x, chi_A, chi_B, i), with
    the crossing between union-grid points i and i + 1."""
    x = union_grid(gA, gB)
    m2x, i = crossing(x, interp(gB, UB, x) - interp(gA, UA, x))
    chi = [np.exp(interp(g, np.log(c), m2x)) for g, c in ((gA, chiA), (gB, chiB))]
    return m2x, interp(gA, UA, m2x), chi[0], chi[1], i


def pair_quadratic(gA, UA, chiA, gB, UB, chiB, m2_guess):
    """Same, from quadratics fitted per L to the 4 grid points nearest m2_guess. Returns (m2_x, chi_A, chi_B)."""
    fits = []
    for g, U, c in ((gA, UA, chiA), (gB, UB, chiB)):
        near = np.argsort(np.abs(g - m2_guess))[:4]
        fits.append((np.polyfit(g[near], U[near], 2), np.polyfit(g[near], np.log(c[near]), 2)))
    roots = np.roots(fits[1][0] - fits[0][0])
    roots = roots[np.isreal(roots)].real
    m2x = roots[np.argmin(np.abs(roots - m2_guess))]
    assert abs(m2x - m2_guess) < 0.02, "quadratic crossing far from the linear one"
    return m2x, np.exp(np.polyval(fits[0][1], m2x)), np.exp(np.polyval(fits[1][1], m2x))


def pair_reweighted(rwA, MA, LA, rwB, MB, LB, x):
    """Crossing of the reweighted U4 curves. The bracket is the one interval of the union grid x where U4(2L) - U4(L)
    changes sign (exactly one is required), then Brent's method. Returns (m2_x, U4_x, chi_A, chi_B).
    (Not required to be the interval of the linear crossing: at (32,64) the linear D is +0.0006 at a grid point, so the
    two estimates can fall either side of it.)"""
    D = lambda m2: u4_chi(rwB, MB, LB, m2)[0] - u4_chi(rwA, MA, LA, m2)[0]
    _, i = crossing(x, np.array([D(m2) for m2 in x]))
    m2x = brentq(D, x[i], x[i + 1], xtol=1e-10)
    (UA, chiA), (_, chiB) = u4_chi(rwA, MA, LA, m2x), u4_chi(rwB, MB, LB, m2x)
    return m2x, UA, chiA, chiB


def thinned_blocks(M, E):
    """Per chain: the thinned series split into the N_BLOCKS jackknife blocks. Returns lists [chain][block] of arrays."""
    return ([np.array_split(m[::THIN], N_BLOCKS) for m in M], [np.array_split(e[::THIN], N_BLOCKS) for e in E])


def reweighter_replica(g, Mb, Eb, r, f_start):
    """Reweighter and concatenated M for replica r (0 = all data, 1 + b = without block b of every chain)."""
    keep = lambda blocks: np.concatenate([x for j, x in enumerate(blocks) if j != r - 1])
    E = [keep(eb) for eb in Eb]
    return Reweighter(E, g, f_start), np.concatenate([keep(mb) for mb in Mb])


def quotient_replicas(sources, pairs):
    """{(source, L): (n_blocks + 1, len(COLS)) replicas, row 0 the full estimate}, one entry per pair (L, 2L)."""
    out = {}
    for src, plist in pairs.items():
        Ls = sorted({L for p in plist for L in p})
        reps = {L: replicas(sources[src][L][1], L) for L in Ls}
        for LA, LB in plist:
            gA, gB = sources[src][LA][0], sources[src][LB][0]
            rows = []
            for r in range(N_BLOCKS + 1):
                args = (gA, reps[LA][0][r], reps[LA][1][r], gB, reps[LB][0][r], reps[LB][1][r])
                m2x, U4x, chiA, chiB, _ = pair_linear(*args)
                m2q, chiAq, chiBq = pair_quadratic(*args, m2x)
                rows.append([m2x, U4x, chiA, chiB, delta_sigma_eff(chiA, chiB), m2x, delta_sigma_eff(chiA, chiB),
                             m2q, delta_sigma_eff(chiAq, chiBq)])
            out[(src, LA)] = np.array(rows)
        if src != "cluster":
            continue
        # Reweighted main estimate: one Reweighter per (L, replica), shared by the pairs that use that L.
        blocks = {L: thinned_blocks(sources[src][L][1], sources[src][L][2]) for L in Ls}
        f_full = {L: Reweighter([np.concatenate(eb) for eb in blocks[L][1]], sources[src][L][0]).f for L in Ls}
        for r in range(N_BLOCKS + 1):
            rw = {L: reweighter_replica(sources[src][L][0], *blocks[L], r, f_full[L]) for L in Ls}
            for LA, LB in plist:
                x = union_grid(sources[src][LA][0], sources[src][LB][0])
                m2x, U4x, chiA, chiB = pair_reweighted(rw[LA][0], rw[LA][1], LA, rw[LB][0], rw[LB][1], LB, x)
                out[(src, LA)][r, :5] = [m2x, U4x, chiA, chiB, delta_sigma_eff(chiA, chiB)]
            print(f"  reweighting replica {r}/{N_BLOCKS} done", end="\r", flush=True)
    return out


def fit(Ls, dse, err, omega):
    """Weighted least squares dse = Delta_sigma + a L^(-omega), weights 1/err^2. Returns (Delta_sigma, a)."""
    Ls = np.asarray(Ls, dtype=float)
    A = np.stack([np.ones_like(Ls), Ls**-omega], axis=1) / err[:, None]
    return np.linalg.lstsq(A, dse / err, rcond=None)[0]


def fit_replicas(dse_reps, Ls, omega):
    """Fit every replica with weights fixed from the full jackknife errors. dse_reps: (n_blocks + 1, n_pairs).
    Returns (fit replicas (n_blocks + 1, 2), correlated chi^2 of the full fit, degrees of freedom)."""
    err = jk_err(dse_reps)
    reps = np.array([fit(Ls, row, err, omega) for row in dse_reps])
    Ls = np.asarray(Ls, dtype=float)
    res = dse_reps[0] - (reps[0, 0] + reps[0, 1] * Ls**-omega)
    loo = dse_reps[1:]
    cov = (len(loo) - 1) * np.cov(loo.T, bias=True)  # jackknife covariance of the pairs (shared L's correlate them)
    return reps, float(res @ np.linalg.solve(cov, res)), len(Ls) - 2


def fit_table(q, fset):
    """All extrapolations reported, each as (label, fit replicas, chi^2, dof)."""
    rows = []
    for Lmin in (8, 16):
        sel = [(s, L) for s, L in fset if L >= Lmin]
        Ls = [L for _, L in sel]
        dse = np.stack([q[k][:, COLS.index("dse")] for k in sel], axis=1)
        for w in (OMEGAS if Lmin == 8 else (1.75,)):
            reps, chi2, dof = fit_replicas(dse, Ls, w)
            rows.append((f"pairs L >= {Lmin:2d}, omega = {w:4.2f}", reps, chi2, dof))
    return rows


def load_quotient():
    """Reads runs/quotient.npz back as {(source, L): replicas}."""
    qz = np.load(RUNS / "quotient.npz")
    return {(k.split("_")[0], int(k.split("_")[1])): qz[k] for k in qz["keys"]}


if __name__ == "__main__":
    sources = load_sources()
    pairs = pairs_of(sources)
    q = quotient_replicas(sources, pairs)
    print(f"\nquotient method, lam = 1, cluster runs {CLUSTER_RUNS}, joint jackknife over {N_BLOCKS} blocks")
    c = {name: i for i, name in enumerate(COLS)}
    for (src, L), reps in q.items():
        v, e = reps[0], jk_err(reps)
        how = "reweighted" if src == "cluster" else "linear    "
        print(f"  {src:10s} L={L:3d},{2 * L:3d} ({how}): m2_x = {v[0]:.4f} ± {e[0]:.4f}   U4_x = {v[1]:.4f} ± {e[1]:.4f}   "
              f"chi = {v[2]:8.2f}, {v[3]:8.2f}   Delta_sigma_eff = {v[4]:.4f} ± {e[4]:.4f}   "
              f"(linear {v[c['dse_lin']]:.4f}, quadratic {v[c['dse_quad']]:.4f})")
    for label, reps, chi2, dof in fit_table(q, fit_set(pairs)):
        print(f"  extrapolated, {label}: Delta_sigma = {reps[0, 0]:.4f} ± {jk_err(reps)[0]:.4f}   a = {reps[0, 1]:+.3f}"
              f"   chi2/dof = {chi2:.1f}/{dof}   (exact 1/8 = 0.125)")
    np.savez(RUNS / "quotient.npz", cols=COLS, keys=[f"{s}_{L}" for s, L in q], **{f"{s}_{L}": v for (s, L), v in q.items()})
    print("saved runs/quotient.npz")
