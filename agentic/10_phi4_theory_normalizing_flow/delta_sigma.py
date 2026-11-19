"""Step 7a: Delta_sigma from chi(L) at fixed Binder cumulant, using the 6b Metropolis chains (no new simulations).

At criticality chi ~ L^(gamma/nu) = L^(2 - 2 Delta_sigma), where Delta_sigma is the scaling dimension of the spin
field (exact 2D Ising: 1/8). Instead of sitting at an estimated m2_c, each L is evaluated where U4(L) equals the
universal 2D Ising value U4* = 0.61069: this point drifts to m2_c as L grows, and it removes the need to know m2_c.

    1. For each L: m2*(L) and chi*(L) where U4 = U4*. log chi is interpolated against U4 (smoother than against
       m2), linearly between the bracketing grid points; a quadratic fit to the 4 nearest points is the
       systematic check (check_delta_sigma.py).
    2. Pairs (L, 2L):  Delta_sigma_eff(L) = 1 - 1/2 log2[chi*(2L) / chi*(L)].
    3. Extrapolate Delta_sigma_eff(L) = Delta_sigma + a L^(-omega). omega = 7/4 is the main fit: the analytic
       background in chi (chi = A L^(7/4) + B) corrects the ratio at relative order L^(-7/4). omega = 1, 2 show the
       sensitivity to that assumption.

Errors: joint jackknife. Replica b drops block b from every chain at once (all chains are independent), and the whole
pipeline (interpolation, ratios, fit) is rerun per replica, so correlations between U4 and chi are kept.

Lattice units, lam = 1.  Run: python delta_sigma.py   (needs runs/scan_fine.npz; writes runs/delta_sigma.npz)
"""
from pathlib import Path

import numpy as np

from scan import U4_ISING, crossing, doubling_pairs

RUNS = Path(__file__).parent / "runs"
N_BLOCKS = 50   # same blocking as scan.binder_chi; 6b showed every block spans >= 10 tau_int
OMEGAS = (1.0, 1.75, 2.0)  # correction exponents tried in the extrapolation; 1.75 is the main one


def replicas(M, L, n_blocks=N_BLOCKS):
    """U4 and chi for every chain of one L, as (n_blocks + 1, n_m2): row 0 the full estimate, row 1 + b without block b.

    M: (n_m2, n_sweeps) magnetization series, one independent chain per row."""
    M = M.astype(np.float64)
    m2b, m4b = (np.stack([b.mean(axis=1) for b in np.array_split(M**p, n_blocks, axis=1)], axis=1) for p in (2, 4))
    m2, m4 = (np.vstack([b.mean(axis=1), (b.sum(axis=1, keepdims=True) - b).T / (n_blocks - 1)]) for b in (m2b, m4b))
    return 1 - m4 / (3 * m2**2), L * L * m2


def jk_err(x):
    """Jackknife error from replicas x: (n_blocks + 1, ...), row 0 the full estimate."""
    loo = x[1:]
    return np.sqrt((len(loo) - 1) * np.mean((loo - loo.mean(axis=0)) ** 2, axis=0))


def at_u4(m2, U4, chi, target=U4_ISING):
    """Where U4 = target: (m2*, chi* from linear interpolation of log chi vs U4, chi* from a quadratic fit to the
    4 nearest points)."""
    m2_star, i = crossing(m2, U4 - target)
    assert 1 <= i <= len(m2) - 3, "need a grid point on each side of the bracket for the quadratic check"
    y = np.log(chi)
    t = (target - U4[i]) / (U4[i + 1] - U4[i])
    lin = y[i] + t * (y[i + 1] - y[i])
    near = slice(i - 1, i + 3)
    quad = np.polyval(np.polyfit(U4[near], y[near], 2), target)
    return m2_star, np.exp(lin), np.exp(quad)


def delta_sigma_eff(chi_L, chi_2L):
    """1 - 1/2 log2[chi(2L)/chi(L)]. Equals Delta_sigma exactly if chi = c L^(2 - 2 Delta_sigma)."""
    return 1 - 0.5 * np.log2(chi_2L / chi_L)


def extrapolate(Ls, dse, omega):
    """Least-squares fit dse = Delta_sigma + a L^(-omega). Returns (Delta_sigma, a)."""
    Ls = np.asarray(Ls, dtype=float)
    A = np.stack([np.ones_like(Ls), Ls**-omega], axis=1)
    return np.linalg.lstsq(A, dse, rcond=None)[0]


def analyse(m2, U4, chi, Ls):
    """The whole pipeline for one replica. U4, chi: {L: (n_m2,)}. Returns a dict of arrays."""
    star = np.array([at_u4(m2, U4[L], chi[L]) for L in Ls])  # (nL, 3): m2*, chi* linear, chi* quadratic
    m2s, chi_lin, chi_quad = (dict(zip(Ls, col)) for col in star.T)
    pairs = doubling_pairs(Ls)
    dse = np.array([delta_sigma_eff(chi_lin[a], chi_lin[b]) for a, b in pairs])
    dse_quad = np.array([delta_sigma_eff(chi_quad[a], chi_quad[b]) for a, b in pairs])
    fits = np.array([extrapolate([a for a, _ in pairs], dse, w) for w in OMEGAS])  # (n_omega, 2)
    return dict(m2_star=star[:, 0], chi_star=star[:, 1], chi_star_quad=star[:, 2], dse=dse, dse_quad=dse_quad,
                delta_sigma=fits[:, 0], a=fits[:, 1])


def target_scan(m2, reps, Ls, target, omega=1.75):
    """Delta_sigma_eff of every pair and the extrapolated Delta_sigma at another U4 target (a systematic check).
    reps: {L: (U4 replicas, chi replicas)} from replicas(). Returns (n_blocks + 1, n_pairs + 1) replicas."""
    pairs = doubling_pairs(Ls)
    out = []
    for r in range(len(reps[Ls[0]][0])):
        chi_star = {L: at_u4(m2, reps[L][0][r], reps[L][1][r], target)[1] for L in Ls}
        dse = np.array([delta_sigma_eff(chi_star[a], chi_star[b]) for a, b in pairs])
        out.append(np.r_[dse, extrapolate([a for a, _ in pairs], dse, omega)[0]])
    return np.array(out)


if __name__ == "__main__":
    d = np.load(RUNS / "scan_fine.npz")
    Ls, m2 = [int(L) for L in d["Ls"]], d["m2"]
    reps = {L: replicas(d[f"M_{L}"], L) for L in Ls}
    out = [analyse(m2, {L: reps[L][0][r] for L in Ls}, {L: reps[L][1][r] for L in Ls}, Ls) for r in range(N_BLOCKS + 1)]
    res = {k: np.array([o[k] for o in out]) for k in out[0]}  # each (n_blocks + 1, ...)
    val, err = {k: v[0] for k, v in res.items()}, {k: jk_err(v) for k, v in res.items()}

    print(f"fixed U4 = {U4_ISING} (2D Ising), lam = 1, 6b chains, joint jackknife over {N_BLOCKS} blocks")
    for j, L in enumerate(Ls):
        print(f"  L={L:2d}: m2* = {val['m2_star'][j]:.4f} ± {err['m2_star'][j]:.4f}   "
              f"chi* = {val['chi_star'][j]:7.2f} ± {err['chi_star'][j]:.2f}   (quadratic: {val['chi_star_quad'][j]:7.2f})")
    for j, (a, b) in enumerate(doubling_pairs(Ls)):
        print(f"  L={a:2d},{b:2d}: Delta_sigma_eff = {val['dse'][j]:.4f} ± {err['dse'][j]:.4f}   "
              f"(quadratic: {val['dse_quad'][j]:.4f})")
    for k, w in enumerate(OMEGAS):
        print(f"  extrapolated, omega = {w:4.2f}: Delta_sigma = {val['delta_sigma'][k]:.4f} ± {err['delta_sigma'][k]:.4f}"
              f"   a = {val['a'][k]:+.3f}   (exact 1/8 = 0.125)")
    np.savez(RUNS / "delta_sigma.npz", Ls=Ls, pairs=doubling_pairs(Ls), omegas=OMEGAS, u4_target=U4_ISING,
             **{f"rep_{k}": v for k, v in res.items()})
    print("saved runs/delta_sigma.npz")
