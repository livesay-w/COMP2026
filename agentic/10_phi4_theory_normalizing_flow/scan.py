"""Step 6: scan m2 at fixed lam with local Metropolis (+ global flip) to locate the phase transition.

For each L and each m2 on the grid, run an independent chain and keep the magnetization series M(t)
and the final configuration. Binder U4 and chi = V<M^2> are computed from M afterwards (check_scan.py,
plot_scan.py), so nothing here depends on how the analysis is done.

Lattice units (a = 1). Each (L, m2) chain gets its own seed [seed, L, i] so points are independent
and any single point can be reproduced on its own. Chains run in parallel over CPU cores.

Run: python -u scan.py coarse|fine  (configs below; results go to runs/scan_<name>.npz)
Each chain is also kept in runs/scan_<name>/L<L>_i<i>.npz; rerunning after a crash only runs the missing chains.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from phi4 import jackknife, metropolis_magnetization

RUNS = Path(__file__).parent / "runs"
U4_ISING = 0.61069  # Binder crossing of the 2D Ising class on periodic squares (Salas & Sokal 2000)

CONFIGS = {
    # 6a: wide grid across the rough transition region (m2 ~ -3.5 to -4 at lam = 1), two sizes.
    # n_sweeps: tau_int(M^2) peaks at ~240 sweeps for L = 16 at the crossing (critical slowing down);
    # 50 jackknife blocks of >= 10 tau need >= 120k sweeps (check_scan.py [3]). 40k failed that check.
    "coarse": dict(Ls=(8, 16), m2=np.linspace(-2.0, -5.5, 29), lam=1.0,
                   n_sweeps={8: 150000, 16: 150000}, n_therm={8: 5000, 16: 5000}, step=1.5, seed=0),
    # 6b: narrow grid around the 6a crossing (m2 ~ -3.82), five sizes. Sweeps scale with the measured
    # tau_int(M^2) ~ 219 (L/16)^2.2 so every L gets 50 blocks of >= ~12 tau; thermalise >= ~20 tau.
    # L = 24 at 350k sweeps failed check [3] (max tau = 730 -> 9.6 tau per block), so it now runs 500k.
    # seed=1, so the L = 8, 16 points shared with 6a are independent re-measurements (check_scan.py [6]).
    "fine": dict(Ls=(8, 12, 16, 24, 32), m2=np.linspace(-3.65, -4.0, 15), lam=1.0,
                 n_sweeps={8: 60000, 12: 100000, 16: 150000, 24: 500000, 32: 600000},
                 n_therm={8: 2000, 12: 4000, 16: 5000, 24: 12000, 32: 20000}, step=1.5, seed=1),
}


def point_file(name, L, i):
    return RUNS / f"scan_{name}" / f"L{L}_i{i:02d}.npz"


def run_point(name, L, i, m2_i, lam, n_sweeps, n_therm, step, seed):
    """One independent chain, written to its own file as soon as it finishes, so a crash loses only the chains
    still running. A chain whose file already exists is skipped (resume after an interrupted scan)."""
    out = point_file(name, L, i)
    if out.exists():
        print(f"L={L:3d}  m2={m2_i:+.4f}  already done ({out.name})", flush=True)
        return
    t0 = time.time()
    rng = np.random.default_rng([seed, L, i])
    M, snap, acc = metropolis_magnetization(L, m2_i, lam, n_sweeps, n_therm, step, rng, flip=True)
    np.savez(out, M=M.astype(np.float32), snap=snap, acc=acc, m2=m2_i, lam=lam, n_therm=n_therm, step=step, seed=seed)
    print(f"L={L:3d}  m2={m2_i:+.4f}  acc={acc:.3f}  <M^2>={np.mean(M**2):.4f}  ({time.time() - t0:.0f} s)", flush=True)


def run_scan(name, Ls, m2, lam, n_sweeps, n_therm, step, seed):
    """Runs every missing chain, then gathers all of them.
    Returns {L: M (len(m2), n_sweeps[L])}, acceptance (len(Ls), len(m2)), {L: snapshots (len(m2), L, L)}."""
    (RUNS / f"scan_{name}").mkdir(parents=True, exist_ok=True)
    # Largest L first: those chains are the slowest, so starting them first balances the cores.
    tasks = [(name, L, i, m2[i], lam, n_sweeps[L], n_therm[L], step, seed) for L in sorted(Ls, reverse=True)
             for i in range(len(m2))]
    with Pool() as pool:
        pool.starmap(run_point, tasks)
    M = {L: np.empty((len(m2), n_sweeps[L]), dtype=np.float32) for L in Ls}
    snaps = {L: np.empty((len(m2), L, L)) for L in Ls}
    acc = np.empty((len(Ls), len(m2)))
    for a, L in enumerate(Ls):
        for i in range(len(m2)):
            p = np.load(point_file(name, L, i))
            # A chain file left over from a different config must not be mixed in silently.
            assert (p["m2"], p["lam"], p["n_therm"], p["step"], p["seed"]) == (m2[i], lam, n_therm[L], step, seed)
            M[L][i], snaps[L][i], acc[a, i] = p["M"], p["snap"], p["acc"]
    return M, acc, snaps


def binder_chi(M, L):
    """U4 = 1 - <M^4>/(3<M^2>^2) and chi = V<M^2>, each as (value, jackknife error), from one series M."""
    M = M.astype(np.float64)
    U4 = jackknife(lambda m2, m4: 1 - m4 / (3 * m2**2), M**2, M**4)
    chi = jackknife(lambda m2: L * L * m2, M**2)
    return U4, chi


def load_scan(name):
    """Reads runs/scan_<name>.npz and returns (data, U4, dU4, chi, dchi), the last four shaped (len(Ls), len(m2))."""
    d = np.load(RUNS / f"scan_{name}.npz")
    shape = (len(d["Ls"]), len(d["m2"]))
    U4, dU4, chi, dchi = (np.empty(shape) for _ in range(4))
    for a, L in enumerate(d["Ls"]):
        for i in range(shape[1]):
            (U4[a, i], dU4[a, i]), (chi[a, i], dchi[a, i]) = binder_chi(d[f"M_{L}"][i], L)
    return d, U4, dU4, chi, dchi


def crossing(x, D):
    """x where the sampled curve D(x) changes sign, by linear interpolation. Requires exactly one sign change."""
    idx = np.flatnonzero(np.diff(np.sign(D)) != 0)
    assert len(idx) == 1, f"expected one sign change, got {len(idx)}"
    i = idx[0]
    return x[i] - D[i] * (x[i + 1] - x[i]) / (D[i + 1] - D[i]), i


def doubling_pairs(Ls):
    """(L, 2L) pairs available in Ls: the pairs whose chi ratio is 2^(2 - 2 Delta_sigma) at criticality."""
    return [(L, 2 * L) for L in Ls if 2 * L in Ls]


def pair_crossing(m2, U4, dU4, Ls, L1, L2):
    """Where U4(L1) and U4(L2) cross: (m2_c, U4 at the crossing). Uses only points where they differ by > 2 sigma,
    so noise near the crossing cannot fake extra sign changes."""
    a, b = list(Ls).index(L1), list(Ls).index(L2)
    D, dD = U4[b] - U4[a], np.hypot(dU4[b], dU4[a])
    sig = np.abs(D) > 2 * dD
    assert (D[sig] < 0).any() and (D[sig] > 0).any(), f"L={L1},{L2}: no significant point on one side of the crossing"
    m2c, _ = crossing(m2[sig], D[sig])
    U4c = 0.5 * (np.interp(-m2c, -m2, U4[a]) + np.interp(-m2c, -m2, U4[b]))  # np.interp needs increasing x
    return m2c, U4c


if __name__ == "__main__":
    name = sys.argv[1]
    cfg = CONFIGS[name]
    print(f"scan '{name}': {cfg}", flush=True)
    t0 = time.time()
    M, acc, snaps = run_scan(name, **cfg)
    np.savez(RUNS / f"scan_{name}.npz", acc=acc, Ls=np.array(cfg["Ls"]), m2=cfg["m2"], lam=cfg["lam"],
             **{f"M_{L}": m for L, m in M.items()}, **{f"snap_{L}": s for L, s in snaps.items()})
    print(f"saved runs/scan_{name}.npz  ({(time.time() - t0) / 60:.1f} min)")
