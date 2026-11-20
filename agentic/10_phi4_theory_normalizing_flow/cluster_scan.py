"""Step 7d: cluster-algorithm (cluster.py) chains at L = 16 ... 128 around m2_c, for the quotient method (quotient.py).

Same layout as scan.py: one independent chain per (L, m2), each written to runs/cluster_<name>/L<L>_m<code>.npz as soon as it
finishes (code = round(-m2 * 1e5)); rerunning skips finished chains. Each L has its own m2 grid, because the critical window
shrinks like 1/L. Every chain records M and sum phi^2 after each update (update = 1 Metropolis sweep + 1 SW step).

Lattice units, lam = 1.  Run: python -u cluster_scan.py <name> <n_workers>   (names in CONFIGS)
"""
import os

# Must precede the numpy import (spawned workers inherit it). OpenBLAS commits buffers for every thread in every process:
# ~740 MB per worker with its default 12 threads, so 12 workers exhausted memory on the first production launch. Nothing
# here uses BLAS; with one thread a worker commits ~36 MB.
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from cluster import cluster_chain, m_and_phi2

RUNS = Path(__file__).parent / "runs"
GRID_6B = np.linspace(-3.65, -4.0, 15)  # the 6b Metropolis grid (scan.py "fine")

CONFIGS = {
    # Pilot at the 6b crossing: tau_int(M^2) per update at each L, to size the production runs.
    "pilot": dict(grids={L: np.array([-3.80]) for L in (16, 32, 64, 128)}, lam=1.0,
                  n_updates={16: 20000, 32: 20000, 64: 10000, 128: 5000},
                  n_therm={16: 1000, 32: 1000, 64: 1000, 128: 1000}, step=1.5, seed=10),
    # Result (runs/cluster_pilot.log): tau_int(M^2) = 21, 19, 16, 17 updates for L = 16, 32, 64, 128 -- flat in L
    # (Metropolis: ~220 sweeps at L = 16, growing ~L^2.5). tau(M) ~ 1; tau(sum phi^2) ~ 22-29 is the magnitude
    # relaxation by the local sweep, which the cluster step cannot help. But U4 at m2 = -3.80 falls with L
    # (0.594, 0.587, 0.557, 0.469): the crossings move to more negative m2 at large L, away from the 6b value -3.806.
    # Pilot 2 locates them roughly before the production grids are fixed.
    "pilot2": dict(grids={L: np.array([-3.82, -3.84, -3.86, -3.88]) for L in (64, 128)}, lam=1.0,
                   n_updates={64: 10000, 128: 5000}, n_therm={64: 1000, 128: 1000}, step=1.5, seed=11),
    # Result (runs/cluster_pilot2.log): U4(64) - U4(128) changes sign between m2 = -3.80 (0.557 vs 0.469) and -3.82
    # (0.602 vs 0.613), so the (64,128) crossing is near -3.82; (32,64) comes out near -3.82 to -3.83. At L = 128 U4 goes
    # 0.47 -> 0.645 over -3.80 ... -3.84, so its grid must be that narrow and fine.
    #
    # Production. L = 16, 32 sit on 6b grid points (indices 3..9) so every point is a direct cluster-vs-Metropolis check
    # (check_quotient.py), and they give an independent (16,32) pair. L = 64 and 128 bracket the pilot crossings with
    # spacings chosen so U4 changes by ~0.02-0.04 between neighbouring points at every L (slope dU4/dm2 grows ~L).
    # Lengths: tau_int(M^2) ~ 20 updates, so 150k-300k updates give > 150 tau per jackknife block (50 blocks) and
    # chi to ~0.6%.
    "prod": dict(grids={16: GRID_6B[3:10], 32: GRID_6B[3:10], 64: np.round(np.linspace(-3.79, -3.85, 7), 5),
                        128: np.round(np.linspace(-3.80, -3.84, 9), 5)}, lam=1.0,
                 n_updates={16: 300000, 32: 300000, 64: 300000, 128: 150000},
                 n_therm={16: 2000, 32: 2000, 64: 2000, 128: 2000}, step=1.5, seed=12),
    # L = 256, for the (128,256) pair. Reweighted pair crossings (quotient.py) moved -3.8112, -3.8215, -3.8240 for
    # (16,32), (32,64), (64,128), so (128,256) should cross near -3.825 to -3.827; the grid covers -3.812 ... -3.840, inside
    # the L = 128 grid. Spacing from the reweighting overlap: std(E) grows ~L (73 at L = 128, so ~150 here), and 0.004 puts
    # neighbouring E means ~0.6 std apart (0.2-0.4 at L <= 128; check_reweight.py [2], [3]). tau_int ~ 25 updates, so 200k
    # updates give > 150 tau per block. Cost: ~15 ms/update alone, ~25 ms with 8 workers -> ~1.5 h on 8 workers.
    # Run: python -u cluster_scan.py prod256 8   then add "prod256" to quotient.CLUSTER_RUNS.
    "prod256": dict(grids={256: np.round(np.linspace(-3.812, -3.840, 8), 5)}, lam=1.0,
                    n_updates={256: 200000}, n_therm={256: 3000}, step=1.5, seed=13),
}


def point_file(name, L, m2):
    return RUNS / f"cluster_{name}" / f"L{L}_m{round(-m2 * 1e5):06d}.npz"


def run_point(name, L, m2, lam, n_updates, n_therm, step, seed):
    out = point_file(name, L, m2)
    if out.exists():
        print(f"L={L:3d}  m2={m2:+.5f}  already done ({out.name})", flush=True)
        return
    t0 = time.time()
    rng = np.random.default_rng([seed, L, round(-m2 * 1e5)])
    rec, snap, acc = cluster_chain(L, m2, lam, n_updates, n_therm, step, rng, m_and_phi2)
    np.savez(out, M=rec[:, 0], phi2=rec[:, 1], snap=snap, acc=acc, m2=m2, lam=lam, n_updates=n_updates,
             n_therm=n_therm, step=step, seed=seed, seconds=time.time() - t0)
    print(f"L={L:3d}  m2={m2:+.5f}  acc={acc:.3f}  <M^2>={np.mean(rec[:, 0] ** 2):.5f}  ({time.time() - t0:.0f} s)", flush=True)


def load(name):
    """{L: (m2 grid, M (n_m2, n_updates), phi2 (n_m2, n_updates))} for a finished config. Checks every file's parameters."""
    cfg = CONFIGS[name]
    out = {}
    for L, grid in cfg["grids"].items():
        files = [np.load(point_file(name, L, m2)) for m2 in grid]
        for p, m2 in zip(files, grid):
            # A chain file left over from a different config must not be mixed in silently.
            assert (float(p["m2"]), float(p["lam"]), int(p["n_updates"]), int(p["n_therm"]), float(p["step"]), int(p["seed"])) \
                == (m2, cfg["lam"], cfg["n_updates"][L], cfg["n_therm"][L], cfg["step"], cfg["seed"])
        out[L] = (grid, np.stack([p["M"] for p in files]), np.stack([p["phi2"] for p in files]))
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    cfg = CONFIGS[name]
    print(f"cluster scan '{name}': {cfg}", flush=True)
    (RUNS / f"cluster_{name}").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Largest L first: those chains are the slowest, so starting them first balances the cores.
    tasks = [(name, L, float(m2), cfg["lam"], cfg["n_updates"][L], cfg["n_therm"][L], cfg["step"], cfg["seed"])
             for L in sorted(cfg["grids"], reverse=True) for m2 in cfg["grids"][L]]
    with Pool(int(sys.argv[2])) as pool:
        pool.starmap(run_point, tasks)
    load(name)
    print(f"all {len(tasks)} chains done ({(time.time() - t0) / 60:.1f} min)")
