"""
Step 7 -- production runs: locate T_c in 3D, and show 2D has no transition.

This is the compute wall of the project.  P(q) is not self-averaging, so the
answer needs many coupling realisations, each with its own full parallel
tempering run.  Each (d, L) is written to its own .npz as soon as it finishes,
so the analysis can be re-run without re-running the simulation and an
interrupted job loses at most one geometry.

What is being measured, per temperature:

    xi_L / L     from chi_SG(0)/chi_SG(k_min).  Dimensionless, so curves for
                 different L cross at T_c and nowhere else.
    Binder g     0.5 (3 - <q^4>/<q^2>^2).  Also dimensionless, an independent
                 crossing estimator built from different moments.
    P(q)         the order parameter distribution -- the headline result.

Plus, per sample, everything steps 5 and 6 need to certify the run:
swap acceptance per rung, round trips, the link overlap, and <q^2> measured
over the last half and the last quarter of the run.

3D targets T_c ~ 0.95 for Gaussian bonds.  2D must show *no* crossing at any
temperature reached: it is the negative control, and it is what distinguishes
"we found a transition" from "our sampler got stuck and drew a plausible
histogram".

Run:  python agentic/07_Stat_Mech_and_Ising/step7_production.py
"""

import os
import time

import numpy as np

from ea_model import make_couplings
from ea_pt import make_ladder, run_sample

# (d, L, n_samples, n_equil, n_meas).  L=8 in 3D dominates the cost and is
# about the limit of what plain numpy reaches; its sample count is the thinnest
# part of the whole run, which matters because P(q) is not self-averaging.
JOBS = [
    (3, 4, 32, 6000, 12000),
    (3, 6, 32, 8000, 16000),
    (3, 8, 16, 10000, 20000),
    (2, 8, 20, 6000, 12000),
    (2, 12, 20, 6000, 12000),
    (2, 16, 12, 6000, 12000),
]

# 3D: bracket T_c ~ 0.95 generously.  2D: push lower, since T_c = 0 there and
# the control is more convincing the further down it still shows no crossing.
#
# 18 rungs, not 20: the step 4-6 run measured 0.75-0.88 swap acceptance at L=4,
# far above the 20-40% that indicates a well-matched ladder, so the ladder was
# denser than it needed to be.  Acceptance falls as sqrt(N), so L=8 sets the
# floor -- the production run prints swap_min for exactly this reason.
LADDERS = {3: (0.50, 2.00, 18), 2: (0.30, 2.00, 18)}


def run_job(d, L, n_samples, n_equil, n_meas, seed=1000):
    # Resumable: a finished geometry is never redone.  This run has already
    # been interrupted once (the OS reclaimed memory mid-job), and each
    # geometry is tens of minutes, so skipping completed ones is worth the
    # three lines.  Delete the .npz to force a rerun.
    path = f"step7_d{d}_L{L}.npz"
    if os.path.exists(path):
        print(f"d={d} L={L}: {path} already present, skipping\n")
        return

    t_min, t_max, n_temps = LADDERS[d]
    temps = make_ladder(t_min, t_max, n_temps)
    keys = ("e", "q2", "q4", "chi0", "chik", "q_link", "q2_half", "q2_quarter")
    store = {k: np.empty((n_samples, n_temps)) for k in keys}
    swap = np.empty((n_samples, n_temps - 1))
    trips = np.empty(n_samples)
    hist = None

    print(f"d={d} L={L}: {n_samples} samples, {n_temps} temperatures"
          f" in [{t_min}, {t_max}], {n_equil}+{n_meas} sweeps")
    t0 = time.time()
    for k in range(n_samples):
        rng = np.random.default_rng(seed + 7919 * k + 31 * L + d)
        J = make_couplings(L, d, rng, dist="gaussian")
        out = run_sample(J, L, d, temps, n_equil, n_meas, rng)
        for key in keys:
            store[key][k] = out[key]
        swap[k] = out["swap_acc"]
        trips[k] = out["trips"].mean()
        hist = out["hist"] if hist is None else hist + out["hist"]
        if (k + 1) % 8 == 0 or k == n_samples - 1:
            el = time.time() - t0
            print(f"    sample {k + 1}/{n_samples}   [{el:.0f} s elapsed,"
                  f" {el / (k + 1) * (n_samples - k - 1):.0f} s left]")

    np.savez(path, temps=temps, L=L, d=d, hist=hist, swap=swap, trips=trips,
             **store)
    print(f"    wrote {path}   [{time.time() - t0:.0f} s, "
          f"swap_min {swap.mean(axis=0).min():.3f}, "
          f"trips {trips.mean():.0f}]\n")


if __name__ == "__main__":
    print("=" * 76)
    print("STEP 7: production runs")
    print("=" * 76 + "\n")
    t0 = time.time()
    for job in JOBS:
        run_job(*job)
    print("=" * 76)
    print(f"All jobs complete in {(time.time() - t0) / 60:.1f} min.")
    print("Run step7_plots.py to build the figures.")
    print("=" * 76)
