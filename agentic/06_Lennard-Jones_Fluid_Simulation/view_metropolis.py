"""
The Metropolis run itself: energy per particle as the sweeps progress.

Each curve is one independent simulation at a fixed T*, all started from the
same FCC lattice. The x axis is MC time (1 sweep = N attempted moves), so
reading left to right is watching the system equilibrate.

Run:  python view_metropolis.py   ->  writes metropolis_trace.png
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import lj

N, RHO = 256, 0.8
DELTA = 0.2          # ~36% acceptance at this state point (see lj.py checks)
N_SWEEPS = 2000
TEMPERATURES = [0.5, 1.0, 1.5, 2.5]


if __name__ == "__main__":
    L = lj.box_length(N, RHO)
    lj.check_box_is_legal(L)

    lattice_energy = lj.total_energy(lj.fcc_lattice(N, L), L) / N

    traces, acceptances = {}, {}
    for T in TEMPERATURES:
        # Same seed and same starting lattice for every temperature, so the
        # only thing that differs between curves is T*.
        rng = np.random.default_rng(0)
        positions = lj.fcc_lattice(N, L)

        t0 = time.perf_counter()
        trace, acc = lj.run_sweeps(positions, L, T, DELTA, N_SWEEPS, rng)
        traces[T] = trace / N                 # report per particle
        acceptances[T] = acc

        print(f"T* = {T:4.1f}   E/N: {trace[0]/N:+.3f} -> {trace[-1]/N:+.3f}   "
              f"acceptance {acc.mean():.1%}   ({time.perf_counter()-t0:.1f} s)")

    # -----------------------------------------------------------------
    # Top panel: the energy trace. Bottom: acceptance, to confirm delta
    # is in the 30-50% band and is not drifting over the run.
    # -----------------------------------------------------------------
    fig, (ax_e, ax_a) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2.3, 1]})

    for T in TEMPERATURES:
        ax_e.plot(traces[T], lw=1.2, label=f"$T^*$ = {T}")
        ax_a.plot(acceptances[T], lw=1.0, label=f"$T^*$ = {T}")

    ax_e.axhline(lattice_energy, color="k", ls="--", lw=1,
                 label=f"FCC lattice ({lattice_energy:+.2f})")
    ax_e.set_ylabel("energy per particle  $E/N$")
    ax_e.set_title(f"Metropolis MC, N = {N}, $\\rho^*$ = {RHO}, "
                   f"$\\delta$ = {DELTA}, started from FCC")
    ax_e.legend(fontsize=9, loc="lower right")
    ax_e.grid(alpha=0.3)

    # The 30-50% target band for the step size.
    ax_a.axhspan(0.3, 0.5, color="0.85", zorder=0)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("sweep  (1 sweep = N attempted moves)")
    ax_a.set_ylabel("acceptance")
    ax_a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("metropolis_trace.png", dpi=150)
    print("wrote metropolis_trace.png")
