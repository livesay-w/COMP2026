"""
Density sweep: the same Metropolis run at fixed T* and several rho*.

Density enters only through the box size, L = (N/rho)^(1/3), with N fixed.
Dense box -> particles sit deep in each other's attractive wells -> strongly
negative E/N. Dilute box -> most pairs are beyond the cutoff -> E/N near 0.

Run:  python view_density.py   ->  writes density_trace.png
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import lj

N = 256
TEMPERATURE = 1.5
DELTA = 0.2
N_SWEEPS = 2000
DENSITIES = [0.2, 0.4, 0.6, 0.8]


if __name__ == "__main__":
    traces, acceptances, lattice_energies = {}, {}, {}

    for rho in DENSITIES:
        L = lj.box_length(N, rho)
        lj.check_box_is_legal(L)          # needs r_c < L/2 at every density

        rng = np.random.default_rng(0)
        positions = lj.fcc_lattice(N, L)
        lattice_energies[rho] = lj.total_energy(positions, L) / N

        t0 = time.perf_counter()
        trace, acc = lj.run_sweeps(positions, L, TEMPERATURE, DELTA, N_SWEEPS, rng)
        traces[rho] = trace / N
        acceptances[rho] = acc

        print(f"rho* = {rho:4.1f}  (L = {L:5.2f})   "
              f"E/N: {trace[0]/N:+.3f} -> {trace[-1]/N:+.3f}   "
              f"acceptance {acc.mean():.1%}   ({time.perf_counter()-t0:.1f} s)")

    fig, (ax_e, ax_a) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2.3, 1]})

    for rho in DENSITIES:
        line, = ax_e.plot(traces[rho], lw=1.2, label=f"$\\rho^*$ = {rho}")
        # Dashed marker at each run's own FCC starting energy.
        ax_e.axhline(lattice_energies[rho], color=line.get_color(),
                     ls=":", lw=0.9, alpha=0.7)
        ax_a.plot(acceptances[rho], lw=1.0)

    ax_e.set_ylabel("energy per particle  $E/N$")
    ax_e.set_title(f"Density sweep at fixed $T^*$ = {TEMPERATURE}, "
                   f"N = {N}, $\\delta$ = {DELTA}\n"
                   f"dotted lines = each run's FCC starting energy")
    ax_e.legend(fontsize=9, loc="center right")
    ax_e.grid(alpha=0.3)

    ax_a.axhspan(0.3, 0.5, color="0.85", zorder=0)
    ax_a.set_ylim(0, 1)
    ax_a.set_xlabel("sweep  (1 sweep = N attempted moves)")
    ax_a.set_ylabel("acceptance")
    ax_a.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("density_trace.png", dpi=150)
    print("wrote density_trace.png")
