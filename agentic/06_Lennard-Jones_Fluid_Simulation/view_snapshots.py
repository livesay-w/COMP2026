"""
What the box actually looks like at the end of each run.

Two figures, one panel per run, each showing the final configuration after
2000 sweeps:

    snapshots_temperature.png   fixed rho* = 0.8, four temperatures
    snapshots_density.png       fixed T*   = 1.5, four densities

In the density figure every panel is drawn on the SAME axis scale, so the
boxes really are different sizes on the page -- that size difference IS the
density.

Run:  python view_snapshots.py
"""

import time

import numpy as np
import matplotlib.pyplot as plt

import lj
from view_initial import draw_configuration

N = 256
DELTA = 0.2
N_SWEEPS = 2000

TEMPERATURES = [0.5, 1.0, 1.5, 2.5]
DENSITIES = [0.2, 0.4, 0.6, 0.8]

FIXED_RHO = 0.8
FIXED_T = 1.5


def equilibrated_configuration(rho, T):
    """Run one simulation from the FCC lattice and return (positions, L, E/N)."""
    L = lj.box_length(N, rho)
    lj.check_box_is_legal(L)

    rng = np.random.default_rng(0)
    positions = lj.fcc_lattice(N, L)
    trace, acc = lj.run_sweeps(positions, L, T, DELTA, N_SWEEPS, rng)

    print(f"  rho* = {rho}, T* = {T}:  E/N -> {trace[-1]/N:+.3f}, "
          f"acceptance {acc.mean():.0%}")
    return positions, L, trace[-1] / N


def snapshot_figure(cases, titles, filename, suptitle, common_scale=False):
    """2x2 grid of final configurations."""
    fig = plt.figure(figsize=(11, 10))
    biggest = max(L for _, L, _ in cases)

    for k, ((positions, L, e), title) in enumerate(zip(cases, titles)):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        draw_configuration(ax, positions, L, title)
        if common_scale:
            # Same limits everywhere, so a dense (small) box looks small.
            ax.set_xlim(0, biggest)
            ax.set_ylim(0, biggest)
            ax.set_zlim(0, biggest)

    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=140)
    print(f"wrote {filename}")


if __name__ == "__main__":
    t0 = time.perf_counter()

    print("temperature sweep:")
    temp_cases = [equilibrated_configuration(FIXED_RHO, T) for T in TEMPERATURES]

    print("density sweep:")
    dens_cases = [equilibrated_configuration(rho, FIXED_T) for rho in DENSITIES]

    snapshot_figure(
        temp_cases,
        [f"$T^*$ = {T},  E/N = {e:+.2f}" for T, (_, _, e) in zip(TEMPERATURES, temp_cases)],
        "snapshots_temperature.png",
        f"Final configurations after {N_SWEEPS} sweeps "
        f"(N = {N}, $\\rho^*$ = {FIXED_RHO}, box identical in all four)",
    )

    snapshot_figure(
        dens_cases,
        [f"$\\rho^*$ = {rho},  L = {L:.2f},  E/N = {e:+.2f}"
         for rho, (_, L, e) in zip(DENSITIES, dens_cases)],
        "snapshots_density.png",
        f"Final configurations after {N_SWEEPS} sweeps "
        f"(N = {N}, $T^*$ = {FIXED_T}, drawn to common scale)",
        common_scale=True,
    )

    print(f"total {time.perf_counter() - t0:.0f} s")
