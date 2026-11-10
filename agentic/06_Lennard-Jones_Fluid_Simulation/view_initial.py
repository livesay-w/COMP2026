"""
A look at the box before any dynamics: the FCC starting configuration.

Run:  python view_initial.py      ->  writes initial_configuration.png
"""

import numpy as np
import matplotlib.pyplot as plt

import lj


def box_edges(L):
    """The 12 edges of the cubic box, as (xs, ys, zs) triples for plotting."""
    corners = np.array([[x, y, z] for x in (0.0, L) for y in (0.0, L) for z in (0.0, L)])
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            # Two corners share an edge iff they differ in exactly one coordinate.
            if np.count_nonzero(corners[i] != corners[j]) == 1:
                edges.append(np.array([corners[i], corners[j]]).T)
    return edges


def draw_configuration(ax, positions, L, title):
    """Scatter the particles inside a wireframe box, drawn to scale."""
    for xs, ys, zs in box_edges(L):
        ax.plot(xs, ys, zs, color="0.6", lw=1.0)

    # Colour by depth purely so the 3D structure is readable on a flat page.
    ax.scatter(*positions.T, s=55, c=positions[:, 2], cmap="viridis",
               edgecolors="k", linewidths=0.4, depthshade=True)

    ax.set_box_aspect((1, 1, 1))          # a cube must look like a cube
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title, fontsize=10)


if __name__ == "__main__":
    N, RHO = 256, 0.8
    L = lj.box_length(N, RHO)
    lj.check_box_is_legal(L)

    lattice = lj.fcc_lattice(N, L)
    e_lattice = lj.total_energy(lattice, L) / N

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    draw_configuration(ax, lattice, L,
                       f"FCC starting configuration,  E/N = {e_lattice:+.3f}")

    fig.suptitle(f"N = {N} particles, $\\rho^*$ = {RHO}, box L = {L:.3f}  "
                 f"(reduced units, $\\sigma$ = $\\epsilon$ = 1)")
    fig.tight_layout()
    fig.savefig("initial_configuration.png", dpi=150)
    print(f"L = {L:.4f}")
    print(f"E/N on FCC lattice: {e_lattice:+.4f}")
    print("wrote initial_configuration.png")
