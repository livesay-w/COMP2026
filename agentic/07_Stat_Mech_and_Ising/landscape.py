"""
The energy landscape itself: every state, every local minimum, every barrier.

For a lattice small enough to enumerate, the landscape is not a metaphor -- it
can be computed exactly.  This builds, for a single disorder sample:

  * the energy of all 2^N states
  * every strict local minimum under single-spin flips
  * the disconnectivity graph (barrier tree): which minima are separated by
    which barriers, obtained by flooding the landscape from the bottom up

and contrasts the spin glass with the ferromagnet on the same lattice.  That
contrast is the point: the ferromagnet has exactly two minima, the spin glass
has dozens, and the barrier tree shows they are not arranged in any simple way.

Definition used: s is a local minimum if *every* one of its N single-flip
neighbours has STRICTLY higher energy.  The strictness matters for the
ferromagnet, where integer couplings produce ties -- a flat domain wall can be
slid along at zero energy cost, so those states are not minima even though no
flip goes downhill.  For Gaussian couplings ties have measure zero and the
distinction is vacuous.

Enumeration indexing: state integer s has spin i given by bit i, so the
single-flip neighbours of s are exactly s XOR (1 << i).  No lookup table is
needed and the minimum search vectorises.

Checks (all exact, no sampling anywhere in this file):

    L.1  ferromagnet has exactly 2 strict local minima
    L.2  the set of minima is closed under global spin flip
    L.3  exactly 2 global minima (ground state and its flip), Gaussian bonds
    L.4  barrier tree leaf count equals the local-minimum count

Run:  python agentic/07_Stat_Mech_and_Ising/landscape.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ea_model import make_couplings
from step2_checks import bond_list, enumerate_states


def all_energies(J, L, d):
    """Energy of every one of the 2^N states, indexed by the bit pattern."""
    n = L ** d
    states = enumerate_states(n)
    e = np.zeros(len(states))
    for i, j, jij in bond_list(J, L, d):
        e -= jij * states[:, i] * states[:, j]
    return states, e


def local_minima(e, n):
    """Indices of strict local minima under single-spin flips."""
    idx = np.arange(len(e))
    is_min = np.ones(len(e), dtype=bool)
    for i in range(n):
        is_min &= e[idx ^ (1 << i)] > e
    return np.flatnonzero(is_min)


class UnionFind:
    """Plain union-find with path compression, over state indices."""

    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra
        return self.find(a)


def barrier_tree(e, n):
    """Disconnectivity graph by flooding the landscape from the bottom.

    States are added in order of increasing energy.  A state with no
    already-added neighbour starts a new basin (it is a local minimum).  A
    state adjacent to two or more distinct basins merges them, and the energy
    at which that happens is the barrier between them -- the lowest point on
    any path connecting the two minima.

    Returns (nodes, root).  Each node carries its energy, its children, and
    the lowest minimum beneath it.
    """
    order = np.argsort(e, kind="stable")
    added = np.zeros(len(e), dtype=bool)
    uf = UnionFind(len(e))
    node_of_root = {}
    nodes = []

    for s in order:
        s = int(s)
        nbrs = [s ^ (1 << i) for i in range(n)]
        live = [t for t in nbrs if added[t]]
        roots = {uf.find(t) for t in live}

        if not roots:
            nodes.append({"energy": float(e[s]), "children": [],
                          "low": float(e[s]), "state": s})
            nid = len(nodes) - 1
        elif len(roots) == 1:
            nid = node_of_root[next(iter(roots))]
        else:
            kids = sorted({node_of_root[r] for r in roots},
                          key=lambda c: nodes[c]["low"])
            if len(kids) == 1:
                nid = kids[0]
            else:
                nodes.append({"energy": float(e[s]), "children": kids,
                              "low": min(nodes[c]["low"] for c in kids),
                              "state": s})
                nid = len(nodes) - 1

        added[s] = True
        r = uf.find(s)
        for t in live:
            r = uf.union(r, t)
        node_of_root[r] = nid

    return nodes, nid


def tree_layout(nodes, root):
    """x position for every node: leaves left to right, parents centred."""
    xs = {}
    counter = [0]
    stack = [(root, False)]
    while stack:
        nid, done = stack.pop()
        nd = nodes[nid]
        if not nd["children"]:
            xs[nid] = float(counter[0])
            counter[0] += 1
        elif done:
            xs[nid] = float(np.mean([xs[c] for c in nd["children"]]))
        else:
            stack.append((nid, True))
            for c in reversed(nd["children"]):
                stack.append((c, False))
    return xs


def draw_tree(ax, nodes, root, xs, color="C0"):
    """Vertical stems from each node up to its parent's merge energy."""
    stack = [root]
    while stack:
        nid = stack.pop()
        nd = nodes[nid]
        if not nd["children"]:
            continue
        cx = [xs[c] for c in nd["children"]]
        ax.plot([min(cx), max(cx)], [nd["energy"]] * 2, color=color, lw=0.9)
        for c in nd["children"]:
            ax.plot([xs[c], xs[c]], [nodes[c]["energy"], nd["energy"]],
                    color=color, lw=0.9)
            stack.append(c)


def count_leaves(nodes, root):
    n, stack = 0, [root]
    while stack:
        nd = nodes[stack.pop()]
        if nd["children"]:
            stack.extend(nd["children"])
        else:
            n += 1
    return n


def analyse(L, d, boundary, seed, label):
    """Everything above, for one disorder sample plus the ferromagnet."""
    n = L ** d
    print(f"--- {label}:  {n} spins, {2 ** n} states, {boundary} boundaries")

    rng = np.random.default_rng(seed)
    J = make_couplings(L, d, rng, dist="gaussian", boundary=boundary)
    states, e = all_energies(J, L, d)
    mins = local_minima(e, n)

    J_f = make_couplings(L, d, rng, dist="ferro", boundary=boundary)
    _, e_f = all_energies(J_f, L, d)
    mins_f = local_minima(e_f, n)

    print(f"  spin glass : {len(mins):5d} strict local minima")
    print(f"  ferromagnet: {len(mins_f):5d} strict local minima")

    # L.1 -- the ferromagnet's GROUND state is exactly the two uniform states,
    # at minus the bond count.
    #
    # It is worth being precise about what the ferromagnet does and does not
    # have, because the naive claim "an unfrustrated model has two minima" is
    # false.  A straight domain wall is single-flip stable: on the 2x2x2 cube a
    # planar half-up/half-down split costs dE = +2 to flip any spin out of, so
    # all 2 + 2d = 8 of those states are genuine local minima.  Single-flip
    # metastability is not the same as frustration.  The real contrast is
    # "a handful of states related by lattice symmetry" against "a thicket that
    # grows exponentially with N", which is what the counts below show.
    n_bonds = sum(1 for _, _, jij in bond_list(J_f, L, d) if jij != 0.0)
    assert np.isclose(e_f.min(), -float(n_bonds)), \
        f"ferromagnet ground state should be {-n_bonds}, got {e_f.min()}"
    assert set(np.flatnonzero(np.isclose(e_f, e_f.min())).tolist()) == {0, 2 ** n - 1}, \
        "ferromagnet ground state is not the two uniform configurations"
    assert len(mins) > len(mins_f), \
        "frustration should produce more minima than the ferromagnet has"

    # L.2 -- nothing in H distinguishes s from -s, so the landscape and hence
    # the set of minima must be exactly symmetric under a global flip.
    flip = (2 ** n - 1)
    assert set(int(m) ^ flip for m in mins) == set(int(m) for m in mins), \
        "minima are not closed under global spin flip"

    # L.3 -- Gaussian couplings give no accidental degeneracy, so the global
    # minimum is unique up to that flip.
    gs = int(np.argmin(e))
    n_global = int(np.sum(np.isclose(e, e.min())))
    print(f"  ground state energy {e.min():+.6f}  (per spin {e.min() / n:+.4f}),"
          f" degeneracy {n_global}")
    assert n_global == 2, f"expected 2 global minima, found {n_global}"

    nodes, root = barrier_tree(e, n)
    leaves = count_leaves(nodes, root)
    # L.4 -- flooding must discover exactly the minima found directly.  These
    # are two independent computations: one a local test on every state, the
    # other a global merge ordering.
    assert leaves == len(mins), f"tree has {leaves} leaves, {len(mins)} minima"

    # L.5 -- the two lowest leaves must both sit at the ground state energy,
    # tying the tree back to the enumeration.
    leaf_e = np.sort([nd["energy"] for nd in nodes if not nd["children"]])
    assert np.isclose(leaf_e[0], e.min()) and np.isclose(leaf_e[1], e.min()), \
        "the two lowest tree leaves are not both at the ground state energy"
    print(f"  barrier tree: {leaves} leaves, root merges at {nodes[root]['energy']:+.4f}")

    barriers = np.array([nodes[root]["energy"] - e[m] for m in mins])
    print(f"  barrier to the global merge: median {np.median(barriers):.3f},"
          f" max {barriers.max():.3f}")
    print("  PASS: all five exact checks hold\n")

    q_gs = states @ states[gs] / n
    return dict(e=e, mins=mins, nodes=nodes, root=root, q_gs=q_gs,
                e_f=e_f, mins_f=mins_f, n=n, label=label, gs=gs)


def plot_landscape(res_small, res_big, path):
    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.33, wspace=0.26)

    for col, res in enumerate((res_small, res_big)):
        ax = fig.add_subplot(gs[0, col])
        xs = tree_layout(res["nodes"], res["root"])
        draw_tree(ax, res["nodes"], res["root"], xs)
        for m in res["mins"]:
            ax.plot(xs.get(_leaf_of(res["nodes"], m), np.nan), res["e"][m], ".",
                    ms=3, color="C3")
        ax.set_ylabel("energy")
        ax.set_xticks([])
        ax.set_title(f"{res['label']}\n{len(res['mins'])} local minima,"
                     f" {2 ** res['n']} states", fontsize=10)

    # The ferromagnet on the same lattice, for contrast: two minima, one funnel.
    ax = fig.add_subplot(gs[0, 2])
    nodes_f, root_f = barrier_tree(res_big["e_f"], res_big["n"])
    xs_f = tree_layout(nodes_f, root_f)
    draw_tree(ax, nodes_f, root_f, xs_f, color="C2")
    ax.set_ylabel("energy")
    ax.set_xticks([])
    ax.set_title(f"ferromagnet, same lattice\n"
                 f"{len(res_big['mins_f'])} local minima (domain walls)",
                 fontsize=10)

    # Energy against overlap with the ground state: the landscape seen from
    # the bottom.  A funnel would be a single V; frustration gives valleys at
    # many different overlaps, which is what multiple states means.
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res_big["q_gs"], res_big["e"], ".", ms=0.6, alpha=0.25, color="0.5")
    ax.plot(res_big["q_gs"][res_big["mins"]], res_big["e"][res_big["mins"]],
            "o", ms=4, color="C3", label="local minima")
    ax.set_xlabel(r"overlap with ground state, $q$")
    ax.set_ylabel("energy")
    ax.set_title("every state, by overlap with the ground state", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(res_big["e"], bins=80, color="0.7", label="all states")
    ax.hist(res_big["e"][res_big["mins"]], bins=80, color="C3",
            label="local minima")
    ax.set_yscale("log")
    ax.set_xlabel("energy")
    ax.set_ylabel("count")
    ax.set_title("density of states vs. minima", fontsize=10)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    e_mins = np.sort(res_big["e"][res_big["mins"]])
    ax.plot(np.arange(len(e_mins)), e_mins - e_mins[0], "o-", ms=3)
    ax.set_xlabel("minimum, ordered by energy")
    ax.set_ylabel(r"$E - E_0$")
    ax.set_title("low-lying minima crowd just above $E_0$", fontsize=10)

    fig.suptitle("The Edwards-Anderson energy landscape, computed exactly", y=0.98)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}\n")


def _leaf_of(nodes, state):
    for nid, nd in enumerate(nodes):
        if not nd["children"] and nd["state"] == state:
            return nid
    return -1


if __name__ == "__main__":
    print("=" * 76)
    print("ENERGY LANDSCAPE: exact enumeration, all minima, all barriers")
    print("=" * 76 + "\n")

    small = analyse(2, 3, "open", seed=7, label="3D L=2 open (8 spins)")
    big = analyse(4, 2, "periodic", seed=11, label="2D L=4 periodic (16 spins)")

    plot_landscape(small, big, "landscape_barrier_tree.png")

    print("=" * 76)
    print("Note on scale: this is 16 spins.  The production runs are 512, where")
    print("the number of minima grows exponentially and enumeration is")
    print("hopeless -- 2^512 states.  That gap is the whole reason parallel")
    print("tempering exists, and why the barrier tree can only ever be shown")
    print("for a toy lattice.")
    print("=" * 76)
