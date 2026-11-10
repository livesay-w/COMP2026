"""HP protein folding on the 2D square lattice.

A fold of an N-bead chain is a string of N-1 moves (U/D/L/R). Bead 0 sits at
the origin and each move steps one lattice site. A fold is legal only if the
chain never revisits a site (self-avoiding walk). Lattice spacing = 1.

Step 1: fold -> coordinates -> validity.
Step 2: energy = -(number of non-bonded H-H nearest-neighbor contacts).
Step 3: brute force over every fold (N <= 15) for the exact ground state.
Step 4: move set (end, corner, crankshaft, pivot) checked against brute force (N <= 12).
Step 5: Metropolis at fixed T checked against the exact Boltzmann distribution (N <= 15).

Usage:
    python hp_fold.py                  # default 10-bead sequence
    python hp_fold.py HPHPPHHPHPPHPHHPPHPH
"""
import argparse
import itertools
from pathlib import Path

import numpy as np

DIRS = {'U': (0, 1), 'D': (0, -1), 'L': (-1, 0), 'R': (1, 0)}
BRUTE_FORCE_MAX = 15   # enumerating every fold is ~2.6^N; skip above this length
GRAPH_MAX = 12         # step 4 builds every fold's neighbor list; too slow above this
# Number of self-avoiding walks of n steps on the square lattice (OEIS A001411).
SAW_COUNTS = {1: 4, 2: 12, 3: 36, 4: 100, 5: 284, 6: 780, 7: 2172, 8: 5916, 9: 16268,
              10: 44100, 11: 120292, 12: 324932, 13: 881500, 14: 2374444}


def moves_to_coords(moves):
    """N-1 moves -> (N, 2) integer array of bead positions, bead 0 at the origin."""
    steps = [(0, 0)] + [DIRS[m] for m in moves]
    return np.cumsum(steps, axis=0)


def is_valid(coords):
    """True if no two beads share a lattice site."""
    return len({tuple(c) for c in coords}) == len(coords)


def contacts(seq, coords):
    """List of (i, j), i < j, of H-H pairs on neighboring sites but not bonded (j > i+1)."""
    site = {tuple(c): i for i, c in enumerate(coords)}
    out = []
    for i, (x, y) in enumerate(coords):
        if seq[i] != "H":
            continue
        for dx, dy in DIRS.values():
            j = site.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == "H":
                out.append((i, j))
    return out


def energy(seq, coords):
    """HP energy: -1 per H-H contact (dimensionless, contact energy = 1)."""
    return -len(contacts(seq, coords))


def energy_naive(seq, coords):
    """O(N^2) double loop over all pairs, used only to cross-check energy()."""
    e = 0
    for i in range(len(seq)):
        for j in range(i + 2, len(seq)):
            if seq[i] == seq[j] == "H" and np.abs(coords[i] - coords[j]).sum() == 1:
                e -= 1
    return e


def random_folds(N, n, rng, max_tries=2_000_000):
    """n legal folds drawn uniformly by rejection: random move strings, keep the valid ones."""
    folds, tries = [], 0
    while len(folds) < n:
        tries += 1
        assert tries <= max_tries, f"only {len(folds)} legal folds in {max_tries} tries; N = {N} too long"
        moves = "".join(rng.choice(list("UDLR"), N - 1))
        if is_valid(moves_to_coords(moves)):
            folds.append(moves)
    return folds


# The 8 symmetries of the square lattice (4 rotations x optional mirror).
SYMMETRIES = [np.array(m) for m in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]],
                                    [[0, 1], [-1, 0]], [[-1, 0], [0, 1]], [[1, 0], [0, -1]],
                                    [[0, 1], [1, 0]], [[0, -1], [-1, 0]])]


def check_step2(seq, rng):
    # Hand cases: the four distinct HPPH shapes, only the unit square has a contact.
    for moves, e in [("RRR", 0), ("RRU", 0), ("RUR", 0), ("RUL", -1)]:
        assert energy("HPPH", moves_to_coords(moves)) == e, moves
    # Alternating H's are all even-indexed: parity forbids every contact.
    assert energy("HPHPHPHPH", moves_to_coords("UURRDDLU")) == 0   # 3x3 spiral

    for moves in random_folds(len(seq), 200, rng):
        c = moves_to_coords(moves)
        e = energy(seq, c)
        assert e == energy_naive(seq, c), moves
        # Rotating / mirroring the fold, or reading the chain backwards, can't change E.
        for S in SYMMETRIES:
            assert energy(seq, c @ S.T) == e, moves
        assert energy(seq[::-1], c[::-1]) == e, moves
        # Lattice parity: neighboring sites have opposite x+y parity, so i+j is odd.
        assert all((i + j) % 2 == 1 for i, j in contacts(seq, c)), moves
    print("step 2 checks passed (hand cases + 200 random folds)")


H_COLOR, P_COLOR, INK, MUTED = "#eb6834", "#2a78d6", "#0b0b0b", "#8a8984"


def draw_fold(ax, seq, coords, title=""):
    """Chain on the lattice: H filled orange, P open blue, H-H contacts dashed, bead 0 ringed."""
    for i, j in contacts(seq, coords):
        ax.plot(*coords[[i, j]].T, ls=(0, (2, 1.5)), lw=1.5, color=INK, zorder=1)
    ax.plot(*coords.T, lw=2, color=MUTED, zorder=2)
    h = np.array([s == "H" for s in seq])
    ax.scatter(*coords[0], s=260, facecolor="none", edgecolor=INK, lw=1.2, zorder=3, label="bead 0")
    ax.scatter(*coords[h].T, s=80, color=H_COLOR, zorder=4, label="H")
    ax.scatter(*coords[~h].T, s=80, facecolor="white", edgecolor=P_COLOR, lw=2, zorder=4, label="P")
    ax.set_aspect("equal")
    ax.margins(0.25)
    ax.axis("off")
    ax.set_title(title, fontsize=9, color=INK)


def plot_step2(seq, rng, path):
    """Estimated energy distribution over random legal folds + a gallery of folds by energy."""
    import matplotlib.pyplot as plt

    folds = random_folds(len(seq), 2000, rng)
    E = np.array([energy(seq, moves_to_coords(m)) for m in folds])

    # Gallery: lowest-energy folds first, at most 2 per energy level, 8 total.
    order = np.argsort(E, kind="stable")
    gallery, per_level = [], {}
    for k in order:
        if per_level.get(E[k], 0) < 2 and len(gallery) < 8:
            gallery.append(k)
            per_level[E[k]] = per_level.get(E[k], 0) + 1

    fig = plt.figure(figsize=(11, 7), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    levels = np.arange(E.min(), 1)
    frac = np.array([(E == e).mean() for e in levels])
    ax.bar(levels, frac, width=0.6, color=P_COLOR)
    for e, f in zip(levels, frac):
        ax.annotate(f"{f:.1%}", (e, f), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(levels)
    ax.set_xlabel("energy E (number of H–H contacts, negated)")
    ax.set_ylabel("fraction of legal folds")
    ax.set_title(f"{seq}: energy of {len(folds)} random legal folds "
                 f"(lowest seen E = {E.min()})", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for n, k in enumerate(gallery):
        a = fig.add_subplot(gs[1 + n // 4, n % 4])
        draw_fold(a, seq, moves_to_coords(folds[k]), f"E = {E[k]}   {folds[k]}")
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles + [plt.Line2D([], [], ls=(0, (2, 1.5)), color=INK)], labels + ["H–H contact"],
               loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print(f"saved {path}")
    return E


def enumerate_folds(seq):
    """Every legal fold with first move R, as (moves, E). Depth-first growth of the chain.

    Fixing the first move loses nothing: the other three first moves are
    rotations, with identical energies, so every energy fraction is unchanged.
    E is accumulated as each bead is placed, independently of energy().
    """
    N = len(seq)
    site = {(0, 0): 0, (1, 0): 1}
    path, moves, out = [(0, 0), (1, 0)], ["R"], []

    def grow(e):
        k = len(path)
        if k == N:
            out.append(("".join(moves), e))
            return
        x, y = path[-1]
        for m, (dx, dy) in DIRS.items():
            p = (x + dx, y + dy)
            if p in site:
                continue
            de = 0
            if seq[k] == "H":   # new contacts of bead k with earlier, non-bonded H's
                for ddx, ddy in DIRS.values():
                    j = site.get((p[0] + ddx, p[1] + ddy))
                    if j is not None and j < k - 1 and seq[j] == "H":
                        de -= 1
            site[p] = k
            path.append(p)
            moves.append(m)
            grow(e + de)
            del site[p]
            path.pop()
            moves.pop()

    grow(0)
    return out


def check_step3(seq, folds, E_sample, rng):
    N = len(seq)
    moves = [m for m, _ in folds]
    E = np.array([e for _, e in folds])
    # Count must match the known number of self-avoiding walks (1/4: first move fixed).
    assert len(folds) == SAW_COUNTS[N - 1] // 4, (len(folds), SAW_COUNTS[N - 1] // 4)
    assert len(set(moves)) == len(moves), "duplicate folds"
    # The incremental energy must agree with energy() from step 2.
    for k in rng.choice(len(folds), min(500, len(folds)), replace=False):
        c = moves_to_coords(moves[k])
        assert is_valid(c) and energy(seq, c) == E[k], moves[k]
    # Fully independent route for short chains: all 4^(N-1) strings, step 1 + 2 code only.
    if N <= 10:
        e_all = [energy(seq, c) for c in map(moves_to_coords, itertools.product("UDLR", repeat=N - 1))
                 if is_valid(c)]
        for e in range(E.min(), 1):
            assert e_all.count(e) == 4 * (E == e).sum(), e
        print(f"  all 4^{N - 1} = {4 ** (N - 1)} move strings agree with the enumeration")
    # Random sampling can't beat the true minimum, and its fractions must agree
    # with the exact ones within sampling error (4 sigma, binomial).
    assert E.min() <= E_sample.min()
    n = len(E_sample)
    print(f"  {'E':>3} {'exact':>8} {'sampled':>8} {'4 sigma':>8}")
    for e in range(E.min(), 1):
        p, p_hat = (E == e).mean(), (E_sample == e).mean()
        tol = 4 * np.sqrt(p * (1 - p) / n)
        print(f"  {e:3d} {p:8.3%} {p_hat:8.3%} {tol:8.3%}")
        assert abs(p_hat - p) <= tol, (e, p, p_hat)
    print(f"step 3 checks passed: {len(folds)} folds, E_min = {E.min()}, "
          f"{(E == E.min()).sum()} ground-state folds")


def plot_step3(seq, folds, E_sample, path):
    """Exact energy distribution (brute force), same layout as step 2, + ground states and traps."""
    import matplotlib.pyplot as plt

    E = np.array([e for _, e in folds])
    E_min = E.min()

    # Gallery: distinct ground states (a fold and its mirror U<->D count once), then
    # folds one contact short of the minimum.
    mirror = str.maketrans("UD", "DU")
    seen, gallery = set(), []
    for target in (E_min, E_min + 1):
        for m, e in folds:
            key = min(m, m.translate(mirror))
            if e == target and key not in seen and len(gallery) < 8:
                seen.add(key)
                gallery.append((m, e))

    fig = plt.figure(figsize=(11, 7), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1])
    ax = fig.add_subplot(gs[0, :])
    levels = np.arange(E_min, 1)
    frac = np.array([(E == e).mean() for e in levels])
    frac_s = np.array([(E_sample == e).mean() for e in levels])
    ax.bar(levels, frac, width=0.6, color=P_COLOR, label="exact (brute force)")
    ax.scatter(levels, frac_s, marker="D", s=40, color=H_COLOR, zorder=3,
               label=f"step 2 sample ({len(E_sample)} folds)")
    for e, f, cnt in zip(levels, frac, [(E == e).sum() for e in levels]):
        ax.annotate(f"{f:.3%} ({cnt})", (e, f), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(levels)
    ax.set_xlabel("energy E (number of H–H contacts, negated)")
    ax.set_ylabel("fraction of legal folds")
    ax.set_title(f"{seq}: exact energy of all {len(folds)} folds with first move R "
                 f"(E_min = {E_min})", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for n, (m, e) in enumerate(gallery):
        a = fig.add_subplot(gs[1 + n // 4, n % 4])
        kind = "ground state" if e == E_min else "one short"
        draw_fold(a, seq, moves_to_coords(m), f"{kind}: E = {e}   {m}")
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles + [plt.Line2D([], [], ls=(0, (2, 1.5)), color=INK)], labels + ["H–H contact"],
               loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print(f"saved {path}")


# ---- Step 4: move set -------------------------------------------------------
# Every move is an edit of the move string. Pivot applies one of the 7
# non-identity lattice symmetries to every move after position i, which rotates
# or mirrors the rest of the chain about bead i.
OPP = {"U": "D", "D": "U", "L": "R", "R": "L"}
SYM_MAPS = [str.maketrans("RULD", t) for t in
            ("ULDR", "LDRU", "DRUL",            # rotate 90, 180, 270 (counterclockwise)
             "LURD", "RDLU", "URDL", "DLUR")]   # mirror in y axis, x axis, y=x, y=-x
TO_R = {"R": "RULD", "U": "DRUL", "L": "LDRU", "D": "ULDR"}   # rotation taking first move to R


def canonical(moves):
    """Rotate the whole fold so its first move is R, matching the brute-force list."""
    return moves.translate(str.maketrans("RULD", TO_R[moves[0]]))


def perpendicular(a, b):
    return (a in "UD") != (b in "UD")


def move_table(N, pivot=True):
    """{kind: (positions, number of choices)}. Kind, position and choice are each drawn
    uniformly and independently of the current fold, so P(A -> B) = P(B -> A)."""
    table = {"end": ([0, N - 2], 3),          # new direction for the first or last bond
             "corner": (list(range(N - 2)), 1),
             "crank": (list(range(N - 3)), 1)}
    if pivot:
        table["pivot"] = (list(range(1, N - 1)), len(SYM_MAPS))
    return table


def apply_move(moves, kind, i, c):
    """The fold after one move, or None if the move doesn't apply here or the chain overlaps."""
    if kind == "end":                        # end bead swings to another neighboring site
        new = moves[:i] + [d for d in "UDLR" if d != moves[i]][c] + moves[i + 1:]
    elif kind == "corner":                   # "RU" <-> "UR": corner bead to the opposite corner
        a, b = moves[i:i + 2]
        if not perpendicular(a, b):
            return None
        new = moves[:i] + b + a + moves[i + 2:]
    elif kind == "crank":                    # "URD" <-> "DRU": U-shape flips across its base
        a, b, z = moves[i:i + 3]
        if z != OPP[a] or not perpendicular(a, b):
            return None
        new = moves[:i] + OPP[a] + b + a + moves[i + 3:]
    else:                                    # pivot the tail about bead i
        new = moves[:i] + moves[i:].translate(SYM_MAPS[c])
    return new if is_valid(moves_to_coords(new)) else None


def propose(moves, table, rng):
    """One random move; None means rejected (doesn't apply, or overlaps)."""
    kind = list(table)[rng.integers(len(table))]
    positions, n_choices = table[kind]
    return apply_move(moves, kind, positions[rng.integers(len(positions))], rng.integers(n_choices))


def all_proposals(moves, table):
    """Every (probability, new fold or None) that propose() can produce, exactly."""
    for kind, (positions, n_choices) in table.items():
        p = 1 / (len(table) * len(positions) * n_choices)
        for i in positions:
            for c in range(n_choices):
                yield p, apply_move(moves, kind, i, c)


def move_graph(fold_list, table):
    """{fold: {neighbor: total proposal probability}} over canonical folds."""
    graph = {}
    for a in fold_list:
        nbrs = graph.setdefault(a, {})
        for p, new in all_proposals(a, table):
            if new is not None and canonical(new) != a:
                b = canonical(new)
                nbrs[b] = nbrs.get(b, 0) + p
    return graph


def reachable(graph, start):
    seen, stack = {start}, [start]
    while stack:
        for b in graph[stack.pop()]:
            if b not in seen:
                seen.add(b)
                stack.append(b)
    return seen


def traps(graph, E_of):
    """Plateaus (folds joined by equal-energy moves) with no move to lower energy, excluding
    the global minimum. A T = 0 search that falls into one can never leave. {E: count}."""
    E_min, done, out = min(E_of.values()), set(), {}
    for a in graph:
        if a in done:
            continue
        plateau, stack = {a}, [a]
        while stack:
            for b in graph[stack.pop()]:
                if E_of[b] == E_of[a] and b not in plateau:
                    plateau.add(b)
                    stack.append(b)
        done |= plateau
        exit_down = any(E_of[b] < E_of[a] for f in plateau for b in graph[f])
        if not exit_down and E_of[a] > E_min:
            out[E_of[a]] = out.get(E_of[a], 0) + 1
    return out


def random_walk(seq, table, n_steps, rng, every=20):
    """Infinite-temperature walk: accept every legal move. Returns distinct folds visited
    after each step, and energies sampled every `every` steps."""
    moves = "R" * (len(seq) - 1)
    visited, coverage, E = set(), np.empty(n_steps, int), []
    for t in range(n_steps):
        new = propose(moves, table, rng)
        if new is not None:
            moves = new
        visited.add(canonical(moves))
        coverage[t] = len(visited)
        if t % every == 0:
            E.append(energy(seq, moves_to_coords(moves)))
    return coverage, np.array(E)


def check_step4(seq, folds, rng, n_steps=200_000):
    E_of = dict(folds)
    fold_list = list(E_of)
    E_exact = np.array(list(E_of.values()))
    start = canonical("R" * (len(seq) - 1))
    out = {"n_folds": len(fold_list)}
    for name, table in [("local + pivot", move_table(len(seq))),
                        ("local only", move_table(len(seq), pivot=False))]:
        graph = move_graph(fold_list, table)
        # Every proposal a -> b must be matched by b -> a with the same probability.
        for a, nbrs in graph.items():
            for b, p in nbrs.items():
                assert np.isclose(graph[b].get(a, 0), p), (a, b, p, graph[b].get(a, 0))
        n_reach = len(reachable(graph, start))
        trap_counts = traps(graph, E_of)
        coverage, E_walk = random_walk(seq, table, n_steps, rng)
        print(f"  {name:13s}: reaches {n_reach}/{len(fold_list)} folds from the straight chain, "
              f"walk visited {coverage[-1]} in {n_steps} steps, "
              f"trapping plateaus by E: {dict(sorted(trap_counts.items()))}")
        out[name] = (coverage, E_walk, trap_counts, n_reach)

    # The full move set must be ergodic, and its unbiased walk must visit folds uniformly,
    # so its energy fractions match brute force (4 sigma on the thinned samples).
    coverage, E_walk, _, n_reach = out["local + pivot"]
    assert n_reach == len(fold_list), "full move set can't reach every fold"
    for e in range(E_exact.min(), 1):
        p, p_hat = (E_exact == e).mean(), (E_walk == e).mean()
        assert abs(p_hat - p) <= 4 * np.sqrt(p * (1 - p) / len(E_walk)), (e, p, p_hat)
    print("step 4 checks passed: proposals symmetric, full move set ergodic, walk uniform")
    return out


def plot_step4(seq, folds, out, path):
    """Walk coverage, walk vs exact energy fractions, and T = 0 traps for both move sets."""
    import matplotlib.pyplot as plt

    E_exact = np.array([e for _, e in folds])
    levels = np.arange(E_exact.min(), 1)
    colors = {"local + pivot": P_COLOR, "local only": H_COLOR}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4), facecolor="#fcfcfb")

    ax = axs[0]
    for name, c in colors.items():
        coverage = out[name][0]
        ax.plot(np.arange(1, len(coverage) + 1), coverage, lw=2, color=c, label=name)
    ax.axhline(out["n_folds"], ls="--", lw=1, color=MUTED)
    ax.annotate(f"all {out['n_folds']} folds", (1, out["n_folds"]), xytext=(0, 4),
                textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("walk step (every legal move accepted)")
    ax.set_ylabel("distinct folds visited")
    ax.set_title("coverage of the fold space", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="center left")

    ax = axs[1]
    E_walk = out["local + pivot"][1]
    ax.bar(levels, [(E_exact == e).mean() for e in levels], width=0.6, color=MUTED,
           label="exact (brute force)")
    ax.scatter(levels, [(E_walk == e).mean() for e in levels], marker="D", s=40,
               color=P_COLOR, zorder=3, label=f"walk, local + pivot ({len(E_walk)} samples)")
    ax.set_xticks(levels)
    ax.set_xlabel("energy E")
    ax.set_ylabel("fraction")
    ax.set_title("unbiased walk visits folds uniformly", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axs[2]
    w = 0.38
    for k, (name, c) in enumerate(colors.items()):
        t = out[name][2]
        counts = [t.get(e, 0) for e in levels[1:]]
        bars = ax.bar(levels[1:] + (k - 0.5) * w, counts, width=w - 0.04, color=c, label=name)
        ax.bar_label(bars, fontsize=8, color=INK, padding=2)
    ax.set_xticks(levels[1:])
    ax.set_xlabel("energy E of the trap")
    ax.set_ylabel("trapping plateaus (no downhill move)")
    ax.set_title(f"T = 0 traps (true E_min = {E_exact.min()})", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    for ax in axs:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{seq}: step 4 move set", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print(f"saved {path}")


# ---- Step 5: Metropolis -----------------------------------------------------
def metropolis(seq, temps, table, rng, moves=None, stop_E=None):
    """One Metropolis step per entry of temps, so T may change every step: constant in
    step 5, cooling in step 6, zero for a quench. Temperature in units of the contact
    energy (k_B = 1). Starts from the straight chain unless a fold is given; stops early
    once E <= stop_E if that is given.
    Returns the energy after every step, which steps were accepted, and the best fold seen."""
    if moves is None:
        moves = "R" * (len(seq) - 1)
    E = energy(seq, moves_to_coords(moves))
    best_E, best_moves = E, moves
    E_trace, accepted = np.empty(len(temps), int), np.zeros(len(temps), bool)
    for t, T in enumerate(temps):
        new = propose(moves, table, rng)
        if new is not None:
            E_new = energy(seq, moves_to_coords(new))
            dE = E_new - E
            # Downhill or level: always. Uphill: with probability exp(-dE/T), never at T = 0.
            if dE <= 0 or (T > 0 and rng.random() < np.exp(-dE / T)):
                moves, E = new, E_new
                accepted[t] = True
                if E < best_E:
                    best_E, best_moves = E, moves
        E_trace[t] = E
        if stop_E is not None and E <= stop_E:
            return E_trace[:t + 1], accepted[:t + 1], best_E, best_moves
    return E_trace, accepted, best_E, best_moves


def time_to_ground(seq, E_min, rng, temps=(0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0),
                   n_runs=48, max_steps=40_000):
    """Steps from the straight chain until E_min is first reached, n_runs per fixed T.
    A run that never gets there is recorded as max_steps (so medians are lower bounds)."""
    table = move_table(len(seq))
    out = {}
    for T in temps:
        hits = [len(metropolis(seq, np.full(max_steps, T), table, rng, stop_E=E_min)[0])
                for _ in range(n_runs)]
        out[T] = np.array(hits)
        print(f"  T = {T}: median {np.median(hits):.0f} steps to reach E = {E_min}, "
              f"{(out[T] < max_steps).sum()}/{n_runs} runs got there within {max_steps}")
    return out, max_steps


def boltzmann(E_exact, T):
    """Exact P(E) at temperature T from the brute-force fold counts g(E): P ~ g(E) exp(-E/T)."""
    levels = np.arange(E_exact.min(), 1)
    g = np.array([(E_exact == e).sum() for e in levels])
    w = g * np.exp(-(levels - levels[0]) / T)   # shift by E_min so nothing overflows
    return levels, w / w.sum()


def check_step5(seq, folds, rng, temps=(0.35, 0.5, 1.0, 2.0),
                n_steps=200_000, burn=20_000, n_blocks=20):
    """At fixed T the time spent at each energy must match the Boltzmann prediction.
    Successive steps are correlated, so the error bar comes from splitting the run into
    blocks and taking the spread of the block averages (block averaging)."""
    E_exact = np.array([e for _, e in folds])
    table = move_table(len(seq))
    out = {}
    for T in temps:
        E_trace, accepted, _, _ = metropolis(seq, np.full(n_steps, T), table, rng)
        samples = E_trace[burn:]
        blocks = np.array_split(samples, n_blocks)
        levels, p = boltzmann(E_exact, T)
        print(f"  T = {T}: acceptance {accepted.mean():.1%}")
        print(f"    {'E':>3} {'Boltzmann':>10} {'simulated':>10} {'± 1 sigma':>10}")
        frac, err = [], []
        for e, pe in zip(levels, p):
            f_blocks = np.array([(b == e).mean() for b in blocks])
            f, s = f_blocks.mean(), f_blocks.std(ddof=1) / np.sqrt(n_blocks)
            # Floor the error at the uncorrelated binomial value, so a rare level that
            # never shows up in any block still gets a sensible tolerance.
            s = max(s, np.sqrt(pe * (1 - pe) / len(samples)))
            print(f"    {e:3d} {pe:10.3%} {f:10.3%} {s:10.3%}")
            assert abs(f - pe) <= 4 * s, (T, e, pe, f, s)
            frac.append(f)
            err.append(s)
        E_blocks = np.array([b.mean() for b in blocks])
        mean_E, mean_err = E_blocks.mean(), E_blocks.std(ddof=1) / np.sqrt(n_blocks)
        assert abs(mean_E - (levels * p).sum()) <= 4 * mean_err, (T, mean_E, (levels * p).sum())
        # Steps per independent sample: n_steps / n_eff, with n_eff = var(E) / (error of <E>)^2.
        corr_steps = len(samples) * mean_err**2 / samples.var()
        out[T] = (np.array(frac), np.array(err), mean_E, mean_err, accepted.mean(), corr_steps)
    print("step 5 checks passed: fixed-T Metropolis matches Boltzmann at every T")
    return out


# OPEN CONCERNS about the step 5 figure (raised by the student, 2026-10-30), revisit
# before building on it:
# 1. The bottom row used to show raw E(t) traces at each T; those were too hard to read.
#    They are now "fraction of runs that have reached E_min by step t". The student likes
#    the bottom-right "steps vs T" panel best; keep the per-T panels in that spirit.
# 2. "Steps per independent sample" is easy to misread as "steps to change state". It is
#    NOT 1/acceptance (~17 steps at T = 0.35); it is how long until the chain leaves its
#    basin (~1744). The panel titles should make that distinction clear.
# 3. The student worried that dividing by steps-per-independent-sample lets low T catch up
#    to high T and hides the difference they want to show. Nothing is normalized: the blue
#    curves are raw step counts; orange is a separate measurement on the same axis.
#    Consider dropping or separating the orange line if it keeps causing confusion.
# 4. Goal of the figure: make the low-T downside (stuck in traps) and high-T downside
#    (rarely settles at E_min) obvious. At N = 10 the gaps are modest (curves overlap,
#    best fixed T ~ 0.5); rerun at N = 20 to check whether they widen as expected.
def plot_step5(seq, folds, out, hits, path):
    """Top: simulated vs exact P(E) at each T, and <E> vs T.
    Bottom: fraction of runs that have reached E_min by step t at each T, and steps vs T."""
    import matplotlib.pyplot as plt

    E_exact = np.array([e for _, e in folds])
    E_min = E_exact.min()
    temps = sorted(out)
    fig, grid = plt.subplots(2, len(temps) + 1, figsize=(3.2 * (len(temps) + 1), 7.6),
                             facecolor="#fcfcfb")
    axs = grid[0]
    for ax, T in zip(axs, temps):
        levels, p = boltzmann(E_exact, T)
        frac, err, _, _, acc, _ = out[T]
        ax.bar(levels, p, width=0.6, color=MUTED, label="Boltzmann (exact)")
        ax.errorbar(levels, frac, yerr=err, fmt="D", ms=6, color=P_COLOR, capsize=3,
                    label="Metropolis")
        ax.set_xticks(levels)
        ax.set_ylim(0, 1)
        ax.set_xlabel("energy E")
        ax.set_title(f"T = {T}   (accept {acc:.0%})", color=INK)
    axs[0].set_ylabel("fraction of time at E")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")

    ax = axs[-1]
    T_grid = np.linspace(0.15, 3, 200)
    ax.plot(T_grid, [(lambda l, p: (l * p).sum())(*boltzmann(E_exact, T)) for T in T_grid],
            lw=2, color=MUTED, label="exact ⟨E⟩")
    ax.errorbar(temps, [out[T][2] for T in temps], yerr=[4 * out[T][3] for T in temps],
                fmt="D", ms=6, color=P_COLOR, capsize=3, label="Metropolis (±4σ)")
    ax.axhline(E_exact.min(), ls="--", lw=1, color=INK)
    ax.annotate(f"E_min = {E_exact.min()}", (3, E_exact.min()), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=8, color=INK)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("average energy ⟨E⟩")
    ax.set_title("⟨E⟩ falls as T drops", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="center right")

    # Bottom row: for each T, the fraction of runs (from the straight chain) that have
    # reached E_min by step t. Further left = faster. Other temperatures in gray.
    hit_hits, max_steps = hits
    assert set(temps) <= set(hit_hits), "time_to_ground must include the step 5 temperatures"

    def reached(h):
        """Step curve of the fraction of runs done by step t; runs that hit the limit never count."""
        t = np.sort(h[h < max_steps])
        frac = np.arange(1, len(t) + 1) / len(h)
        return (np.concatenate([[1], t, [max_steps]]),
                np.concatenate([[0], frac, [len(t) / len(h)]]))

    for ax, T in zip(grid[1], temps):
        for T_other in hit_hits:
            if T_other != T:
                ax.plot(*reached(hit_hits[T_other]), lw=1, color=MUTED, alpha=0.35,
                        drawstyle="steps-post")
        ax.plot(*reached(hit_hits[T]), lw=2, color=P_COLOR, drawstyle="steps-post")
        med = np.median(hit_hits[T])
        ax.axvline(med, ls="--", lw=1, color=INK)
        ax.annotate(f"median\n{med:.0f}", (med, 0.05), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=INK)
        ax.set_xscale("log")
        ax.set_xlim(10, max_steps)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("step")
        ax.set_title(f"T = {T}\n{out[T][5]:.0f} steps per independent sample",
                     fontsize=10, color=INK)
    grid[1, 0].set_ylabel(f"fraction of runs that have\nreached E = {E_min}")
    grid[1, 0].plot([], [], lw=2, color=P_COLOR, label="this T")
    grid[1, 0].plot([], [], lw=1, color=MUTED, alpha=0.6, label="other T")
    grid[1, 0].legend(frameon=False, fontsize=8, loc="upper left")

    ax = grid[1, -1]
    hT = sorted(hit_hits)
    med = np.array([np.median(hit_hits[T]) for T in hT])
    lo = med - np.array([np.percentile(hit_hits[T], 25) for T in hT])
    hi = np.array([np.percentile(hit_hits[T], 75) for T in hT]) - med
    ax.errorbar(hT, med, yerr=[lo, hi], fmt="o-", ms=6, lw=2, color=P_COLOR, capsize=3,
                label=f"steps to first reach E = {E_min}\n(median, 25–75%, {len(hit_hits[hT[0]])} runs)")
    ax.plot(temps, [out[T][5] for T in temps], "D-", ms=6, lw=2, color=H_COLOR,
            label="steps per independent sample")
    ax.axhline(max_steps, ls="--", lw=1, color=MUTED)
    ax.annotate("run limit", (hT[0], max_steps), xytext=(0, -11), textcoords="offset points",
                fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(20, 2e6)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("steps")
    ax.set_title("how long each T takes", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    for ax in grid.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(f"{seq}: step 5, Metropolis at fixed temperature", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print(f"saved {path}")


def check_step1():
    # Hand-worked fold from the notes: HPPH folded into a unit square.
    assert moves_to_coords("RUL").tolist() == [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert is_valid(moves_to_coords("RUL"))
    # Bead 3 lands back on bead 1.
    assert not is_valid(moves_to_coords("RUD"))
    # A closed square returns bead 4 to the origin.
    assert not is_valid(moves_to_coords("RULD"))
    assert is_valid(moves_to_coords("R" * 20))

    # Non-trivial check: try every one of the 4^n move strings and count the
    # legal ones. These must equal the known self-avoiding-walk counts on the
    # square lattice (OEIS A001411), which no hand-written case could fake.
    for n in range(1, 7):
        expected = SAW_COUNTS[n]
        count = sum(is_valid(moves_to_coords(m))
                    for m in itertools.product("UDLR", repeat=n))
        assert count == expected, (n, count, expected)
        print(f"  {n} moves: {count:4d} legal folds of {4**n:5d} strings (expected {expected})")
    print("step 1 checks passed")


def random_sequence(N, rng):
    """Random HP sequence with 40-50% H, H positions shuffled uniformly."""
    k_min, k_max = -(-2 * N // 5), N // 2          # ceil(0.4 N), floor(0.5 N)
    assert k_min <= k_max, f"no H count in 40-50% for N = {N}"
    k = rng.integers(k_min, k_max + 1)
    seq = np.array(["P"] * N)
    seq[rng.permutation(N)[:k]] = "H"
    return "".join(seq)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seq", nargs="?", default="HHPPHPPHPH",
                        help="HP sequence; its length sets N (default: 10 beads)")
    parser.add_argument("--random", type=int, metavar="N",
                        help="ignore seq and draw a random N-bead sequence with 40-50%% H")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --random")
    args = parser.parse_args()

    if args.random is not None:
        seq = random_sequence(args.random, np.random.default_rng(args.seed))
    else:
        seq = args.seq.upper()
    assert set(seq) <= {"H", "P"}, f"sequence may only contain H and P, got {seq!r}"
    N = len(seq)
    print(f"sequence {seq}, N = {N}, H fraction = {seq.count('H') / N:.0%}")

    # Square-lattice parity: beads i, j can touch only if i + j is odd, so if
    # every H has the same index parity no H-H contact is possible and E = 0.
    h_idx = [i for i, s in enumerate(seq) if s == "H"]
    n_even = sum(i % 2 == 0 for i in h_idx)
    assert 0 < n_even < len(h_idx), "all H share index parity: E = 0 for every fold, pick another seed"

    check_step1()
    rng = np.random.default_rng(args.seed)
    check_step2(seq, rng)
    E_sample = plot_step2(seq, rng, Path(__file__).with_name(f"step2_{seq}.png"))

    if N > BRUTE_FORCE_MAX:
        print(f"N = {N} > {BRUTE_FORCE_MAX}: skipping brute force (step 3)")
    else:
        folds = enumerate_folds(seq)
        check_step3(seq, folds, E_sample, rng)
        plot_step3(seq, folds, E_sample, Path(__file__).with_name(f"step3_{seq}.png"))
        if N > GRAPH_MAX:
            print(f"N = {N} > {GRAPH_MAX}: skipping move-graph checks (step 4)")
        else:
            out = check_step4(seq, folds, rng)
            plot_step4(seq, folds, out, Path(__file__).with_name(f"step4_{seq}.png"))
        out = check_step5(seq, folds, rng)
        hits = time_to_ground(seq, min(e for _, e in folds), rng)
        plot_step5(seq, folds, out, hits, Path(__file__).with_name(f"step5_{seq}.png"))


if __name__ == "__main__":
    main()
