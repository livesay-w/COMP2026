"""Step 5: many exits. Does the agent settle for a lesser exit before finding the best one?

30 x 30 maze of big_maze.py with N_EXITS exits. Reaching exit k ends the episode and pays prize
R_k, so committing to exit k is worth  v_k = R_k - (least energy from the start to k).
Prizes are set so the farthest exit is the best (v = 7) and the nearest is a tempting second
best (v = 5, the trap); the rest get v uniform in [0.5, 4].

With prizes, Q0 = 0 is no longer optimistic: once a prize is found its route looks better than
any untried move, so a greedy agent stops exploring. Q0 = R_max (the biggest prize) is optimistic:
every true Q*(s,a) < R_max, so untried moves always look at least as good as anything known.
Natural units m = g = 1, eta = 0, gamma = 1.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # one BLAS thread per worker (7 GB laptop)

import multiprocessing as mp
from pathlib import Path

import numpy as np

from big_maze import MAZE, C_STEP, MAX_STEPS
from maze_env import EnergyMaze
from terrain_gen import make_maze
from exact import value_iteration

N_EXITS, EXIT_SEED = 15, 5
N_EPISODES, ALPHA, SEEDS = 1500, 0.5, range(10)


class PrizeMaze(EnergyMaze):
    """EnergyMaze with several exits: entering exit k ends the episode and pays prize R_k."""

    def __init__(self, exits, prizes, **kwargs):
        super().__init__(**kwargs)
        self.prize = {self.index(cell): R for cell, R in zip(exits, prizes)}

    def transition(self, s, a):
        s_next, r, _ = super().transition(s, a)
        if s_next in self.prize:
            return s_next, r + self.prize[s_next], True
        return s_next, r, False


def least_energy_to(terrain, start, exits, k):
    """Least energy from start to exit k with the other exits walled off (an agent heading
    for k cannot pass through another exit: the episode would end there)."""
    walled = [list(row) for row in terrain]
    for j, (r, c) in enumerate(exits):
        if j != k:
            walled[r][c] = "#"
    env = EnergyMaze(terrain=["".join(row) for row in walled], start=start, goal=exits[k],
                     c_step=C_STEP)
    return -value_iteration(env)[env.start].max()


def build():
    """Returns (env, exits, v) with v[k] = exact value of committing to exit k."""
    terrain, start, _ = make_maze(**MAZE)
    n, rng = MAZE["n"], np.random.default_rng(EXIT_SEED)
    dist = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
    room_cells = [(r, c) for r in range(n) for c in range(n) if terrain[r][c] != "#"
                  and r % 5 != 4 and c % 5 != 4 and dist((r, c), start) >= 4]   # not doorways
    exits = []
    for i in rng.permutation(len(room_cells)):
        if all(dist(room_cells[i], e) >= 5 for e in exits):
            exits.append(room_cells[i])
        if len(exits) == N_EXITS:
            break
    E = np.array([least_energy_to(terrain, start, exits, k) for k in range(N_EXITS)])
    v = rng.uniform(0.5, 4.0, N_EXITS)
    v[np.argmax(E)], v[np.argmin(E)] = 7.0, 5.0       # best = farthest, trap = nearest
    best = int(np.argmax(v))
    env = PrizeMaze(exits, E + v, terrain=terrain, start=start, goal=exits[best], c_step=C_STEP,
                    max_steps=MAX_STEPS)
    return env, exits, v


def greedy_outcome(env, Q):
    """Follow argmax Q from the start: (exit state reached, return), or (None, -inf) if it loops."""
    s, G, seen = env.start, 0.0, {env.start}
    while True:
        s, r, terminated = env.transition(s, int(np.argmax(Q[s])))
        G += r
        if terminated:
            return s, G
        if s in seen:
            return None, -np.inf
        seen.add(s)


def train(env, q0, eps, rng):
    """Q-learning with constant eps. Records, after every episode, which exit the greedy policy
    leads to, plus how often each cell was visited."""
    Q = np.full((env.n_states, env.n_actions), q0)
    visits = np.zeros(env.n_states, dtype=int)
    exit_hist = np.full(N_EPISODES, -1)
    for k in range(N_EPISODES):
        s = env.reset()
        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s_next, r, terminated, truncated = env.step(a)
            target = r if terminated else r + Q[s_next].max()
            Q[s, a] += ALPHA * (target - Q[s, a])
            visits[s_next] += 1
            s = s_next
            if terminated or truncated:
                break
        ex, _ = greedy_outcome(env, Q)
        exit_hist[k] = -1 if ex is None else ex
    return Q, visits, exit_hist


def strategies(R_max):
    return {"Q0 = 0, greedy": (0.0, 0.0), "Q0 = 0, eps = 0.1": (0.0, 0.1),
            "Q0 = R_max, greedy": (R_max, 0.0), "Q0 = R_max, eps = 0.1": (R_max, 0.1)}


def run(job):
    name, seed = job
    env, exits, v = build()
    q0, eps = strategies(max(env.prize.values()))[name]
    Q, visits, exit_hist = train(env, q0, eps, np.random.default_rng(seed))
    return name, seed, Q, visits, exit_hist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plots import draw_terrain, path_xy, BLUE, ORANGE, INK, MUTED, GRID

    env, exits, v = build()
    exit_states = [env.index(e) for e in exits]
    rank = {s: int(r) for s, r in zip(exit_states, np.argsort(np.argsort(-v)) + 1)}   # 1 = best
    best = exit_states[int(np.argmax(v))]
    E_exit = np.array([env.prize[s] for s in exit_states]) - v     # least energy to each exit
    trap = exit_states[int(np.argmin(E_exit))]
    R_max = max(env.prize.values())
    STRATS = strategies(R_max)

    # --- Check 1: the exact optimum commits to the best exit, and V*(start) = max_k v_k --------
    Q_star = value_iteration(env)
    ex_star, G_star = greedy_outcome(env, Q_star)
    assert ex_star == best and np.isclose(G_star, v.max()) and np.isclose(Q_star[env.start].max(), v.max())
    print(f"check 1 passed: exact optimum heads for the best exit (rank 1, v = {v.max():.2f}); "
          f"V*(start) = max_k v_k")
    print("exit values v_k, best first:", np.round(np.sort(v)[::-1], 2))
    print(f"nearest exit (the trap): rank {rank[trap]}, energy {E_exit.min():.1f} away; "
          f"best exit: energy {E_exit[int(np.argmax(v))]:.1f} away\n")

    jobs = [(name, seed) for name in STRATS for seed in SEEDS]
    with mp.Pool(min(6, mp.cpu_count())) as pool:
        results = pool.map(run, jobs, chunksize=1)

    # --- Check 2: optimistic runs keep Q >= Q* (outside walls and exits) and all end on the best
    # exit, with the exact optimal return when rolled out through env.step.
    inside = ~env.walls.ravel()
    inside[exit_states] = False
    final = {name: [] for name in STRATS}
    for name, seed, Q, visits, exit_hist in results:
        final[name].append(rank.get(exit_hist[-1], None))
        if name.startswith("Q0 = R_max"):
            assert (Q[inside] >= Q_star[inside] - 1e-9).all(), (name, seed)
            s, G, done = env.reset(), 0.0, False
            while not done:
                s, r, done, truncated = env.step(int(np.argmax(Q[s])))
                assert not truncated
                G += r
            assert s == best and np.isclose(G, v.max(), atol=1e-9, rtol=0), (name, seed, G)
    print("check 2 passed: every Q0 = R_max run keeps Q >= Q* and ends on the best exit with "
          "the exact optimal return\n")
    for name in STRATS:
        ranks = final[name]
        print(f"  {name:>22}: final exit rank per seed {ranks}  -> best exit in "
              f"{sum(r == 1 for r in ranks)}/{len(ranks)} runs")

    # --- Figure 1: the exits, and where two agents explored -----------------------------------
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    colors = dict(zip(STRATS, [BLUE, ORANGE, "#1baf7a", "#eda100"]))

    def draw_exits(ax):
        for s in exit_states:
            r, c = env.cell(s)
            ax.plot(c, r, "s", ms=9, mfc="white", mec=INK, mew=1.2)
            ax.text(c, r, str(rank[s]), ha="center", va="center", fontsize=6.5, color=INK,
                    fontweight="bold" if s == best else "normal")

    def route(Q):
        s, path = env.start, [env.start]
        while s not in exit_states and len(path) < 500:
            s = env.transition(s, int(np.argmax(Q[s])))[0]
            path.append(s)
        return path

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.4))
    draw_terrain(axes[0], env, numbers=False)
    xy = path_xy(env, route(Q_star))
    axes[0].plot(xy[:, 0], xy[:, 1], color=INK, lw=2, label="exact optimal route")
    draw_exits(axes[0])
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.04), fontsize=8, frameon=False)
    axes[0].set_title(f"{N_EXITS} exits, labelled by value rank (1 = best)\n"
                      f"nearest exit = rank {rank[trap]} (the trap)", fontsize=10)
    for ax, name in zip(axes[1:], ["Q0 = 0, greedy", "Q0 = R_max, greedy"]):
        _, _, Q, visits, exit_hist = next(r for r in results if r[0] == name and r[1] == 0)
        draw_terrain(ax, env, numbers=False)
        vis = np.ma.masked_equal(visits.reshape(env.h.shape), 0)
        im = ax.imshow(np.ma.log10(vis), cmap="Blues", alpha=0.75, vmin=0,
                       vmax=np.log10(max(r[3].max() for r in results)))
        xy = path_xy(env, route(Q))
        ax.plot(xy[:, 0], xy[:, 1], color=colors[name], lw=2.2, label="final greedy route")
        draw_exits(ax)
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.04), fontsize=8, frameon=False)
        explored = 100 * (visits > 0).sum() / (~env.walls).sum()
        ax.set_title(f"{name} (seed 0): ends on exit rank {rank.get(exit_hist[-1])}\n"
                     f"blue = cells visited ({explored:.0f}% of the maze)", fontsize=10)
    fig.colorbar(im, ax=axes[1:], shrink=0.7, label="log10(visits)")
    fig.savefig(out / "step5_exits_map.png", dpi=150, bbox_inches="tight")
    print(f"\nsaved {out / 'step5_exits_map.png'}")

    # --- Figure 2: commitment over time --------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5),
                                        gridspec_kw={"width_ratios": [1.2, 1.2, 0.8], "wspace": 0.45})
    ep = np.arange(1, N_EPISODES + 1)
    for i, name in enumerate(STRATS):
        _, _, _, _, exit_hist = next(r for r in results if r[0] == name and r[1] == 0)
        y = np.array([rank.get(e, N_EXITS + 1.5) for e in exit_hist], dtype=float)  # none yet
        ax1.step(ep, y + (i - 1.5) * 0.12, where="post", color=colors[name], lw=1.6, label=name)
        frac = np.mean([[rank.get(e) == 1 for e in r[4]] for r in results if r[0] == name], axis=0)
        ax2.plot(ep, frac, color=colors[name], lw=2, label=name)
    ax1.set_ylim(N_EXITS + 2.2, 0.3)
    ax1.set_yticks(list(range(1, N_EXITS + 1)) + [N_EXITS + 1.5])
    ax1.set_yticklabels([str(k) for k in range(1, N_EXITS + 1)] + ["none yet\n(exploring)"])
    ax1.set_xscale("log")
    ax1.set_xlabel("episode", color=INK)
    ax1.set_ylabel("rank of the exit the greedy policy leads to (1 = best)", color=INK)
    ax1.set_title("Which exit is the agent committed to? (seed 0)", fontsize=10)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel("episode", color=INK)
    ax2.set_ylabel(f"fraction of the {len(SEEDS)} seeds heading for the best exit", color=INK)
    ax2.set_title("Committed to the best exit, across seeds\n(both Q0 = 0 lines stay at 0)",
                  fontsize=10)
    for i, name in enumerate(STRATS):
        ranks = np.array([np.nan if r is None else r for r in final[name]], dtype=float)
        ax3.plot(ranks, np.full(len(ranks), i) + np.linspace(-0.25, 0.25, len(ranks)), "o",
                 color=colors[name], ms=7, mec="white")
    ax3.set_yticks(range(len(STRATS)))
    ax3.set_yticklabels(list(STRATS), fontsize=8)
    ax3.set_xlim(0.5, N_EXITS + 0.5)
    ax3.set_xticks(range(1, N_EXITS + 1))
    ax3.invert_yaxis()
    ax3.set_xlabel("final exit rank (1 = best)", color=INK)
    ax3.set_title(f"Where each run ended ({N_EPISODES} episodes)", fontsize=10)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(labelsize=8, labelcolor=MUTED)
        ax.grid(color=GRID, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    ax1.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.45))
    fig.savefig(out / "step5_exits_commitment.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'step5_exits_commitment.png'}")
