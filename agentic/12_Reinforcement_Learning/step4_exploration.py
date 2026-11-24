"""Step 4 (quick version): how does the cost of learning scale with maze size, per strategy?

For each maze size (generator settings of big_maze.py, plain seeds, hill count scaled with area)
and each exploration strategy, Q-learning runs until its greedy policy has been optimal for
STREAK episodes in a row. We record the number of environment steps taken when that streak
began. Runs that exhaust BUDGET steps are censored (shown, not dropped).
Strategies:
  undirected  eps = 0.2, Q0 = -1000 (pessimistic): untried moves look bad, exploration is random
  mixed       eps = 0.2, Q0 = 0 (optimistic)
  directed    eps = 0,   Q0 = 0: pure greedy optimism, "go where you have not been"
Natural units m = g = 1, eta = 0, gamma = 1.
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   # one BLAS thread per worker (7 GB laptop)

import multiprocessing as mp
from pathlib import Path

import numpy as np

from big_maze import MAZE, C_STEP
from maze_env import EnergyMaze
from terrain_gen import make_maze
from exact import value_iteration, state_values, random_walk_exact
from qlearning import greedy_energy

SIZES = [10, 15, 20, 25, 30]
SEEDS = [0, 1, 2]
STRATEGIES = {"undirected": dict(eps=0.2, q0=-1000.0),
              "mixed": dict(eps=0.2, q0=0.0),
              "directed": dict(eps=0.0, q0=0.0)}
ALPHA, BUDGET, STREAK, MAX_STEPS = 0.5, 1_500_000, 20, 50_000


def build(size, seed):
    settings = {**MAZE, "n": size, "seed": seed, "n_hills": round(MAZE["n_hills"] * size**2 / 900)}
    terrain, start, goal = make_maze(**settings)
    return EnergyMaze(terrain=terrain, start=start, goal=goal, c_step=C_STEP, max_steps=MAX_STEPS)


def learn_until_optimal(env, eps, q0, E_opt, rng):
    """Returns (steps when the final optimal streak began, or None if censored; Q)."""
    Q = np.full((env.n_states, env.n_actions), q0)
    total, streak, began = 0, 0, None
    while total < BUDGET:
        s = env.reset()
        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s_next, r, terminated, truncated = env.step(a)
            target = r if terminated else r + Q[s_next].max()
            Q[s, a] += ALPHA * (target - Q[s, a])
            s, total = s_next, total + 1
            if terminated or truncated or total >= BUDGET:
                break
        if np.isclose(greedy_energy(env, Q), E_opt, atol=1e-9, rtol=0):
            streak += 1
            began = total if streak == 1 else began
            if streak == STREAK:
                return began, Q
        else:
            streak = 0
    return None, Q


def run(job):
    size, seed, name = job
    env = build(size, seed)
    E_opt = -state_values(env, value_iteration(env))[env.start]
    steps, Q = learn_until_optimal(env, **STRATEGIES[name], E_opt=E_opt,
                                   rng=np.random.default_rng(seed))
    if steps is not None:   # independent check: roll out the greedy policy through env.step
        s, E, done = env.reset(), 0.0, False
        while not done:
            s, r, done, truncated = env.step(int(np.argmax(Q[s])))
            assert not truncated
            E -= r
        assert np.isclose(E, E_opt, atol=1e-9, rtol=0), (size, seed, name, E, E_opt)
    return size, seed, name, int((~env.walls).sum()), steps


def random_walk_time(job):
    size, seed = job
    env = build(size, seed)
    _, _, pos, t, _ = random_walk_exact(env)
    return size, seed, t[pos[env.start]]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plots import BLUE, ORANGE, INK, MUTED, GRID

    jobs = [(n, s, name) for n in SIZES for s in SEEDS for name in STRATEGIES]
    jobs.sort(key=lambda j: -j[0])                      # biggest first, for load balance
    with mp.Pool(min(6, mp.cpu_count())) as pool:
        results = pool.map(run, jobs, chunksize=1)
        rw = pool.map(random_walk_time, [(n, s) for n in SIZES for s in SEEDS])
    print("check passed: every uncensored run's greedy policy, rolled out through env.step, "
          "spends exactly the value-iteration optimum\n")

    print(f"{'size':>4} {'open':>5}  " + "  ".join(f"{name:>22}" for name in STRATEGIES))
    for n in SIZES:
        for s in SEEDS:
            row = {name: steps for (nn, ss, name, N, steps) in results if nn == n and ss == s}
            N = next(N for (nn, ss, _, N, _) in results if nn == n and ss == s)
            print(f"{n:>4} {N:>5}  " + "  ".join(
                f"{row[name]:>22,}" if row[name] is not None else f"{'> budget':>22}"
                for name in STRATEGIES))

    # --- Figure: steps to learn vs number of open cells, log-log ------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = {"undirected": MUTED, "mixed": BLUE, "directed": ORANGE}
    print("\nscaling exponents (steps to learn ~ N^p, fit to per-size geometric means):")
    for name in STRATEGIES:
        pts = [(N, steps) for (_, _, nm, N, steps) in results if nm == name]
        done = np.array([(N, st) for N, st in pts if st is not None], dtype=float)
        cens = np.array([N for N, st in pts if st is None], dtype=float)
        if len(done):
            ax.loglog(done[:, 0], done[:, 1], "o", color=colors[name], ms=6, mec="white", mew=1,
                      label=name)
        if len(cens):
            ax.loglog(cens, np.full(len(cens), BUDGET), "^", color=colors[name], ms=7,
                      mfc="none", mew=1.5, label=f"{name}: hit budget (censored)")
        complete = [n for n in SIZES if all(st is not None for (nn, _, nm, _, st) in results
                                            if nn == n and nm == name)]
        if len(complete) >= 3:
            g = np.array([[np.mean([N for (nn, _, nm, N, _) in results if nn == n and nm == name]),
                           np.exp(np.mean([np.log(st) for (nn, _, nm, _, st) in results
                                           if nn == n and nm == name]))] for n in complete])
            p, c = np.polyfit(np.log(g[:, 0]), np.log(g[:, 1]), 1)
            xs = np.linspace(g[:, 0].min(), g[:, 0].max(), 50)
            ax.loglog(xs, np.exp(c) * xs**p, color=colors[name], lw=1.5)
            ax.text(xs[-1] * 1.05, np.exp(c) * xs[-1]**p, f"p = {p:.2f}", color=INK, fontsize=8,
                    va="center")
            print(f"  {name:>10}: p = {p:.2f}  (sizes {complete})")
        else:
            print(f"  {name:>10}: too many censored runs to fit")
    rwN = [next(N for (nn, ss, _, N, _) in results if nn == n and ss == s) for (n, s, _) in rw]
    ax.loglog(rwN, [t for (_, _, t) in rw], "x", color=INK, ms=6,
              label="exact mean random-walk time to goal (reference)")
    ax.axhline(BUDGET, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("open cells N (maze sizes 10 x 10 to 30 x 30)", color=INK)
    ax.set_ylabel("environment steps until the policy is optimal", color=INK)
    ax.tick_params(labelsize=8, labelcolor=MUTED)
    ax.grid(color=GRID, lw=0.6, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title(f"Cost of learning vs maze size ({len(SEEDS)} seeds per size)", fontsize=10)
    out = Path(__file__).parent / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "step4_exploration.png", dpi=150, bbox_inches="tight")
    print(f"saved {out / 'step4_exploration.png'}")
