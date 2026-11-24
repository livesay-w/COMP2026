"""Step 3.5: the same Q-learner on a generated 30 x 30 maze of rooms, with dead ends, loops
and hills (settings and seed in big_maze.py).

  a. the maze: every open cell reachable; dead-end pockets exist
  b. exact answer: value iteration = Dijkstra; the route has the features asked for
     (heads away from the goal, detours to save energy, beats the fewest-moves route)
  c. how hard: exact mean random-walk time to the goal
  d. Q-learning (step 3 algorithm and checks, more steps per episode)
Natural units m = g = 1, eta = 0, gamma = 1.
"""
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from big_maze import big_run, MAZE, C_STEP, N_EPISODES, ALPHA
from exact import (value_iteration, state_values, greedy_path, bfs_distances, dijkstra_cost_to_go,
                   fewest_moves_actions, replay, random_walk_exact)
from qlearning import check_against_exact, epsilon
from plots import (draw_terrain, draw_path, draw_trails, draw_learning_curve, draw_policy,
                   path_xy, BLUE, ORANGE, INK)
from maze_env import MOVES

env, Q, paths, energy, greedy = big_run()
open_ = ~env.walls.ravel()
h = env.h.ravel()
print(f"maze: {env.n_rows} x {env.n_cols}, {open_.sum()} open cells, settings {MAZE}, "
      f"c_step = {C_STEP}\n")


def reachable(walls, source, n_cols):
    """Boolean map of open cells reachable from `source` (breadth-first search on the grid)."""
    seen = np.zeros(walls.shape, dtype=bool)
    seen[source] = True
    queue = deque([source])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES:
            r1, c1 = r + dr, c + dc
            if 0 <= r1 < walls.shape[0] and 0 <= c1 < n_cols and not walls[r1, c1] and not seen[r1, c1]:
                seen[r1, c1] = True
                queue.append((r1, c1))
    return seen


# --- a. The maze ----------------------------------------------------------------------------
goal_cell, start_cell = env.cell(env.goal), env.cell(env.start)
assert (reachable(env.walls, goal_cell, env.n_cols) == ~env.walls).all()
print("check a1 passed: every open cell is reachable from the goal")

# A dead-end pocket: cells cut off from the goal when one cell (its only doorway) is blocked.
# An agent that walks in must come back out the same way. Pockets holding the start are the
# way out, not dead ends, so they are skipped.
pocket = np.zeros(env.walls.shape, dtype=bool)
for s in np.flatnonzero(open_):
    if s == env.goal:
        continue
    blocked = env.walls.copy()
    blocked[env.cell(s)] = True
    cut = ~blocked & ~reachable(blocked, goal_cell, env.n_cols)
    if cut.any() and not cut[start_cell]:
        pocket |= cut
assert pocket.sum() > 0
print(f"check a2 passed: {pocket.sum()} open cells ({100 * pocket.sum() / open_.sum():.0f}%) "
      "lie in dead-end pockets")

# --- b. Exact answer and route features -----------------------------------------------------
Q_star = value_iteration(env)
V_star = state_values(env, Q_star)
assert np.allclose(-V_star[open_], dijkstra_cost_to_go(env)[open_], atol=1e-9, rtol=0)
print("check b1 passed: value iteration and Dijkstra agree in every cell")

path, E_opt = greedy_path(env, Q_star)
short, E_short = replay(env, env.start, fewest_moves_actions(env, env.start, env.goal))
d_maze = bfs_distances(env, env.goal)
d_board = np.abs(np.divmod(np.arange(env.n_states), env.n_cols)[0] - goal_cell[0]) + \
          np.abs(np.divmod(np.arange(env.n_states), env.n_cols)[1] - goal_cell[1])
steps = list(zip(path[:-1], path[1:]))
away = [b for a, b in steps if d_board[b] > d_board[a]]       # farther from the goal on the board
detour = [b for a, b in steps if d_maze[b] > d_maze[a]]       # farther from the goal through the maze
saving = (E_short - E_opt) / E_opt
assert len(away) >= 6 and len(detour) >= 3 and saving > 0.15
print(f"check b2 passed: least-energy route heads away from the goal {len(away)} times, takes "
      f"{len(detour)} detour steps, and saves {100 * saving:.0f}% energy over the fewest-moves route")


def ascent(p):
    dh = np.diff(h[p])
    return dh[dh > 0].sum()


print(f"  least-energy route : {len(path) - 1:3d} steps, ascent {ascent(path):4.0f}, energy {E_opt:6.2f}")
print(f"  fewest-moves route : {len(short) - 1:3d} steps, ascent {ascent(short):4.0f}, energy {E_short:6.2f}")

# --- c. How hard is exploration here? --------------------------------------------------------
P, cells, pos, t_rw, e_rw = random_walk_exact(env)
T_rw, E_rw = t_rw[pos[env.start]], e_rw[pos[env.start]]
print(f"\nc. random walk needs {T_rw:,.0f} steps and {E_rw:,.0f} energy on average to reach the goal "
      f"(10 x 10 maze: 298 steps)\n")

# --- d. Q-learning ----------------------------------------------------------------------------
learned, E_learned = check_against_exact(env, Q, greedy, Q_star)
trunc = [k for k, p in enumerate(paths) if len(p) - 1 == env.max_steps]
print(f"\n{len(trunc)} of {N_EPISODES} episodes hit the {env.max_steps}-step time limit, the last "
      f"at episode {trunc[-1]} (eps = {epsilon(trunc[-1], N_EPISODES):.2f}); "
      f"{sum(len(p) - 1 for p in paths):,} moves in total")

# --- Figure 1: the maze and its exact solution ------------------------------------------------
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.6))
im = draw_terrain(ax1, env, numbers=False)
fig.colorbar(im, ax=ax1, shrink=0.75, label="height h", ticks=range(0, int(h.max()) + 1))
ax1.contour(pocket.astype(float), levels=[0.5], colors=INK, linewidths=1.2, linestyles="--")
draw_path(ax1, env, short, ORANGE, f"fewest moves ({E_short:.2f}, {len(short) - 1} steps)", 0.12, 2)
draw_path(ax1, env, path, BLUE, f"least energy ({E_opt:.2f}, {len(path) - 1} steps)", -0.12, 2)
xy = path_xy(env, away, -0.12)
ax1.plot(xy[:, 0], xy[:, 1], "o", color=BLUE, mec="white", ms=6, mew=1.2,
         label="least-energy step away from the goal")
ax1.plot([], [], color=INK, lw=1.2, ls="--", label="dead-end pockets")
ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.04), ncol=2, fontsize=8, frameon=False)
ax1.set_title("Generated 30 x 30 maze (hatched = wall)", fontsize=10)

E_to_go = np.ma.masked_array((0.0 - V_star).reshape(env.h.shape), env.walls)
im2 = ax2.imshow(E_to_go, cmap="Blues")
fig.colorbar(im2, ax=ax2, shrink=0.75, label="least energy still to spend, -V*(s)")
draw_path(ax2, env, path, ORANGE, "least-energy route", lw=1.5)
ax2.tick_params(length=0, labelsize=8)
ax2.set_title("Energy-to-go from value iteration", fontsize=10)
fig.savefig(out / "step3_5_maze.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'step3_5_maze.png'}")

# --- Figure 2: learning ---------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6),
                                    gridspec_kw={"width_ratios": [1, 1.2, 1]})
draw_terrain(ax1, env, numbers=False)
draw_trails(ax1, env, paths, alpha=0.02, lw=0.6)
draw_path(ax1, env, learned, ORANGE, f"learned greedy path ({E_learned:.2f})", lw=1.8)
ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.04), fontsize=8, frameon=False)
ax1.set_title(f"All {N_EPISODES} attempts (light = early, dark = late)", fontsize=10)

draw_learning_curve(ax2, energy, greedy, E_opt, E_rw)
ax2.set_title(f"Learning curve (alpha = {ALPHA}, eps 1 -> {epsilon(N_EPISODES, N_EPISODES):.2f} "
              f"over {int(0.8 * N_EPISODES)} episodes)", fontsize=10)

err = np.nanmax(np.abs(state_values(env, Q) - V_star))
draw_terrain(ax3, env, numbers=False)
draw_path(ax3, env, learned, ORANGE, "learned path", lw=1.8)
draw_policy(ax3, env, Q)
ax3.set_title(f"Learned policy from every cell\n(max |V - V*| over the map = {err:.0e})", fontsize=10)
fig.savefig(out / "step3_5_q_learning.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'step3_5_q_learning.png'}")
