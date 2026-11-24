"""Step 0: the energy maze. Check the rules of motion and the energy bookkeeping.

Natural units m = g = 1: climbing dh costs work dh.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from maze_env import EnergyMaze, MOVES
from exact import bfs_distances, fewest_moves_actions, replay
from plots import draw_terrain

rng = np.random.default_rng(0)
env = EnergyMaze()
open_states = np.flatnonzero(~env.walls.ravel())
h = env.h.ravel()

# --- Check 1: a blocked move (wall or map edge) leaves you in place and costs c_step ------
n_blocked = 0
for s in open_states:
    r, c = env.cell(s)
    for a, (dr, dc) in enumerate(MOVES):
        r1, c1 = r + dr, c + dc
        if not (0 <= r1 < env.n_rows and 0 <= c1 < env.n_cols) or env.walls[r1, c1]:
            s_next, reward, _ = env.transition(s, a)
            assert s_next == s and reward == -env.c_step
            n_blocked += 1
print(f"check 1 passed: all {n_blocked} blocked moves stay put and cost c_step = {env.c_step}")

# --- Check 2: only entering the goal terminates; time-outs are 'truncated', not 'terminated'
for s in open_states:
    for a in range(env.n_actions):
        s_next, _, terminated = env.transition(s, a)
        assert terminated == (s_next == env.goal)
s = env.reset()
route = fewest_moves_actions(env, s, env.goal)
for i, a in enumerate(route):
    s, _, terminated, truncated = env.step(a)
    assert terminated == (i == len(route) - 1) and not truncated
short = EnergyMaze(max_steps=5)
short.reset()
for i in range(5):                       # action 2 = S: the start is on the south edge, so blocked
    s, _, terminated, truncated = short.step(2)
    assert s == short.start and not terminated and truncated == (i == 4)
print("check 2 passed: only the goal terminates; hitting max_steps sets truncated, not terminated")

# --- Check 3: the goal is reachable -----------------------------------------------------
d_start = bfs_distances(env, env.goal)[env.start]
assert np.isfinite(d_start)
print(f"check 3 passed: goal reachable, fewest moves start -> goal = {d_start:.0f}")

# --- Check 4: energy bookkeeping on random closed loops and on start -> goal paths -------
# Physics: with eta = 1 (conservative) the work around any closed loop is exactly 0 and the
# work from start to goal is h_goal - h_start whatever the path. With eta = 0 the cost of a
# loop is its total ascent, and around a closed loop total ascent = total descent.
conservative = EnergyMaze(c_step=0.0, eta=1.0)
effort = EnergyMaze(c_step=0.0, eta=0.0)
for trial in range(200):
    s0 = int(rng.choice(open_states))
    outward = list(rng.integers(env.n_actions, size=200))
    s_far = replay(env, s0, outward)[0][-1]
    loop = outward + fewest_moves_actions(env, s_far, s0)

    states, E_cons = replay(conservative, s0, loop)
    assert states[-1] == s0 and abs(E_cons) < 1e-12

    states, E_effort = replay(effort, s0, loop)
    dh = np.diff(h[states])
    assert E_effort == dh[dh > 0].sum() == -dh[dh < 0].sum()

E_goal = []
for trial in range(20):                   # random walks that wander until they hit the goal
    s, E = env.start, 0.0
    while s != env.goal:
        s, r, _ = conservative.transition(s, int(rng.integers(env.n_actions)))
        E -= r
    E_goal.append(E)
E_goal.append(replay(conservative, env.start, fewest_moves_actions(env, env.start, env.goal))[1])
assert np.allclose(E_goal, h[env.goal] - h[env.start], atol=1e-12, rtol=0)
print("check 4 passed: eta = 1 loops cost 0; eta = 0 loops cost their ascent (= descent); "
      f"eta = 1 start -> goal costs h_goal - h_start = {h[env.goal] - h[env.start]:.0f} "
      f"on all {len(E_goal)} paths tried")

# --- Figure ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
im = draw_terrain(ax, env)
fig.colorbar(im, ax=ax, shrink=0.8, label="height h", ticks=range(0, int(h.max()) + 1))
ax.set_title(f"Energy maze: start h = {h[env.start]:.0f}, goal h = {h[env.goal]:.0f}\n"
             "ridge with a saddle at (5,8), pit beside the goal, hatched = wall", fontsize=9)
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "step0_terrain.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'step0_terrain.png'}")
