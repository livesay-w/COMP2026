"""Step 1: the exact least-energy solution by value iteration (the learner's answer key).

Natural units m = g = 1. gamma = 1: we want the actual total energy, not a discounted one.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from maze_env import EnergyMaze, TERRAIN
from exact import (value_iteration, state_values, greedy_path, bfs_distances,
                   dijkstra_cost_to_go, fewest_moves_actions, replay)
from plots import draw_terrain, draw_path, BLUE, ORANGE, MUTED

env = EnergyMaze()
Q = value_iteration(env)
V = state_values(env, Q)              # V(s) = -(least energy from s to the goal)
open_ = ~env.walls.ravel()
h = env.h.ravel()
d = bfs_distances(env, env.goal)

# --- Check 1: flat map (all heights 0, same walls) -> V(s) = -c_step * (fewest moves) ----
flat = EnergyMaze(terrain=["".join("#" if ch == "#" else "0" for ch in row) for row in TERRAIN])
V_flat = state_values(flat, value_iteration(flat))
assert np.allclose(V_flat[open_], -flat.c_step * d[open_], atol=1e-12, rtol=0)
print("check 1 passed: flat map, V*(s) = -c_step x BFS distance in every cell")

# --- Check 2: Dijkstra, a different algorithm, gives the same least energy everywhere ----
assert np.allclose(-V[open_], dijkstra_cost_to_go(env)[open_], atol=1e-12, rtol=0)
print("check 2 passed: value iteration and Dijkstra agree in every cell")

# --- Check 3: conservative force (eta = 1) -> energy is path-independent, so the optimum
# is the fewest-moves path:  V(s) = -(h_goal - h(s)) - c_step * BFS distance.
# (c_step > 0 is needed here: with c_step = 0 AND eta = 1 there are zero-cost loops, and with
#  gamma = 1 the problem has no unique solution - the agent could wander forever for free.)
conservative = EnergyMaze(eta=1.0)
V_cons = state_values(conservative, value_iteration(conservative))
expected = -(h[env.goal] - h) - conservative.c_step * d
assert np.allclose(V_cons[open_], expected[open_], atol=1e-12, rtol=0)
print("check 3 passed: eta = 1, elevation drops out of the route choice (V* = -dh - c_step x BFS)")

# --- Check 4: the greedy path from Q* costs exactly -V*(start), and that energy decomposes
# into c_step x steps + total ascent. With eta = 0 the path climbs only the unavoidable
# h_goal - h_start: it never wastes height.
path, energy = greedy_path(env, Q)
dh = np.diff(h[path])
ascent = dh[dh > 0].sum()
assert np.isclose(energy, -V[env.start])
assert np.isclose(energy, env.c_step * (len(path) - 1) + ascent)
assert ascent == h[env.goal] - h[env.start]
print(f"check 4 passed: greedy path energy {energy} = -V*(start) = "
      f"{env.c_step} x {len(path) - 1} steps + {ascent:.0f} ascent")

short_path, short_energy = replay(env, env.start, fewest_moves_actions(env, env.start, env.goal))
dh_s = np.diff(h[short_path])
print(f"\nleast-energy path : {len(path) - 1:2d} steps, ascent {ascent:.0f}, energy {energy:.2f}")
print(f"fewest-moves path : {len(short_path) - 1:2d} steps, ascent {dh_s[dh_s > 0].sum():.0f}, "
      f"energy {short_energy:.2f}")

# --- Figure ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
draw_terrain(ax1, env)
draw_path(ax1, env, path, BLUE, f"least energy ({energy:.2f}, {len(path) - 1} steps)", -0.1)
draw_path(ax1, env, short_path, ORANGE,
          f"fewest moves ({short_energy:.2f}, {len(short_path) - 1} steps)", 0.1)
ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8, frameon=False)
ax1.set_title(f"Exact paths (eta = {env.eta}, c_step = {env.c_step})", fontsize=10)

E_to_go = np.ma.masked_array((0.0 - V).reshape(env.h.shape), env.walls)   # 0.0 - V avoids "-0.0"
im = ax2.imshow(E_to_go, cmap="Blues")
fig.colorbar(im, ax=ax2, shrink=0.8, label="least energy still to spend, -V*(s)")
for (r, c), e in np.ndenumerate(E_to_go):
    if not env.walls[r, c]:
        ax2.text(c, r, f"{e:.1f}", ha="center", va="center", fontsize=6.5,
                 color="white" if e > E_to_go.max() / 2 else "black")
ax2.tick_params(length=0, labelsize=8, labelcolor=MUTED)
ax2.set_title("Energy-to-go from value iteration", fontsize=10)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "step1_exact.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'step1_exact.png'}")
