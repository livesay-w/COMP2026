# RL Energy Maze

An agent starts in a valley and must reach a goal on a higher plateau, crossing a terrain with
an impassable wall, a high ridge with one low saddle, and a pit beside the goal. It is **not
told** where the goal, walls or hills are: it sees only its current cell and the energy each
move costs, and learns the least-energy route by tabular Q-learning.

Natural units m = g = 1. Energy cost of one move with height change dh:

- climbing: `c_step + dh` (full work against gravity)
- descending: `c_step - eta*|dh|` (a fraction eta of the energy is recovered)
- reward = -cost, gamma = 1 (we want the actual total energy, not a discounted one)

eta = 1 is a conservative force: energy is path-independent and elevation cannot affect the
route (potential-based reward shaping). eta = 0 is pure effort: descending is free, but lost
height must be re-climbed, so the optimal route never wastes height. Baseline: eta = 0,
c_step = 0.25 (1 unit of climb = 4 flat steps).

## Must

| Step | File | What | Checks |
|---|---|---|---|
| 0 | `maze_env.py`, `step0_env.py` | Environment | blocked moves stay put and cost c_step; only the goal terminates and time-outs are *truncated*; goal reachable; eta = 1 loops cost 0, eta = 0 loops cost their ascent, eta = 1 start->goal costs h_goal - h_start on every path |
| 1 | `exact.py`, `step1_exact.py` | Exact answer key by value iteration | flat map gives V = -c_step x BFS distance; Dijkstra agrees everywhere; eta = 1 gives V = -dh - c_step x BFS; greedy path energy = -V*(start) = c_step x steps + ascent |
| 2 | `step2_random_walk.py` | Random-walk baseline | simulated mean steps and energy match the exact hitting-time solution of (I - P) t = 1 |
| 3 | `qlearning.py`, `step3_q_learning.py`, `step3_video.py` (GIF of every attempt) | Q-learning, eps-greedy, Q0 = 0 (optimistic because all rewards < 0) | learned greedy path energy = V*(start) exactly; Q -> Q* along the optimal path; learning curve -> V*(start) |

## Step 3.5: 30 x 30 generated maze (used for all later steps)

`terrain_gen.py` builds a maze of 4 x 4 rooms (randomized DFS, so dead-end rooms) with 30% extra
doorways (loops = competing routes), 40 Gaussian hills and a tilt toward the goal. Settings and
seed live in `big_maze.py`; c_step = 0.1 so that detours around hills can pay off. Seed 11 was
chosen from seeds 0-19 because its least-energy route heads away from the goal >= 6 times, takes
>= 3 detour steps and saves > 15% energy over the fewest-moves route; `step3_5_big_maze.py`
re-asserts all three, plus: every cell reachable, dead-end pockets exist, value iteration =
Dijkstra, the exact random-walk time, and the four step 3 learner checks. `step3_5_video.py`
makes the GIF.

## Step 4: cost of learning vs maze size (`step4_exploration.py`, quick version)

Sizes 10-30, 3 seeds, three strategies; steps until the greedy policy is optimal for 20 episodes.
Optimistic agents always succeed, with cost ~ N^1.5; the undirected agent is faster when lucky
but missed the optimum within budget in 3/15 runs. Measures speed, not whether a better goal
is missed (single goal).

## Step 5: many exits (`step5_exits.py`)

15 exits with prizes; exact value of each exit v_k = R_k - least energy (value iteration with the
other exits walled off). Best exit = farthest (v = 7), nearest = rank 2 (the trap). Checks: the
exact optimum heads for argmax v_k with V*(start) = max v_k; every Q0 = R_max run keeps Q >= Q*
and ends on the best exit with the exact return. Result: Q0 = 0 trapped on rank 2 in 20/20 runs
(with or without eps = 0.1); Q0 = R_max finds rank 1 in 20/20.

## Later

4. Exploration choices: eps sweep, Q0, alpha; a time-out treated as termination makes the agent refuse to climb.
5. Sweep eta from 0 to 1; rerun on a smooth seeded random landscape.
6. Three prizes (bronze / silver / gold): bronze trap with Q0 = 0 vs Q0 = G; basins-of-attraction map; gold/silver switch at G* = S + (E_gold - E_silver).
