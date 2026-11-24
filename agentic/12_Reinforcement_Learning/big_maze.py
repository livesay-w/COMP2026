"""The 30 x 30 generated maze used from step 3.5 on: settings shared by every script.

Seed 11 was chosen by sweeping seeds 0-19 of these generator settings for a maze whose
least-energy route (i) heads away from the goal at least 6 times, (ii) takes at least 3 detour
steps (moves that lengthen the route) to save energy, and (iii) saves more than 15% energy
over the fewest-moves route. step3_5_big_maze.py re-asserts all three.
"""
import numpy as np

from maze_env import EnergyMaze
from terrain_gen import make_maze
from qlearning import q_learning

MAZE = dict(n=30, seed=11, room=4, knockout=0.3, n_hills=40, widths=(1.2, 2.5), tilt=1.0)
C_STEP = 0.1        # 1 unit of climb costs as much as 10 flat steps, so detours can pay off
N_EPISODES = 2000
ALPHA = 0.5
MAX_STEPS = 3000    # a random walk here needs ~17,500 steps on average (step 3.5c)


def big_env(max_steps=MAX_STEPS):
    terrain, start, goal = make_maze(**MAZE)
    return EnergyMaze(terrain=terrain, start=start, goal=goal, c_step=C_STEP, max_steps=max_steps)


def big_run():
    env = big_env()
    return env, *q_learning(env, N_EPISODES, ALPHA, np.random.default_rng(0))
