"""Animation of the step 3.5 Q-learning run on the 30 x 30 maze.

Frames: every episode for the first 30, then every 20th. The faint trail keeps only the
episodes that got a frame (about 1 in 20); drawing all ~850,000 moves on every frame would
be far too slow. Re-runs the seeded training from big_maze.py, so it is the same run as
figures/step3_5_q_learning.png. Saves figures/step3_5_learning.gif.
"""
from pathlib import Path

from exact import value_iteration, state_values, random_walk_exact
from big_maze import big_run, N_EPISODES
from animate import learning_gif

env, Q, paths, energy, greedy = big_run()
E_opt = -state_values(env, value_iteration(env))[env.start]
_, _, pos, _, e_rw = random_walk_exact(env)
frames = list(range(30)) + list(range(30, N_EPISODES, 20)) + [N_EPISODES - 1] * 24  # hold end
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
learning_gif(env, paths, energy, greedy, E_opt, e_rw[pos[env.start]], frames,
             out / "step3_5_learning.gif", trail="shown", numbers=False, alpha=0.15)
