"""Animation of the step 3 Q-learning run (10 x 10): every attempt drawn on the board.

Frames: every episode for the first 30 (where it changes fastest), then every 20th; all earlier
attempts stay as faint lines. Re-runs the seeded step 3 training, so it is the same run as
figures/step3_q_learning.png. Saves figures/step3_learning.gif.
"""
from pathlib import Path

from exact import value_iteration, state_values
from qlearning import baseline_run, N_EPISODES
from animate import learning_gif

E_RANDOM = 199.1   # exact mean energy of the random walker, from step 2

env, Q, paths, energy, greedy = baseline_run()
E_opt = -state_values(env, value_iteration(env))[env.start]
frames = list(range(30)) + list(range(30, N_EPISODES, 20)) + [N_EPISODES - 1] * 24  # hold end
out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
learning_gif(env, paths, energy, greedy, E_opt, E_RANDOM, frames, out / "step3_learning.gif")
