"""Step 3: tabular Q-learning, epsilon-greedy (eps 1 -> 0.05), Q0 = 0, gamma = 1.

Q0 = 0 is optimistic: every move costs energy, so every true Q*(s,a) < 0, and an untried move
always looks better than a tried one. The agent is graded against the exact Q* from step 1.
Natural units m = g = 1.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from exact import value_iteration, state_values
from qlearning import baseline_run, check_against_exact, epsilon, N_EPISODES, ALPHA
from plots import draw_terrain, draw_path, draw_trails, draw_learning_curve, draw_policy, ORANGE

E_RANDOM = 199.1   # exact mean energy of the random walker, from step 2

env, Q, paths, energy, greedy = baseline_run()
Q_star = value_iteration(env)
path, E_learned = check_against_exact(env, Q, greedy, Q_star)

trunc = [k for k, p in enumerate(paths) if len(p) - 1 == env.max_steps]
print(f"\n{len(trunc)} of {N_EPISODES} episodes hit the {env.max_steps}-step time limit, "
      f"the last at episode {trunc[-1]} (eps = {epsilon(trunc[-1], N_EPISODES):.2f})")

# --- Figure -----------------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2),
                                    gridspec_kw={"width_ratios": [1, 1.25, 1]})
draw_terrain(ax1, env)
draw_trails(ax1, env, paths)
draw_path(ax1, env, path, ORANGE, f"learned greedy path ({E_learned:.2f})")
ax1.legend(loc="upper left", bbox_to_anchor=(0, -0.05), fontsize=8, frameon=False)
ax1.set_title(f"All {N_EPISODES} attempts (light = early, dark = late)", fontsize=10)

draw_learning_curve(ax2, energy, greedy, -state_values(env, Q_star)[env.start], E_RANDOM)
ax2.set_title(f"Learning curve (alpha = {ALPHA}, eps 1 -> "
              f"{epsilon(N_EPISODES, N_EPISODES):.2f} over {int(0.8 * N_EPISODES)} episodes)",
              fontsize=10)

err = np.nanmax(np.abs(state_values(env, Q) - state_values(env, Q_star)))
draw_terrain(ax3, env, numbers=False)
draw_path(ax3, env, path, ORANGE, "learned path")
draw_policy(ax3, env, Q)
ax3.set_title(f"Learned policy from every cell\n(max |V - V*| over the map = {err:.0e})",
              fontsize=10)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "step3_q_learning.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'step3_q_learning.png'}")
