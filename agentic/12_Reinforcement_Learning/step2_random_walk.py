"""Step 2: random-walk baseline.

A uniformly random agent, run through the same interface the learner will use
(env.reset / env.step), compared with the exact absorbing-Markov-chain solution.
Under the random policy the walk is a Markov chain with transition matrix P among the
non-goal cells (the goal absorbs). The exact results are
    mean steps to goal    t = (I - P)^-1 1
    mean energy to goal   e = (I - P)^-1 c,    c(s) = mean energy of one random step from s
    survival probability  S(t) = Prob(not at goal after t steps) = [P^t 1](start)
and S(t) decays as lambda_max(P)^t: the slowest "diffusion mode" of the walk.
Natural units m = g = 1.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from maze_env import EnergyMaze
from exact import random_walk_exact, value_iteration, state_values
from plots import BLUE, ORANGE, INK, MUTED

rng = np.random.default_rng(0)
env = EnergyMaze(max_steps=10**9)   # no time limit: we want the full time to reach the goal
N_EPISODES = 4000

# --- Exact solution of the absorbing Markov chain -----------------------------------------
P, cells, pos, t_exact, e_exact = random_walk_exact(env)
i0 = pos[env.start]

# --- Simulation through the agent interface ------------------------------------------------
steps, energy = np.zeros(N_EPISODES, dtype=int), np.zeros(N_EPISODES)
for k in range(N_EPISODES):
    env.reset()
    terminated = False
    while not terminated:
        _, r, terminated, truncated = env.step(int(rng.integers(env.n_actions)))
        assert not truncated
        steps[k] += 1
        energy[k] -= r

# --- Check 1: mean steps and mean energy agree with the exact solution within 3 std errors
for name, sim, exact in [("steps", steps, t_exact[i0]), ("energy", energy, e_exact[i0])]:
    se = sim.std(ddof=1) / np.sqrt(N_EPISODES)
    z = (sim.mean() - exact) / se
    assert abs(z) < 3, f"{name}: simulated {sim.mean():.1f} vs exact {exact:.1f} ({z:.1f} sigma)"
    print(f"check 1 passed: mean {name:6s} simulated {sim.mean():7.1f} +- {se:4.1f}, "
          f"exact {exact:7.1f}  ({z:+.1f} sigma)")

# --- Check 2: the whole distribution, not just the mean. Kolmogorov-Smirnov distance between
# the simulated and exact survival curves, below the 99% critical value 1.63/sqrt(N).
T_MAX = steps.max()
S_exact = np.empty(T_MAX + 1)
v = np.ones(len(cells))
for t in range(T_MAX + 1):
    S_exact[t] = v[i0]
    v = P @ v
S_sim = (steps[None, :] > np.arange(T_MAX + 1)[:, None]).mean(axis=1)
ks = np.abs(S_sim - S_exact).max()
assert ks < 1.63 / np.sqrt(N_EPISODES), f"KS distance {ks:.4f} too large"
print(f"check 2 passed: KS distance {ks:.4f} < 99% critical value {1.63 / np.sqrt(N_EPISODES):.4f}")

# --- Check 3: the late-time decay rate of S(t) is the leading eigenvalue of P --------------
lam = np.abs(np.linalg.eigvals(P)).max()
late_rate = S_exact[T_MAX] / S_exact[T_MAX - 1]
assert np.isclose(late_rate, lam, rtol=1e-6, atol=0)
print(f"check 3 passed: S(t+1)/S(t) -> lambda_max(P) = {lam:.5f}, decay time "
      f"tau = -1/ln(lambda) = {-1 / np.log(lam):.0f} steps")

E_opt = -state_values(env, value_iteration(env))[env.start]
print(f"\nrandom walk spends {energy.mean():.0f} energy in {steps.mean():.0f} steps on average; "
      f"the least-energy path spends {E_opt:.2f} in 21 steps ({energy.mean() / E_opt:.0f}x)")

# --- Figure ---------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

t_map = np.full(env.n_states, np.nan)
t_map[cells] = t_exact
t_map[env.goal] = 0.0
im = ax1.imshow(np.ma.masked_invalid(t_map.reshape(env.h.shape)), cmap="Blues")
fig.colorbar(im, ax=ax1, shrink=0.85, label="mean random-walk steps to the goal")
for (r, cc), t in np.ndenumerate(t_map.reshape(env.h.shape)):
    if np.isfinite(t):
        ax1.text(cc, r, f"{t:.0f}", ha="center", va="center", fontsize=6.5,
                 color="white" if t > np.nanmax(t_map) / 2 else INK)
ax1.text(env.cell(env.start)[1], env.cell(env.start)[0] + 0.32, "S", ha="center",
         va="center", fontsize=6, color="white", fontweight="bold")
ax1.tick_params(length=0, labelsize=8, labelcolor=MUTED)
ax1.set_title("Exact mean time to reach the goal by random walk\n(white = wall)", fontsize=10)

t_axis = np.arange(T_MAX + 1)
ax2.semilogy(t_axis, S_sim, color=BLUE, lw=2, label=f"simulated ({N_EPISODES} episodes)")
ax2.semilogy(t_axis, S_exact, color=ORANGE, lw=1.5, ls="--", label=r"exact $[P^t\,1]$(start)")
ax2.semilogy(t_axis, S_exact[T_MAX] * lam ** (t_axis - T_MAX), color=MUTED, lw=1, ls=":",
             label=rf"$\lambda_{{max}}^t$ tail, $\tau$ = {-1 / np.log(lam):.0f} steps")
ax2.set_ylim(1 / N_EPISODES, 1.5)
ax2.set_xlabel("steps t", color=INK)
ax2.set_ylabel("Prob(goal not yet reached)", color=INK)
ax2.tick_params(labelsize=8, labelcolor=MUTED)
ax2.grid(color="#e4e3df", lw=0.6)
for side in ("top", "right"):
    ax2.spines[side].set_visible(False)
ax2.legend(fontsize=8, frameon=False)
ax2.set_title("Survival curve of the random walker", fontsize=10)

out = Path(__file__).parent / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "step2_random_walk.png", dpi=150, bbox_inches="tight")
print(f"saved {out / 'step2_random_walk.png'}")
