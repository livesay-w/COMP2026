"""Tabular Q-learning with epsilon-greedy exploration (gamma = 1).

The agent sees only what env.reset() / env.step() return: integer states, rewards, and the
terminated / truncated flags. It never reads the height map, the walls or the goal.
"""
import numpy as np

from maze_env import EnergyMaze

# The step 3 run. The figure and the video both call baseline_run(), so they show the same run.
N_EPISODES = 2000
ALPHA = 0.5        # learning rate; a large value is safe because the terrain is deterministic
MAX_STEPS = 500    # episode time limit (a time-out is truncation, not termination)


def baseline_run():
    env = EnergyMaze(max_steps=MAX_STEPS)
    return env, *q_learning(env, N_EPISODES, ALPHA, np.random.default_rng(0))


def epsilon(k, n_episodes, eps_start=1.0, eps_end=0.05, decay_fraction=0.8):
    """Linear decay from eps_start to eps_end over the first decay_fraction of training."""
    return max(eps_end, eps_start - (eps_start - eps_end) * k / (decay_fraction * n_episodes))


def greedy_energy(env, Q):
    """Energy of following argmax Q from the start; inf if the greedy policy loops
    (early in training the greedy policy often does not reach the goal at all)."""
    s, energy, seen = env.start, 0.0, {env.start}
    while s != env.goal:
        s, r, _ = env.transition(s, int(np.argmax(Q[s])))
        if s in seen:
            return np.inf
        seen.add(s)
        energy -= r
    return energy


def check_against_exact(env, Q, greedy, Q_star):
    """The four step 3 checks of a learned Q against the exact Q* (asserts; prints results).
    Returns (learned greedy path, its energy)."""
    from exact import state_values, greedy_path
    E_opt = -state_values(env, Q_star)[env.start]

    # 1: the greedy policy, run through the agent interface, spends exactly the least energy
    s, E_learned, n_steps, terminated = env.reset(), 0.0, 0, False
    while not terminated:
        s, r, terminated, truncated = env.step(int(np.argmax(Q[s])))
        assert not truncated, "greedy policy never reached the goal"
        E_learned -= r
        n_steps += 1
    assert np.isclose(E_learned, E_opt, atol=1e-9, rtol=0)
    print(f"check 1 passed: learned greedy policy spends {E_learned:.2f} in {n_steps} steps "
          f"= exact least energy {E_opt:.2f}")

    # 2: optimism is never lost. If Q >= Q* before an update, the target r + max Q(s') >=
    # r + max Q*(s') = Q*(s,a), so Q >= Q* after it. A sign error or a wrong max breaks this.
    gap = (Q - Q_star)[~env.walls.ravel()]
    assert (gap >= -1e-9).all(), f"Q fell below Q* by {-gap.min()}"
    print("check 2 passed: Q(s,a) >= Q*(s,a) for every state and move (optimism preserved)")

    # 3: along the learned path the agent knows the exact remaining energy
    path, _ = greedy_path(env, Q)
    best = [int(np.argmax(Q[s])) for s in path[:-1]]
    assert max(abs(Q[s, a] - Q_star[s, a]) for s, a in zip(path[:-1], best)) < 1e-9
    print(f"check 3 passed: |Q - Q*| < 1e-9 at every step of the learned path ({len(path) - 1} steps)")

    # 4: the greedy policy becomes optimal and stays optimal to the end of training
    optimal = np.isclose(greedy, E_opt, atol=1e-9, rtol=0)
    n_last = len(greedy) // 10
    assert optimal[-n_last:].all()
    first, settled = int(np.argmax(optimal)), int(np.flatnonzero(~optimal)[-1]) + 1
    print(f"check 4 passed: greedy policy first optimal at episode {first}, optimal in every "
          f"episode from {settled} on (incl. the last {n_last})")
    return path, E_learned


def q_learning(env, n_episodes, alpha, rng, q0=0.0, **eps_kwargs):
    """Returns Q, the list of state paths (one per episode), the energy spent per episode,
    and the energy of the greedy policy after each episode."""
    Q = np.full((env.n_states, env.n_actions), q0)
    paths, energy, greedy = [], np.zeros(n_episodes), np.zeros(n_episodes)
    for k in range(n_episodes):
        eps = epsilon(k, n_episodes, **eps_kwargs)
        s = env.reset()
        path = [s]
        while True:
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))                    # explore
            else:
                a = int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))  # exploit, random tie-break
            s_next, r, terminated, truncated = env.step(a)
            # Reaching the goal ends all future cost; a time-out does not, so bootstrap from s'.
            target = r if terminated else r + Q[s_next].max()
            Q[s, a] += alpha * (target - Q[s, a])
            energy[k] -= r
            s = s_next
            path.append(s)
            if terminated or truncated:
                break
        paths.append(path)
        greedy[k] = greedy_energy(env, Q)
    return Q, paths, energy, greedy
