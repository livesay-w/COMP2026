"""Exact solutions that use the full map (omniscient): the answer key for the learner."""
import heapq
from collections import deque

import numpy as np

from maze_env import MOVES


def model_arrays(env):
    """Tabulate env.transition for every (state, action): next state, reward, terminated."""
    S = np.zeros((env.n_states, env.n_actions), dtype=int)
    R = np.zeros((env.n_states, env.n_actions))
    D = np.zeros((env.n_states, env.n_actions), dtype=bool)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            S[s, a], R[s, a], D[s, a] = env.transition(s, a)
    return S, R, D


def value_iteration(env, tol=1e-12, max_sweeps=100_000):
    """Solve the Bellman optimality equation with gamma = 1:
        Q(s,a) = r(s,a) + max_a' Q(s',a'),   with no future cost once s' is the goal.
    Wall cells are not states; their rows are NaN in the result."""
    S, R, D = model_arrays(env)
    wall_rows = env.walls.ravel()
    Q = np.zeros_like(R)
    for _ in range(max_sweeps):
        Q_new = R + np.where(D, 0.0, Q.max(axis=1)[S])
        Q_new[wall_rows] = 0.0
        if np.abs(Q_new - Q).max() < tol:
            Q_new[wall_rows] = np.nan
            return Q_new
        Q = Q_new
    raise RuntimeError(f"value iteration did not converge in {max_sweeps} sweeps")


def state_values(env, Q):
    """V(s) = max_a Q(s,a); V(goal) = 0 (nothing left to pay); NaN on walls."""
    V = Q.max(axis=1)
    V[env.goal] = 0.0
    return V


def greedy_path(env, Q):
    """Follow argmax_a Q(s,a) from the start. Returns (states visited, total energy spent)."""
    path, energy, s = [env.start], 0.0, env.start
    while s != env.goal:
        s, r, _ = env.transition(s, int(np.argmax(Q[s])))
        assert s not in path, "greedy policy revisits a state: Q is not a converged solution"
        path.append(s)
        energy -= r
    return path, energy


def bfs_distances(env, target):
    """Fewest moves from every open cell to `target`, by breadth-first search on the grid.
    Uses only the wall layout, not env.transition, so it is an independent check."""
    dist = np.full(env.n_states, np.inf)
    dist[target] = 0
    queue = deque([target])
    while queue:
        s = queue.popleft()
        r, c = env.cell(s)
        for dr, dc in MOVES:
            r1, c1 = r + dr, c + dc
            if 0 <= r1 < env.n_rows and 0 <= c1 < env.n_cols and not env.walls[r1, c1]:
                s1 = env.index((r1, c1))
                if dist[s1] == np.inf:
                    dist[s1] = dist[s] + 1
                    queue.append(s1)
    return dist


def fewest_moves_actions(env, s, target):
    """Actions for one fewest-moves route from s to target (elevation ignored)."""
    dist = bfs_distances(env, target)
    actions = []
    while s != target:
        a = min(range(env.n_actions), key=lambda a: dist[env.transition(s, a)[0]])
        actions.append(a)
        s = env.transition(s, a)[0]
    return actions


def replay(env, s, actions):
    """Apply a fixed action sequence from s. Returns (states visited, total energy spent)."""
    states, energy = [s], 0.0
    for a in actions:
        s, r, _ = env.transition(s, a)
        states.append(s)
        energy -= r
    return states, energy


def random_walk_exact(env):
    """Uniformly random policy as an absorbing Markov chain on the non-goal cells.
    Returns (P, cells, pos, t, e): transition matrix among `cells`, pos[state] = row of P,
    mean steps to goal t = (I - P)^-1 1, mean energy to goal e = (I - P)^-1 c."""
    S, R, _ = model_arrays(env)
    cells = np.flatnonzero(~env.walls.ravel() & (np.arange(env.n_states) != env.goal))
    pos = np.full(env.n_states, -1)
    pos[cells] = np.arange(len(cells))
    P = np.zeros((len(cells), len(cells)))
    for i, s in enumerate(cells):
        for a in range(env.n_actions):
            if S[s, a] != env.goal:              # moves into the goal leave the chain
                P[i, pos[S[s, a]]] += 1 / env.n_actions
    A = np.eye(len(cells)) - P
    return P, cells, pos, np.linalg.solve(A, np.ones(len(cells))), np.linalg.solve(A, -R[cells].mean(axis=1))


def dijkstra_cost_to_go(env):
    """Least energy from every cell to the goal by Dijkstra on the reversed move graph.
    A different algorithm from value iteration; valid only when every step cost is >= 0."""
    S, R, _ = model_arrays(env)
    cost = -R
    open_rows = ~env.walls.ravel()
    assert (cost[open_rows] >= 0).all(), "Dijkstra needs non-negative step costs (eta = 0)"
    into = [[] for _ in range(env.n_states)]   # into[s'] = [(s, cost of s -> s'), ...]
    for s in np.flatnonzero(open_rows):
        for a in range(env.n_actions):
            if S[s, a] != s:
                into[S[s, a]].append((s, cost[s, a]))
    E = np.full(env.n_states, np.inf)
    E[env.goal] = 0.0
    heap = [(0.0, env.goal)]
    while heap:
        e, s1 = heapq.heappop(heap)
        if e > E[s1]:
            continue
        for s, c in into[s1]:
            if e + c < E[s]:
                E[s] = e + c
                heapq.heappush(heap, (E[s], s))
    return E
