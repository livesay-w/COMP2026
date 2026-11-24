"""Seeded generator for big energy mazes, in the same text format as maze_env.TERRAIN
(digits = heights, '#' = wall), so everything downstream is reused unchanged.

Walls: a randomized depth-first-search maze on the even-coordinate cells (every cell reachable,
many dead ends, exactly one route between any two cells), then a fraction of the remaining
walls between cells is knocked out. That creates loops, i.e. competing routes, which is what
gives the energy a choice to make.
Heights: a sum of smooth Gaussian hills, rescaled and rounded to integers 0..9.
"""
import numpy as np


def maze_walls(n, rng, room, knockout):
    """Square rooms of side `room` separated by 1-thick walls, connected as a maze: each opened
    wall gets a one-cell doorway at a random position. room = 1 is the classic corridor maze."""
    p = room + 1                          # period: one room plus one wall
    m = (n + 1) // p                      # rooms per side
    walls = np.ones((n, n), dtype=bool)
    for i in range(m):
        for j in range(m):
            walls[i * p:i * p + room, j * p:j * p + room] = False

    def open_door(a, b):
        (i, j), (i2, j2) = sorted([a, b])
        k = rng.integers(room)
        if i2 != i:
            walls[i2 * p - 1, j * p + k] = False      # wall row between vertical neighbours
        else:
            walls[i * p + k, j2 * p - 1] = False      # wall column between horizontal neighbours

    visited = np.zeros((m, m), dtype=bool)
    visited[0, 0] = True
    stack, opened = [(0, 0)], set()
    while stack:                          # depth-first search with random branch order
        i, j = stack[-1]
        fresh = [(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))
                 if 0 <= i + di < m and 0 <= j + dj < m and not visited[i + di, j + dj]]
        if not fresh:
            stack.pop()
            continue
        nxt = fresh[rng.integers(len(fresh))]
        open_door((i, j), nxt)
        opened.add(frozenset([(i, j), nxt]))
        visited[nxt] = True
        stack.append(nxt)
    closed = [(a, b) for a in np.ndindex(m, m) for b in [(a[0] + 1, a[1]), (a[0], a[1] + 1)]
              if b[0] < m and b[1] < m and frozenset([a, b]) not in opened]
    for k in rng.choice(len(closed), int(knockout * len(closed)), replace=False):
        open_door(*closed[k])             # extra doorways create loops
    return walls


def gaussian_hills(n, rng, n_hills, widths, tilt):
    """Sum of Gaussian hills plus a gentle tilt rising toward the top-right (valley -> plateau)."""
    rows, cols = np.mgrid[0:n, 0:n]
    h = tilt * (cols - rows) / n
    for _ in range(n_hills):
        r0, c0 = rng.uniform(0, n, size=2)
        w = rng.uniform(*widths)
        h += rng.uniform(0.5, 1.0) * np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * w * w))
    return np.rint(9 * (h - h.min()) / (h.max() - h.min())).astype(int)


def make_maze(n=30, seed=0, room=4, knockout=0.2, n_hills=25, widths=(1.5, 3.5), tilt=1.0):
    """Returns (terrain rows, start, goal). Start = lowest open cell in the bottom-left third,
    goal = highest open cell in the top-right third (ties: closest to the corner)."""
    rng = np.random.default_rng(seed)
    walls = maze_walls(n, rng, room, knockout)
    h = gaussian_hills(n, rng, n_hills, widths, tilt)
    third = n // 3
    open_cells = [(r, c) for r in range(n) for c in range(n) if not walls[r, c]]
    start = min((cell for cell in open_cells if cell[0] >= n - third and cell[1] < third),
                key=lambda rc: (h[rc], (n - 1 - rc[0]) + rc[1]))
    goal = max((cell for cell in open_cells if cell[0] < third and cell[1] >= n - third),
               key=lambda rc: (h[rc], -(rc[0] + (n - 1 - rc[1]))))
    terrain = ["".join("#" if walls[r, c] else str(h[r, c]) for c in range(n)) for r in range(n)]
    return terrain, start, goal
