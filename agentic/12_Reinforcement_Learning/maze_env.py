"""Energy-landscape maze: the environment for the RL energy-maze project.

Natural units: m = g = 1, so climbing a height difference dh costs work dh.

Energy cost of one move from cell s to cell s':
    climbing   (dh > 0):   c_step + dh            full work against gravity
    descending (dh < 0):   c_step - eta * |dh|    a fraction eta of the energy is recovered
    flat, or bumping into a wall / the map edge:  c_step   (basal cost of a step, "friction")
The reward is minus the energy cost. eta = 1 is a conservative force (energy is
path-independent); eta = 0 is pure effort (descending is free, nothing is recovered).

The agent only ever sees integer state labels and what step() returns. The height map,
walls and goal live here and are never handed to the agent.
"""
import numpy as np

# Height map: digits are heights, '#' is an impassable wall. Row 0 is north (top of plots).
TERRAIN = [
    "3344555555",
    "3344511144",
    "3344511144",   # goal (2,4) sits on the rim of a pit (the 1s): the valley trap
    "3334511144",
    "2233411144",
    "9999999949",   # ridge, with one low saddle at (5,8)
    "1111111122",
    "111######1",   # wall
    "0000000001",
    "0000000000",   # start (9,4) on the valley floor
]
START = (9, 4)
GOAL = (2, 4)
C_STEP = 0.25   # energy of one flat step: climbing 1 height unit costs as much as 4 flat steps
ETA = 0.0       # baseline: pure effort

MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # actions 0..3 = N, E, S, W


def parse_terrain(rows):
    walls = np.array([[ch == "#" for ch in row] for row in rows])
    h = np.array([[0 if ch == "#" else int(ch) for ch in row] for row in rows], dtype=float)
    return h, walls


class EnergyMaze:
    def __init__(self, terrain=TERRAIN, start=START, goal=GOAL, c_step=C_STEP, eta=ETA,
                 max_steps=500):
        self.h, self.walls = parse_terrain(terrain)
        self.n_rows, self.n_cols = self.h.shape
        assert not self.walls[start] and not self.walls[goal]
        assert 0.0 <= eta <= 1.0 and c_step >= 0.0
        self.start, self.goal = self.index(start), self.index(goal)
        self.c_step, self.eta, self.max_steps = c_step, eta, max_steps
        self.n_states, self.n_actions = self.n_rows * self.n_cols, len(MOVES)

    def index(self, cell):
        return cell[0] * self.n_cols + cell[1]

    def cell(self, s):
        return divmod(s, self.n_cols)

    def transition(self, s, a):
        """The physics: (s', reward, terminated) for action a in state s. No side effects.
        Used by the environment itself and by the omniscient exact solver, never by the agent."""
        r0, c0 = self.cell(s)
        r1, c1 = r0 + MOVES[a][0], c0 + MOVES[a][1]
        if not (0 <= r1 < self.n_rows and 0 <= c1 < self.n_cols) or self.walls[r1, c1]:
            r1, c1 = r0, c0   # blocked: stay put, still pay for the step
        dh = self.h[r1, c1] - self.h[r0, c0]
        work = dh if dh > 0 else self.eta * dh
        s_next = self.index((r1, c1))
        return s_next, -(self.c_step + work), s_next == self.goal

    # --- agent interface -------------------------------------------------------------
    def reset(self):
        self.s, self.t = self.start, 0
        return self.s

    def step(self, a):
        """Returns (s', reward, terminated, truncated). Truncated = ran out of time, which is
        NOT the same as reaching the goal: the learner must still count the future cost."""
        self.s, reward, terminated = self.transition(self.s, a)
        self.t += 1
        truncated = self.t >= self.max_steps and not terminated
        return self.s, reward, terminated, truncated
