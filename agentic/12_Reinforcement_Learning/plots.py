"""Shared drawing helpers: terrain map, paths, training trails, learning curve, policy arrows."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

BLUE, ORANGE = "#2a78d6", "#eb6834"   # path colors (categorical slots 1, 2)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MOVE_VECTORS = np.array([(-1, 0), (0, 1), (1, 0), (0, -1)])   # (d row, d col) for N, E, S, W


def draw_terrain(ax, env, numbers=True):
    """Heights as a grey ramp (light = low, dark = high) with the height printed in each
    cell (if numbers), walls hatched, start S and goal G labelled."""
    h = np.ma.masked_array(env.h, env.walls)
    im = ax.imshow(h, cmap="Greys", vmin=-1.5, vmax=env.h.max() + 1.5)
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            if env.walls[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="white",
                                       edgecolor=MUTED, hatch="////", lw=0))
                continue
            s = env.index((r, c))
            label = {env.start: "S", env.goal: "G"}.get(s, f"{env.h[r, c]:.0f}" if numbers else "")
            ax.text(c, r, label, ha="center", va="center", fontsize=8,
                    fontweight="bold" if label in "SG" else "normal",
                    color="white" if env.h[r, c] > env.h.max() / 2 else INK)
    ax.set_xticks(np.arange(-0.5, env.n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows), minor=True)
    ax.grid(which="minor", color="white", lw=0.8 if env.n_rows <= 10 else 0.3)
    ax.tick_params(which="both", length=0, labelsize=8, labelcolor=MUTED)
    step = 1 if env.n_rows <= 10 else 5
    ax.set_xticks(range(0, env.n_cols, step))
    ax.set_yticks(range(0, env.n_rows, step))
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def path_xy(env, path, offset=0.0):
    """(x, y) = (column, row) of each state on a path, for plotting on the terrain."""
    rows, cols = np.divmod(np.asarray(path), env.n_cols)
    return np.column_stack([cols, rows]) + offset


def draw_path(ax, env, path, color, label, offset=0.0, lw=2.5):
    """A path of states as a line through cell centers (offset keeps overlapping paths visible)."""
    xy = path_xy(env, path, offset)
    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, label=label,
            solid_capstyle="round", solid_joinstyle="round")


def jittered_paths(env, paths, jitter=0.18, seed=1):
    """Each episode's path with its own small fixed offset, so overlapping routes thicken."""
    rng = np.random.default_rng(seed)
    return [path_xy(env, p, rng.uniform(-jitter, jitter, size=2)) for p in paths]


def episode_colors(n):
    """Light blue for early episodes, dark blue for late ones."""
    return plt.get_cmap("Blues")(0.35 + 0.65 * np.arange(n) / n)


def draw_trails(ax, env, paths, alpha=0.06, lw=0.8):
    """Every episode's path as a faint line, colored by episode (light = early, dark = late)."""
    ax.add_collection(LineCollection(jittered_paths(env, paths), colors=episode_colors(len(paths)),
                                     linewidths=lw, alpha=alpha))


def style_curve_axes(ax, energy, E_opt, E_random, window):
    """Axes, reference lines and labels shared by the static learning curve and the video."""
    ax.set_yscale("log")
    ax.axhline(E_opt, color=INK, lw=1, ls="--")
    ax.text(0, E_opt * 1.04, f"exact optimum {E_opt:.2f}", ha="left", va="bottom",
            fontsize=8, color=INK)
    ax.axhline(E_random, color=MUTED, lw=1, ls=":")
    ax.text(len(energy) * 0.35, E_random * 1.05, "random walk (exact mean)", ha="left",
            va="bottom", fontsize=8, color=MUTED)
    ax.set_xlim(-0.01 * len(energy), 1.01 * len(energy))
    ax.set_ylim(0.8 * E_opt, 1.6 * max(energy.max(), E_random))
    ax.set_xlabel("episode", color=INK)
    ax.set_ylabel("energy", color=INK)
    ax.tick_params(labelsize=8, labelcolor=MUTED)
    ax.grid(color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def rolling_mean(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def draw_learning_curve(ax, energy, greedy, E_opt, E_random, window=50):
    """Energy spent per episode, its rolling average, and the greedy policy's energy."""
    ep = np.arange(len(energy))
    ax.plot(ep, energy, ".", color="#b9b8b2", ms=2, label="energy spent, each episode")
    ax.plot(ep[window - 1:], rolling_mean(energy, window), color=BLUE, lw=2,
            label=f"{window}-episode average")
    ax.plot(ep, np.where(np.isfinite(greedy), greedy, np.nan), color=ORANGE, lw=1.5,
            label="greedy policy's energy (gaps: it loops)")
    style_curve_axes(ax, energy, E_opt, E_random, window)
    ax.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1, 0.6))


def draw_policy(ax, env, Q):
    """The learned move argmax_a Q(s,a) as an arrow in every open cell."""
    cells = [s for s in np.flatnonzero(~env.walls.ravel()) if s not in (env.goal, env.start)]
    rows, cols = np.divmod(np.array(cells), env.n_cols)
    moves = MOVE_VECTORS[np.argmax(Q[cells], axis=1)]
    scale = 16 * env.n_rows / 10
    ax.quiver(cols, rows, moves[:, 1], -moves[:, 0], color=BLUE, pivot="middle",
              scale=scale, width=0.007 * (10 / env.n_rows) ** 0.5,
              headwidth=4, headlength=4, headaxislength=3.5)
