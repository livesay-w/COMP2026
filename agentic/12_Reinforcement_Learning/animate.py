"""Training animation: one frame per shown episode, drawn on the board in bold, with earlier
attempts left behind as faint lines, next to the learning curve so far. Saved as a GIF
(Pillow; no ffmpeg needed)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from qlearning import epsilon
from plots import (draw_terrain, jittered_paths, episode_colors, style_curve_axes, rolling_mean,
                   BLUE, ORANGE, INK)


def learning_gif(env, paths, energy, greedy, E_opt, E_random, frames, out, trail="all",
                 window=50, fps=12, numbers=True, alpha=0.06):
    """trail = "all": every earlier episode stays as a faint line;
    trail = "shown": only the earlier episodes that got their own frame (cheaper to draw)."""
    n = len(paths)
    xy = jittered_paths(env, paths)
    colors = episode_colors(n)
    rolling = rolling_mean(energy, window)
    ep = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1, 1.3]})
    draw_terrain(ax1, env, numbers=numbers)
    trails = LineCollection([], linewidths=0.8, alpha=alpha)
    ax1.add_collection(trails)
    current, = ax1.plot([], [], color=ORANGE, lw=2 if env.n_rows <= 10 else 1.5,
                        solid_joinstyle="round")
    head, = ax1.plot([], [], "o", color=ORANGE, ms=7, mec="white", mew=1.5)
    title = ax1.set_title("episode", fontsize=9, color=INK)   # placeholder: tight_layout room

    dots, = ax2.plot([], [], ".", color="#b9b8b2", ms=2.5, label="energy spent, each episode")
    avg, = ax2.plot([], [], color=BLUE, lw=2, label=f"{window}-episode average")
    greedy_line, = ax2.plot([], [], color=ORANGE, lw=1.5, label="greedy policy's energy")
    marker, = ax2.plot([], [], "o", color=ORANGE, ms=6, mec="white", mew=1.5)
    style_curve_axes(ax2, energy, E_opt, E_random, window)
    ax2.legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1, 0.6))
    fig.tight_layout()

    shown = sorted(set(frames))

    def update(k):
        earlier = list(range(k)) if trail == "all" else [j for j in shown if j < k]
        trails.set_segments([xy[j] for j in earlier])
        trails.set_color(colors[earlier])
        current.set_data(*xy[k].T)
        head.set_data([xy[k][-1, 0]], [xy[k][-1, 1]])
        title.set_text(f"episode {k + 1} / {n}    eps = {epsilon(k, n):.2f}    "
                       f"{len(paths[k]) - 1} steps    energy {energy[k]:.2f}")
        dots.set_data(ep[:k + 1], energy[:k + 1])
        avg.set_data(ep[window - 1:k + 1], rolling[:max(0, k + 2 - window)])
        greedy_line.set_data(ep[:k + 1], np.where(np.isfinite(greedy[:k + 1]), greedy[:k + 1], np.nan))
        marker.set_data([k], [energy[k]])
        return trails, current, head, title, dots, avg, greedy_line, marker

    FuncAnimation(fig, update, frames=frames).save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"saved {out} ({len(frames)} frames, {len(frames) / fps:.0f} s)")
