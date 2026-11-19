"""Presentation figures for the setup stage (no training yet). Writes PNGs to figures/.

  1. step_2_coupling_schematic.png    checkerboard mask, the 3x3 kernel window, and the layer alternation
  2. step_1b_field_configurations.png  noise (flow input) vs Metropolis fields across the transition, with M histograms
  3. step_1_free_propagator.png       step-1 check: Metropolis propagator vs exact 1/(m2 + khat^2)

Run: python plot_setup.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from phi4 import binned_mean_err, checkerboard_np, khat2, magnetization, metropolis, momentum_power

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(0)

FROZEN, ACTIVE = "#2a78d6", "#eb6834"   # blue / orange
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
FIELD_CMAP = "RdBu_r"                   # diverging: phi < 0 blue, phi > 0 red, phi = 0 white
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": MUTED, "font.size": 11})


def draw_lattice(ax, L, parity, cell_alpha=1.0):
    """Checkerboard: parity-coloured sites are frozen (blue), the rest active (orange)."""
    frozen = checkerboard_np(L, parity)
    for i in range(L):
        for j in range(L):
            ax.add_patch(Rectangle((j, L - 1 - i), 1, 1, facecolor=FROZEN if frozen[i, j] else ACTIVE,
                                   edgecolor=SURFACE, linewidth=2, alpha=cell_alpha))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------- 1. coupling schematic
L, m2_free = 8, 1.0
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.15]})

# (a) lattice + kernel window centred on an active site
ax = axes[0]
draw_lattice(ax, L, parity=0)
ci, cj = 3, 4  # (3 + 4) odd -> active for parity 0
ax.add_patch(Rectangle((cj - 1, L - 1 - ci - 1), 3, 3, fill=False, edgecolor=INK, linewidth=3))
ax.text(cj + 0.5, L - 1 - ci + 0.5, "x", ha="center", va="center", color="white", fontsize=16, weight="bold")
ax.set_title("(a) Checkerboard mask + 3×3 kernel window", loc="left")
ax.legend(handles=[Rectangle((0, 0), 1, 1, color=FROZEN), Rectangle((0, 0), 1, 1, color=ACTIVE)],
          labels=["frozen: input to the CNN", "active: gets updated"], loc="upper center",
          bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)

# (b) zoom on the window: what the kernel actually sees, and the ideal free-theory shift kernel
ax = axes[1]
w = 1 / (4 + m2_free)
for a in range(3):
    for b in range(3):
        frozen_site = (a + b) % 2 == 1  # around an active centre, N/E/S/W are frozen
        ax.add_patch(Rectangle((b, 2 - a), 1, 1, facecolor=FROZEN if frozen_site else "#e4e3df",
                               edgecolor=SURFACE, linewidth=3))
        weight = f"{w:.2f}" if frozen_site else "0"
        label = "φ = 0\n(masked)" if not frozen_site else f"w = {weight}"
        if (a, b) == (1, 1):
            label = "x\nφ = 0\n(masked)"
        ax.text(b + 0.5, 2 - a + 0.5, label, ha="center", va="center", fontsize=10,
                color="white" if frozen_site else MUTED)
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(b) Kernel over the window", loc="left")
ax.text(1.5, -0.25, r"$t_x = \sum_\delta\, w_\delta\, \varphi_{x+\delta}$" + "   (weights learned from the action)\n"
        f"Free theory (m² = {m2_free:g}) predicts the cross: " + r"$w = 1/(4+m^2)$" + f" = {w:.2f}",
        ha="center", va="top", fontsize=10, color=MUTED)

# (c) layers alternate parity so every site gets updated
ax = axes[2]
ax.axis("off")
ax.set_title("(c) Layers alternate which colour is frozen", loc="left")
for k, (parity, name) in enumerate([(0, "layer 1"), (1, "layer 2"), (0, "layer 3"), (1, "layer 4")]):
    sub = ax.inset_axes([0.02 + 0.25 * k, 0.42, 0.2, 0.36])
    draw_lattice(sub, 4, parity)
    sub.set_title(name, fontsize=10, color=MUTED)
    if k < 3:
        ax.annotate("", xy=(0.27 + 0.25 * k, 0.6), xytext=(0.22 + 0.25 * k, 0.6), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.5})
ax.text(0.5, 0.3, r"active sites:  $\varphi_x \leftarrow e^{s_x}\varphi_x + t_x$,    $(s, t) = \mathrm{CNN}(\mathrm{frozen\ sites})$" + "\n\n"
        r"$\log|\det J| = \sum_{x\,\in\,\mathrm{active}} s_x$" + "   (exact and cheap)\n\n"
        r"noise $z$  →  16 layers  →  field $\varphi \sim q(\varphi) \approx e^{-S}/Z$",
        ha="center", va="top", fontsize=10.5, transform=ax.transAxes)

fig.tight_layout()
fig.savefig(OUT / "step_2_coupling_schematic.png", dpi=150)
plt.close(fig)
print("wrote step_2_coupling_schematic.png")


# ---------------------------------------------------------------- 2. field configurations across the transition
L, lam = 32, 1.0
phases = [(-2.0, "symmetric"), (-3.7, "near critical"), (-4.5, "broken")]
fig, axes = plt.subplots(2, 4, figsize=(16, 8), gridspec_kw={"height_ratios": [1.15, 1]})
vmax = 2.5

z = rng.normal(size=(L, L))
axes[0, 0].imshow(z, cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax)
axes[0, 0].set_title("Noise z ~ N(0,1)\n(flow input)")
axes[1, 0].hist(z.ravel(), bins=40, color=MUTED, density=True)
axes[1, 0].set_title("site values of z", fontsize=10, color=MUTED)
axes[1, 0].set_xlabel("z_x")

for k, (m2, name) in enumerate(phases, start=1):
    configs, acc = metropolis(L, m2, lam, n_sweeps=10000, n_therm=2000, step=1.5, rng=rng, flip=False)
    configs_flip, _ = metropolis(L, m2, lam, n_sweeps=10000, n_therm=2000, step=1.5, rng=rng, flip=True)
    im = axes[0, k].imshow(configs[-1], cmap=FIELD_CMAP, vmin=-vmax, vmax=vmax)
    axes[0, k].set_title(f"{name}\nm² = {m2:g}, λ = {lam:g}")
    # The true P(M) is symmetric under M -> -M. Local updates alone can get stuck in one well;
    # adding the global flip phi -> -phi (dS = 0) restores the missing well.
    M, M_flip = magnetization(configs), magnetization(configs_flip)
    axes[1, k].hist(M, bins=60, range=(-1.2, 1.2), color=ACTIVE, density=True, label="local updates only")
    axes[1, k].hist(M_flip, bins=60, range=(-1.2, 1.2), density=True, histtype="step", color=FROZEN, lw=2,
                    linestyle="--", label="local updates + global flip φ → −φ")
    # Fraction of sweeps with M > 0 must be 0.5 for a correctly mixed chain.
    axes[1, k].set_title(f"fraction of sweeps with M > 0 (exact 0.50)\n"
                         f"local only {np.mean(M > 0):.2f}  |  with flip {np.mean(M_flip > 0):.2f}",
                         fontsize=10, color=INK)
    axes[1, k].set_xlabel("M")
    print(f"  m2 = {m2:5.2f}: acceptance {acc:.2f}, <M> local = {M.mean():+.3f}, <M> flip = {M_flip.mean():+.3f}, "
          f"<|M|> local = {np.abs(M).mean():.3f}, flip = {np.abs(M_flip).mean():.3f}")
axes[1, 2].legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

for ax in axes[0]:
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes[1]:
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="φ_x")
fig.suptitle(f"What the flow must learn: noise in, φ⁴ fields out  (local Metropolis, L = {L})", x=0.02,
             ha="left", fontsize=13)
fig.savefig(OUT / "step_1b_field_configurations.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote step_1b_field_configurations.png")


# ---------------------------------------------------------------- 3. free propagator check
L, m2_free = 8, 1.0
configs, _ = metropolis(L, m2_free, 0.0, n_sweeps=20000, n_therm=1000, step=1.0, rng=rng, flip=False)
mean, err = binned_mean_err(momentum_power(configs))
k2 = khat2(L).ravel()
grid = np.linspace(0, 8, 200)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(grid, 1 / (m2_free + grid), color=FROZEN, lw=2, label="exact  1/(m² + k̂²)")
ax.errorbar(k2, mean.ravel(), yerr=err.ravel(), fmt="o", ms=6, color=ACTIVE, mec=SURFACE, mew=1.5,
            ecolor=ACTIVE, label="Metropolis, all 64 momenta")
ax.set_xlabel("lattice momentum  k̂² = Σ_μ 4 sin²(k_μ/2)")
ax.set_ylabel("⟨|φ_k|²⟩ / V")
ax.set_title(f"Free-theory check (λ = 0, m² = {m2_free:g}, L = {L})", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "step_1_free_propagator.png", dpi=150)
plt.close(fig)
print("wrote step_1_free_propagator.png")
