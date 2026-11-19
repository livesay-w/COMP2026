"""Step 4 checks: did training on the free theory (lam = 0, m2 = 1) work? Compares the trained flow
against EXACT results, so no Metropolis is needed:

  - KL(q||p) = <log q + S>_q + log Z  must be >= 0 (variational bound) and small
  - the action itself: for a quadratic action, S ~ Gamma(V/2, 1) exactly under p, so <S> = V/2
  - the propagator <|phi_k|^2>/V = 1/(m2 + khat^2), raw from the flow and after reweighting by w = e^{-S}/q
  - the learned kernel of the last layer, vs the 1/(4+m2) cross the free theory suggests

Run after: python train.py free
Writes figures/step_4_free_training.png
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flow import Flow
from phi4 import action, exact_logZ, exact_propagator, khat2, momentum_power
from train import ess, load_flow

torch.manual_seed(1)  # different seed from training: fresh samples
flow, cfg, history = load_flow("free")
L, m2, lam = cfg["L"], cfg["m2"], cfg["lam"]
V = L * L
logZ = exact_logZ(L, m2)

with torch.no_grad():
    phi, log_q = zip(*(flow.sample(1000) for _ in range(20)))
    phi, log_q = torch.cat(phi), torch.cat(log_q)
    S = action(phi, m2, lam)
logw = (-S - log_q).double()
N = len(S)

# 1. Variational bound: KL >= 0, and small.
kl = (-logw).mean().item() + logZ
kl_err = logw.std().item() / math.sqrt(N)
print(f"[1] KL(q||p) = <log q + S> + log Z = {kl:.4f} +- {kl_err:.4f}   (must be >= 0;  0 = perfect flow)")
assert kl > -4 * kl_err, "KL below zero: log-det or sign bug"
assert kl < 1.0, "flow far from target"

# 2. Effective sample size.
ess_val = ess(logw)
print(f"[2] ESS = {ess_val:.3f}  on {N} samples   (1 = perfect, 1/N = useless)")
assert ess_val > 0.2

# Self-normalised importance weights, for reweighted observables.
w = torch.softmax(logw, 0).numpy()


def reweighted(f):
    """Importance-sampling estimate of <f>_p and its standard error (f: array (N, ...))."""
    mean = np.tensordot(w, f, axes=1)
    err = np.sqrt(np.tensordot(w**2, (f - mean) ** 2, axes=1))
    return mean, err


# 3. The action distribution. Exact: S ~ Gamma(V/2, 1), so <S> = V/2 and var(S) = V/2.
S_np = S.numpy().astype(float)
S_raw, S_raw_err = S_np.mean(), S_np.std() / math.sqrt(N)
S_rw, S_rw_err = reweighted(S_np)
print(f"[3] <S>: exact {V / 2:.3f}   flow raw {S_raw:.3f} +- {S_raw_err:.3f}   "
      f"reweighted {S_rw:.3f} +- {S_rw_err:.3f}   (z = {(S_rw - V / 2) / S_rw_err:+.2f})")
assert abs(S_rw - V / 2) < 4 * S_rw_err

# 4. Propagator at every momentum.
P = momentum_power(phi.numpy().astype(float))
P_raw = P.mean(axis=0)
P_rw, P_rw_err = reweighted(P)
exact_P = exact_propagator(L, m2)
z = (P_rw - exact_P) / P_rw_err
print(f"[4] propagator: max |raw - exact| / exact = {np.max(np.abs(P_raw / exact_P - 1)):.3f}   "
      f"reweighted max |z| over {V} momenta = {np.max(np.abs(z)):.2f}")
assert np.max(np.abs(z)) < 4.5

# 5. What did the last layer's kernels learn? Effective kernel = d t_x / d phi_y, averaged over samples,
#    for one active site x. Each layer's CNN is 3 nonlinear convs, so this is a response, not one raw kernel.
#    The free theory suggests a cross of 1/(4+m2) on the 4 neighbours, but the flow need not land exactly
#    there (many different stacks of layers produce the same q), so this is printed, not asserted.
last = flow.layers[-1]
with torch.no_grad():
    x = torch.randn(2000, L, L)
    for layer in flow.layers[:-1]:
        x, _ = layer(x)
x.requires_grad_(True)
s, t = last.net(last.frozen * x)
ci, cj = [(i, j) for i in range(L) for j in range(L) if last.frozen[i, j] == 0][L + 1]  # an interior active site
(grad,) = torch.autograd.grad(t[:, ci, cj].sum(), x)
resp = (grad.mean(0) / 1).numpy()
resp = np.roll(np.roll(resp, L // 2 - ci, 0), L // 2 - cj, 1)[L // 2 - 2:L // 2 + 3, L // 2 - 2:L // 2 + 3]
print(f"[5] last layer: effective shift kernel d t_x / d phi_(x+delta) around active site x ({ci},{cj}), 5x5:")
print(np.array2string(resp, precision=3, suppress_small=True))
print(f"    free-theory suggestion: {1 / (4 + m2):.3f} on N/E/S/W.   mean s at active sites = "
      f"{s[:, ci, cj].mean().item():+.3f}  (suggestion -1/2 log(4+m2) = {-0.5 * math.log(4 + m2):+.3f})")

print("All step-4 checks passed.")

# ---------------------------------------------------------------- figure
INK, MUTED, SURFACE, BLUE, ORANGE, GRAY = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834", "#8f8d87"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))

# Untrained flow = identity, so its samples are the Gaussian noise itself.
torch.manual_seed(3)
with torch.no_grad():
    phi_init, log_q_init = Flow(L, cfg["n_layers"], cfg["hidden"]).sample(N)
    S_init = action(phi_init, m2, lam).numpy().astype(float)
kl_init = (log_q_init.double() + S_init).mean().item() + logZ

# (a) gap to the bound = batch estimate of KL(q||p), log-log so the big drop and the plateau both show
a = ax[0]
steps = np.array(history["step"])
gap = np.array(history["loss"]) + logZ
keep = (steps >= 1) & (gap > 0)  # a single batch can dip below 0 by noise; log axis cannot show those
a.plot(steps[keep], gap[keep], color=ORANGE, lw=1.5, label="training batch (64 samples)")
a.axhline(kl_init, color=GRAY, ls=":", lw=2, label=f"untrained flow (step 0): {kl_init:.1f}")
a.plot(steps[-1], kl, "o", color=BLUE, ms=9, label=f"final, 20000 fresh samples: {kl:.3f} ± {kl_err:.3f}")
a.set_xscale("log")
a.set_yscale("log")
a.set_xlabel("training step")
a.set_ylabel("loss − (−log Z)  =  KL(q‖p)")
a.set_title(f"(a) Distance to the exact bound\n{kl_init:.0f} → {kl:.2f} in ~100 steps, then a plateau", loc="left")
a.legend(frameon=False, fontsize=10, loc="lower left")

# (b) action distribution: untrained vs trained vs exact Gamma(V/2, 1), on a log axis (counts per bin)
a = ax[1]
bins = np.geomspace(10, S_init.max(), 120)
centers, widths = np.sqrt(bins[1:] * bins[:-1]), np.diff(bins)
gamma_pdf = np.exp((V / 2 - 1) * np.log(centers) - centers - math.lgamma(V / 2))
a.hist(S_init, bins=bins, color=GRAY, alpha=0.6, label=f"untrained flow (= noise): ⟨S⟩ = {S_init.mean():.0f}")
a.hist(S_np, bins=bins, histtype="step", lw=2, color=ORANGE, label=f"trained flow: ⟨S⟩ = {S_raw:.1f}")
a.plot(centers, N * gamma_pdf * widths, color=BLUE, lw=2, ls="--", label=f"exact Gamma(V/2, 1): ⟨S⟩ = {V / 2:.0f}")
a.set_xscale("log")
a.set_xlabel("action S of each configuration  (log scale)")
a.set_yticks([])
a.set_title(f"(b) Distribution of the action, before and after\nreweighted ⟨S⟩ = {S_rw:.2f} ± {S_rw_err:.2f}",
            loc="left")
a.legend(frameon=False, fontsize=10)

# (c) propagator: noise is flat (= 1 at every k); exact is 1/(m2 + khat^2)
a = ax[2]
k2 = khat2(L).ravel()
g = np.linspace(0, 8, 200)
a.axhline(1.0, color=GRAY, ls=":", lw=2, label="untrained flow (= noise): 1 at every k")
a.plot(g, 1 / (m2 + g), color=BLUE, lw=2, label="exact  1/(m² + k̂²)")
a.plot(k2, P_raw.ravel(), "o", ms=6, color=ORANGE, mec=SURFACE, mew=1.5, label="trained flow, raw")
a.errorbar(k2 + 0.12, P_rw.ravel(), yerr=P_rw_err.ravel(), fmt="s", ms=5, color=INK, mec=SURFACE,
           label="trained flow, reweighted")
a.set_xlabel("k̂²")
a.set_ylabel("⟨|φ_k|²⟩ / V")
a.set_title(f"(c) Propagator at all {V} momenta\nraw flow misses the k = 0 mode by "
            f"{100 * (1 - P_raw[0, 0] / exact_P[0, 0]):.0f}%", loc="left")
a.legend(frameon=False, fontsize=10)

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Free theory: untrained → trained flow vs exact  (L = {L}, m² = {m2:g}, λ = 0, ESS = {ess_val:.2f})",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout()
out = Path(__file__).parent / "figures" / "step_4_free_training.png"
fig.savefig(out, dpi=150)
print(f"wrote {out.name}")
