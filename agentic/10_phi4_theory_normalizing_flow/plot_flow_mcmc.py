"""Step 5 figures: how well does the trained flow do on the interacting theory? Shows the good and the bad.

  step_5_sym_training.png        loss + ESS during training, interacting vs free theory
  step_5_sym_noise_to_field.png  one noise sample pushed through the layers, and flow vs Metropolis samples
  step_5_sym_observables.png     M histogram, correlator, action density, and % deviation (raw flow vs flow MCMC)
  step_5_sym_diagnostics.png     importance-weight spread, chain traces, autocorrelation, rejection streaks
  step_5_sym_kernels.png         what each layer's CNN learned: effective 5x5 shift (t) and scale (s) kernels

Run after: python train.py sym  and  python flow_mcmc.py sym
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flow import Flow
from flow_mcmc import estimates
from phi4 import action, correlator, magnetization, tau_int
from train import load_flow

HERE = Path(__file__).parent
OUT = HERE / "figures"
BLUE, ORANGE, INK, MUTED, SURFACE, GRAY = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#8f8d87"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11})
FIELD = dict(cmap="RdBu_r", vmin=-2.5, vmax=2.5)
COLORS = {"Metropolis + flip": INK, "flow raw": ORANGE, "flow MCMC": BLUE}

flow, cfg, hist = load_flow("sym")
_, cfg_free, hist_free = load_flow("free")
L, m2, lam = cfg["L"], cfg["m2"], cfg["lam"]
V = L * L
d = np.load(HERE / "runs" / "sym_samples.npz")
met, raw, idx, accepted, logw = d["met_phi"], d["flow_phi"], d["idx"], d["accepted"], d["logw"]
chain = raw[idx]
sets = {"Metropolis + flip": met, "flow raw": raw, "flow MCMC": chain}
TAG = f"L = {L}, m² = {m2:g}, λ = {lam:g}"


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


def smooth(y, k=5):
    return np.convolve(y, np.ones(k) / k, mode="valid")


# ---------------------------------------------------------------- 1. training
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
def every_100(h, key):
    """History on the regular 100-step grid (newer runs also log every early step)."""
    s = np.array(h["step"])
    keep = (s > 0) & (s % 100 == 0)
    return s[keep], np.array(h[key])[keep]


for h, label, color in ((hist, f"interacting ({TAG})", BLUE), (hist_free, "free (m² = 1, λ = 0)", GRAY)):
    s, e = every_100(h, "ess")
    ax[1].plot(s[4:], smooth(e), color=color, lw=2, label=label)
s, loss = every_100(hist, "loss")
ax[0].plot(s, loss, color=BLUE, lw=1, alpha=0.4)
ax[0].plot(s[4:], smooth(loss), color=BLUE, lw=2, label="interacting (5-point average)")
ax[0].set_title("(a) Training loss ⟨log q + S⟩\nno exact −log Z here: only the trend is informative", loc="left")
ax[1].set_title("(b) ESS on each batch of 64 (5-point average)\n1 = perfect flow", loc="left")
ax[1].set_ylim(0, 1)
for a in ax:
    a.set_xlabel("training step")
    a.legend(frameon=False)
    clean(a)
fig.tight_layout()
fig.savefig(OUT / "step_5_sym_training.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 2. noise -> field, and sample gallery
torch.manual_seed(5)
z = torch.randn(1, L, L)
snapshots = [z[0].numpy().copy()]
with torch.no_grad():
    x = z
    for k, layer in enumerate(flow.layers):
        x, _ = layer(x)
        if k % 2 == 1:
            snapshots.append(x[0].numpy().copy())
fig, ax = plt.subplots(3, 9, figsize=(18, 6.6))
for k, snap in enumerate(snapshots):
    ax[0, k].imshow(snap, **FIELD)
    ax[0, k].set_title("noise z" if k == 0 else f"after layer {2 * k}", fontsize=10)
rng = np.random.default_rng(3)
for k, n in enumerate(rng.choice(len(met), 9, replace=False)):
    ax[1, k].imshow(met[n], **FIELD)
for k in range(9):
    im = ax[2, k].imshow(raw[k], **FIELD)
ax[1, 0].set_ylabel("Metropolis", fontsize=11)
ax[2, 0].set_ylabel("flow", fontsize=11)
ax[0, 0].set_ylabel("one sample\nthrough the flow", fontsize=11)
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.6, label="φ_x")
fig.suptitle(f"Top: noise becomes a field, two layers at a time.   Bottom: spot the difference ({TAG})",
             x=0.01, ha="left", fontsize=13)
fig.savefig(OUT / "step_5_sym_noise_to_field.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- 3. observables
est = {k: estimates(v, m2, lam) for k, v in sets.items()}
fig, ax = plt.subplots(2, 2, figsize=(14, 9.5))

a = ax[0, 0]
for k, phi in sets.items():
    M = magnetization(phi)
    if k == "Metropolis + flip":
        a.hist(M, bins=70, range=(-0.6, 0.6), density=True, color=GRAY, alpha=0.5, label=k)
    else:
        a.hist(M, bins=70, range=(-0.6, 0.6), density=True, histtype="step", lw=2, color=COLORS[k], label=k)
a.set_xlabel("magnetization M")
a.set_yticks([])
a.set_title("(a) Distribution of M", loc="left")
a.legend(frameon=False)

a = ax[0, 1]
r = np.arange(L // 2 + 1)
for k in sets:
    vals = np.array([est[k][f"C({i})"][0] for i in r])
    errs = np.array([est[k][f"C({i})"][1] for i in r])
    a.errorbar(r + {"Metropolis + flip": 0, "flow raw": 0.06, "flow MCMC": 0.12}[k], vals, yerr=errs, fmt="o-",
               color=COLORS[k], ms=6, lw=1.5, label=k)
a.set_yscale("log")
a.set_xlabel("distance r")
a.set_ylabel("C(r) = ⟨φ_x φ_{x+r}⟩")
a.set_title("(b) Two-point correlator", loc="left")
a.legend(frameon=False)

a = ax[1, 0]
for k, phi in sets.items():
    s = action(torch.from_numpy(phi.astype(np.float64)), m2, lam).numpy() / V
    if k == "Metropolis + flip":
        a.hist(s, bins=70, density=True, color=GRAY, alpha=0.5, label=k)
    else:
        a.hist(s, bins=70, density=True, histtype="step", lw=2, color=COLORS[k], label=k)
a.set_xlabel("action per site S/V")
a.set_yticks([])
a.set_title("(c) Distribution of the action", loc="left")
a.legend(frameon=False)

a = ax[1, 1]
names = list(est["Metropolis + flip"])
for j, k in enumerate(("flow raw", "flow MCMC")):
    dev, err = [], []
    for n in names:
        (ref, eref), (v, e) = est["Metropolis + flip"][n], est[k][n]
        dev.append(100 * (v / ref - 1))
        err.append(100 * abs(v / ref) * np.hypot(e / v, eref / ref))
    a.errorbar(dev, np.arange(len(names)) + 0.15 * (j - 0.5), xerr=err, fmt="o", color=COLORS[k], ms=7, label=k)
a.axvline(0, color=MUTED, lw=1)
a.set_yticks(np.arange(len(names)))
a.set_yticklabels(names)
a.invert_yaxis()
a.set_xlabel("deviation from Metropolis + flip  (%)")
a.set_title("(d) Raw flow is biased; the Metropolis-corrected flow is not", loc="left")
a.legend(frameon=False, loc="lower left")
for a in ax.ravel():
    clean(a)
fig.suptitle(f"Interacting theory: flow vs Metropolis ({TAG}, {len(idx)} samples each, "
             f"flow-MCMC acceptance {accepted[1:].mean():.2f})", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "step_5_sym_observables.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 4. diagnostics
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
a = ax[0, 0]
lw = logw - np.log(np.mean(np.exp(logw - logw.max()))) - logw.max()  # log of w / <w>
a.hist(lw, bins=100, color=BLUE, alpha=0.8)
a.axvline(0, color=INK, lw=1.5, ls="--")
a.set_xlabel("log (w / ⟨w⟩),   w = e^{−S} / q")
a.set_yticks([])
a.set_title(f"(a) Importance weights: std(log w) = {logw.std():.2f}  (0 for a perfect flow)\n"
            "left tail = proposals the chain rejects", loc="left")

a = ax[0, 1]
n_show = 300
a.plot(np.abs(magnetization(met[:n_show])), color=GRAY, lw=1.2, label="Metropolis + flip (one sweep per step)")
a.plot(np.abs(magnetization(chain[:n_show])), color=BLUE, lw=1.5, label="flow MCMC (flat = rejected proposal)")
a.set_xlabel("Markov-chain step")
a.set_ylabel("|M|")
a.set_title("(b) Chain traces", loc="left")
a.legend(frameon=False, fontsize=10, loc="upper right")

a = ax[1, 0]
lags = np.arange(40)
for k, phi, color in (("Metropolis + flip", met, GRAY), ("flow MCMC", chain, BLUE)):
    y = np.abs(magnetization(phi))
    y = y - y.mean()
    acf = np.array([np.mean(y[: len(y) - t] * y[t:]) for t in lags]) / np.mean(y * y)
    tau = tau_int(np.abs(magnetization(phi)))
    t_run = float(d["t_flow"] if k == "flow MCMC" else d["t_met"])
    a.plot(lags, acf, "o-", ms=4, color=color,
           label=f"{k}: τ_int = {tau:.2f},  {len(phi) / tau / t_run:.0f} effective samples/s")
a.axhline(0, color=MUTED, lw=1)
a.set_xlabel("lag (chain steps)")
a.set_ylabel("autocorrelation of |M|")
a.set_title("(c) Autocorrelation: how fast each chain forgets", loc="left")
a.legend(frameon=False, fontsize=10)

a = ax[1, 1]
runs, count = [], 0
for acc in accepted[1:]:
    if acc:
        if count:
            runs.append(count)
        count = 0
    else:
        count += 1
runs = np.array(runs)
p_rej = 1 - accepted[1:].mean()
bins = np.arange(1, runs.max() + 2)
a.hist(runs, bins=bins, color=BLUE, alpha=0.8, label="observed")
a.plot(bins[:-1] + 0.5, len(runs) * (1 - p_rej) * p_rej ** (bins[:-1] - 1), color=INK, lw=2,
       label=f"geometric, if rejections were independent (p = {p_rej:.2f})")
a.set_yscale("log")
a.set_ylim(0.5, None)
a.set_xlabel("length of a run of consecutive rejections")
a.set_title(f"(d) Rejection streaks: longest = {runs.max()}\nheavy tail = chain stuck on a rare high-weight sample",
            loc="left")
a.legend(frameon=False, fontsize=10)
for a in ax.ravel():
    clean(a)
fig.suptitle(f"Flow-MCMC diagnostics ({TAG})", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "step_5_sym_diagnostics.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 5. learned kernels, layer by layer
# Effective kernel of layer l: d t_x / d phi_{x+delta} (and the same for s), averaged over the actual fields
# entering that layer, around one active site x, cropped to 5x5. Masked (active) neighbours are exactly 0.
kern_t, kern_s = [], []
with torch.no_grad():
    x = torch.randn(2000, L, L)
inputs = []
with torch.no_grad():
    for layer in flow.layers:
        inputs.append(x.clone())
        x, _ = layer(x)
c = L // 2
for layer, xin in zip(flow.layers, inputs):
    ci, cj = (c, c) if layer.frozen[c, c] == 0 else (c, c + 1)
    xin = xin.clone().requires_grad_(True)
    s_out, t_out = layer.net(layer.frozen * xin)
    for out, store in ((t_out, kern_t), (s_out, kern_s)):
        (g,) = torch.autograd.grad(out[:, ci, cj].sum(), xin, retain_graph=True)
        g = np.roll(np.roll(g.mean(0).numpy(), c - ci, 0), c - cj, 1)
        store.append(g[c - 2:c + 3, c - 2:c + 3])
fig, ax = plt.subplots(4, 8, figsize=(16, 8.6))
for row, (kern, label) in enumerate(((kern_t, "shift t"), (kern_s, "scale s"))):
    vmax = np.max(np.abs(kern))
    for l, k in enumerate(kern):
        a = ax[2 * row + l // 8, l % 8]
        im = a.imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor=INK, lw=1.5))
        a.set_title(f"{label}, layer {l + 1}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax[2 * row:2 * row + 2].ravel().tolist(), shrink=0.8,
                 label=f"d {label.split()[1]}_x / d φ_(x+δ)")
fig.suptitle("What each layer's CNN learned: response of the update at site x (boxed) to each frozen neighbour\n"
             f"Interacting flow ({TAG}). Checkerboard zeros are masked sites.", x=0.01, ha="left", fontsize=13)
fig.savefig(OUT / "step_5_sym_kernels.png", dpi=150, bbox_inches="tight")
plt.close(fig)
# ---------------------------------------------------------------- 6. action: initialization vs final
# The untrained flow is the identity (last conv layer starts at zero), so its samples are the noise itself.
torch.manual_seed(7)
untrained = Flow(L, cfg["n_layers"], cfg["hidden"])
with torch.no_grad():
    phi_init, _ = untrained.sample(len(idx))
    S_init = action(phi_init, m2, lam).numpy() / V
    z = torch.randn(5000, L, L)                      # one batch through the TRAINED flow, layer by layer
    S_layers = [action(z, m2, lam).numpy() / V]
    for layer in flow.layers:
        z, _ = layer(z)
        S_layers.append(action(z, m2, lam).numpy() / V)
S_final = {k: action(torch.from_numpy(v.astype(np.float64)), m2, lam).numpy() / V for k, v in sets.items()}

fig, ax = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.3, 1, 1]})
a = ax[0]
bins = np.geomspace(0.05, S_init.max(), 200)  # log-spaced: both distributions keep their shape
a.hist(S_init, bins=bins, color=GRAY, alpha=0.6, label=f"untrained flow (= noise): ⟨S/V⟩ = {S_init.mean():.2f}")
a.hist(S_final["flow raw"], bins=bins, histtype="step", lw=2, color=ORANGE,
       label=f"trained flow: ⟨S/V⟩ = {S_final['flow raw'].mean():.3f}")
a.hist(S_final["Metropolis + flip"], bins=bins, histtype="step", lw=2, color=INK, ls="--",
       label=f"target (Metropolis + flip): ⟨S/V⟩ = {S_final['Metropolis + flip'].mean():.3f}")
a.set_xscale("log")
a.set_xlabel("action per site S/V  (log scale)")
a.set_yticks([])
a.set_title("(a) Before vs after training: same action, same axis", loc="left")
a.legend(frameon=False, fontsize=10)

a = ax[1]
means = np.array([s.mean() for s in S_layers])
lo, hi = np.array([np.percentile(s, 10) for s in S_layers]), np.array([np.percentile(s, 90) for s in S_layers])
a.fill_between(np.arange(len(means)), lo, hi, color=BLUE, alpha=0.2, label="10–90% of samples")
a.plot(means, "o-", color=BLUE, ms=5, lw=2, label="mean")
a.axhline(S_final["Metropolis + flip"].mean(), color=INK, ls="--", lw=1.5, label="target mean")
a.set_yscale("log")
a.set_xlabel("after layer (0 = noise)")
a.set_ylabel("S/V")
a.set_title("(b) Inside the trained flow: S/V rises, then falls\n(only the output is scored by the loss)", loc="left")
a.legend(frameon=False, fontsize=10)

a = ax[2]
bins = np.linspace(0.05, 0.7, 80)
a.hist(S_final["Metropolis + flip"], bins=bins, density=True, color=GRAY, alpha=0.5, label="target (Metropolis + flip)")
a.hist(S_final["flow raw"], bins=bins, density=True, histtype="step", lw=2, color=ORANGE, label="trained flow, raw")
a.hist(S_final["flow MCMC"], bins=bins, density=True, histtype="step", lw=2, color=BLUE, label="trained flow + MCMC")
a.set_xlabel("action per site S/V")
a.set_yticks([])
a.set_title("(c) Zoom on the final distribution", loc="left")
a.legend(frameon=False, fontsize=10)
for a in ax:
    clean(a)
fig.suptitle(f"Distribution of the action, initialization → trained ({TAG})", x=0.01, ha="left", fontsize=13)
fig.tight_layout()
fig.savefig(OUT / "step_5_sym_action_init_vs_final.png", dpi=150)
plt.close(fig)
print(f"S/V by layer: {np.round(means, 3)}")

print("wrote sym_training, sym_noise_to_field, sym_observables, sym_diagnostics, sym_kernels, sym_action_init_vs_final")
