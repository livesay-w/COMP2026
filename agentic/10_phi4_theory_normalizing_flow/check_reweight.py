"""Step 7e checks: multi-histogram reweighting in m2 (reweight.py).

Run: python check_reweight.py   (about 3 min; [2]-[4] need the cluster_scan.py "prod" chains)
"""
import numpy as np

from cluster import cluster_chain, m_and_phi2
from cluster_scan import load
from phi4 import khat2, tau_int
from quotient import THIN
from reweight import Reweighter, u4_chi
from scan import binder_chi

rng = np.random.default_rng(0)

# 1. Free theory (lam = 0), where every m2 has an exact answer: chi = V<M^2> = 1/m2 (the k = 0 propagator) and
#    <E> = <1/2 sum phi^2> = 1/2 sum_k 1/(m2 + khat^2). Chains at m2 = 0.3, 0.4, 0.5 on L = 8, reweighted to 9 values
#    between and on them, must hit both. Control: the same data with E doubled (forgetting the 1/2 in S = S_0 + m2 * 1/2 sum phi^2)
#    must fail, so the check is sensitive to how E enters.
L, grid = 8, np.array([0.3, 0.4, 0.5])
chains = [cluster_chain(L, m2, 0.0, n_updates=40_000, n_therm=1000, step=1.0, rng=rng, measure=m_and_phi2)[0] for m2 in grid]
M = np.concatenate([c[:, 0] for c in chains])
E = [0.5 * c[:, 1] for c in chains]
targets = np.linspace(0.3, 0.5, 9)
for scale, label in ((1.0, "E"), (2.0, "control, 2E")):
    rw = Reweighter([scale * e for e in E], grid)
    # Errors: 20-block jackknife of the whole reweighting (refit per replica).
    blocks = [np.array_split(np.arange(len(c)), 20) for c in chains]
    est = []
    for r in range(21):
        keep = [np.concatenate([b for j, b in enumerate(bl) if j != r - 1]) for bl in blocks]
        rw_r = Reweighter([scale * E[k][keep[k]] for k in range(3)], grid, rw.f)
        M_r = np.concatenate([chains[k][keep[k], 0] for k in range(3)])
        E_r = np.concatenate([E[k][keep[k]] for k in range(3)])  # true E as the observable, whatever the weights used
        est.append([[L * L * rw_r.mean(np.stack([M_r**2, E_r], axis=1), t)[0], rw_r.mean(E_r[:, None], t)[0]] for t in targets])
    est = np.array(est)  # (21, n_targets, 2)
    err = np.sqrt(19 * np.mean((est[1:] - est[1:].mean(axis=0)) ** 2, axis=0))
    exact = np.stack([1 / targets, [0.5 * np.sum(1 / (t + khat2(L))) for t in targets]], axis=1)
    z = (est[0] - exact) / err
    print(f"[1] free theory, {label:11s}: max |z| over 9 m2 values: chi {np.max(np.abs(z[:, 0])):.1f}, <E> {np.max(np.abs(z[:, 1])):.1f}"
          f"   (m2 = 0.35: chi {est[0, 1, 0]:.3f} ± {err[1, 0]:.3f} vs exact {exact[1, 0]:.3f})")
    if label == "E":
        assert np.max(np.abs(z)) < 3.5
    else:
        assert np.max(np.abs(z)) > 10

# 2. Production chains, leave one chain out: drop an interior chain, reweight the others to its m2, and compare with that
#    chain's own direct estimate. Tests that neighbouring histograms overlap enough for reweighting across a full grid step.
#    The prediction's error comes from a 10-block jackknife of the reweighting; the direct one from scan.binder_chi.
cl = load("prod")
zs = []
for L, (g, Mall, phi2) in cl.items():
    for i in range(1, len(g) - 1):
        others = [k for k in range(len(g)) if k != i]
        Mo, Eo = [Mall[k, ::THIN] for k in others], [0.5 * phi2[k, ::THIN] for k in others]
        blocks = [np.array_split(np.arange(len(m)), 10) for m in Mo]
        pred = []
        for r in range(11):
            keep = [np.concatenate([b for j, b in enumerate(bl) if j != r - 1]) for bl in blocks]
            rw = Reweighter([e[kp] for e, kp in zip(Eo, keep)], g[others], None if r == 0 else f0)
            f0 = rw.f if r == 0 else f0
            pred.append(u4_chi(rw, np.concatenate([m[kp] for m, kp in zip(Mo, keep)]), L, g[i]))
        pred = np.array(pred)
        perr = np.sqrt(9 * np.mean((pred[1:] - pred[1:].mean(axis=0)) ** 2, axis=0))
        (du, due), (dc, dce) = binder_chi(Mall[i], L)
        zs.append(((pred[0, 0] - du) / np.hypot(perr[0], due), (pred[0, 1] - dc) / np.hypot(perr[1], dce)))
    zL = np.array(zs[-(len(g) - 2):])
    print(f"[2] L={L:3d}: leave-one-chain-out at {len(g) - 2} interior m2: U4 z = {np.round(zL[:, 0], 1)}   chi z = {np.round(zL[:, 1], 1)}")
zs = np.array(zs)
print(f"    all {len(zs)} points: max |z| = {np.max(np.abs(zs)):.2f}, sum z^2 = {np.sum(zs**2):.1f} for {zs.size} comparisons")
assert np.max(np.abs(zs)) < 3.5

# 3. Overlap: neighbouring chains' E histograms. Reported, with the gap between their means in units of the E spread.
for L, (g, _, phi2) in cl.items():
    E_ = 0.5 * phi2
    print(f"[3] L={L:3d}: (mean gap between neighbouring chains) / std(E) = {np.max(np.abs(np.diff(E_.mean(axis=1))) / E_.std(axis=1)[:-1]):.2f} at most")

# 4. Thinning by THIN costs little, because tau_int(M^2) ~ 20 >> THIN: the jackknife errors of U4 and chi from the thinned
#    series are on average < 3% larger than from the full series. Asserted on the mean over all chains: each single ratio
#    scatters by ~1.6% (noise in the error estimates themselves; ratios fall below 1 too), so a per-chain 5% bound failed
#    by chance on the first run (max 1.052 of 60) while the mean loss is 0.6% ± 0.2%.
ratio = []
for L, (g, Mall, _) in cl.items():
    for j in range(len(g)):
        full, thin = binder_chi(Mall[j], L), binder_chi(Mall[j, ::THIN], L)
        ratio += [thin[0][1] / full[0][1], thin[1][1] / full[1][1]]
ratio = np.array(ratio)
print(f"[4] THIN = {THIN}: error(thinned) / error(full) for U4 and chi at all {len(ratio) // 2} chains: mean "
      f"{ratio.mean():.4f} ± {ratio.std(ddof=1) / np.sqrt(len(ratio)):.4f}, range {ratio.min():.3f} to {ratio.max():.3f};  "
      f"tau_int(M^2) of the thinned L = 128 chains "
      f"{min(tau_int(m[::THIN] ** 2) for m in cl[128][1]):.1f} to {max(tau_int(m[::THIN] ** 2) for m in cl[128][1]):.1f}")
assert ratio.mean() < 1.03

print("All step-7e reweighting checks passed.")
