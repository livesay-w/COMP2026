"""Step 7d checks: the embedded-Ising Swendsen-Wang update (cluster.py).

Run: python check_cluster.py   (about 2 min; check [4] needs runs/scan_fine.npz from step 6b)
"""
from pathlib import Path

import numpy as np

from cluster import bonds, cluster_chain, clusters, m_and_phi2, sw_step
from phi4 import binned_mean_err, exact_propagator, momentum_power, tau_int
from scan import binder_chi

RUNS = Path(__file__).parent / "runs"
rng = np.random.default_rng(0)

# 1. Structure of one SW step, on a random field: |phi| is untouched, every cluster is flipped as a whole
#    (phi_new / phi_old is +1 or -1 on all of it), and bonds only join sites of equal sign inside one cluster.
L = 8
phi = rng.normal(size=(L, L))
new, labels = sw_step(phi, rng)
ratio = new / phi
assert np.array_equal(np.abs(new), np.abs(phi))
assert all(np.unique(ratio[labels == c]).size == 1 for c in np.unique(labels))
bond = bonds(phi, rng)
lab, n = clusters(bond)
for ax, b in enumerate(bond):
    nbr_phi, nbr_lab = np.roll(phi, -1, axis=ax), np.roll(lab, -1, axis=ax)
    assert np.all(phi[b] * nbr_phi[b] > 0) and np.array_equal(lab[b], nbr_lab[b])
print(f"[1] L={L}: |phi| unchanged, {len(np.unique(labels))} clusters each flipped as a unit, "
      f"{sum(b.sum() for b in bond)} bonds all between equal signs within one cluster")


# 2. SW alone at frozen |phi| must sample the embedded Ising model p(s) ∝ exp(sum_<xy> |phi_x phi_y| s_x s_y) exactly.
#    On 4x4 there are 2^16 sign configurations, so <s_0 s_x> (all 16 x) and <M^2> of phi are computed exactly by
#    enumeration and compared with a long SW-only chain. Control: the same chain on |phi| / sqrt(2) samples couplings J / 2
#    (as if the bond probability were 1 - exp(-J)); it must fail the same comparison, so the test has teeth.
L = 4
a = rng.uniform(0.4, 1.0, size=(L, L))  # frozen |phi|: couplings J = a_x a_y in [0.16, 1]
s_all = 1 - 2 * ((np.arange(2**16)[:, None] >> np.arange(16)) & 1).reshape(-1, L, L)  # (65536, 4, 4) of +-1
E = sum(np.sum(a * np.roll(a, -1, axis=ax) * s_all * np.roll(s_all, -1, axis=ax + 1), axis=(1, 2)) for ax in (0, 1))
w = np.exp(E - E.max())
w /= w.sum()
corr_exact = np.einsum("n,nij->ij", w, s_all[:, :1, :1] * s_all)  # <s_0 s_x>
M2_exact = w @ np.mean(a * s_all, axis=(1, 2)) ** 2


def sw_only(scale, n_steps):
    """SW-only chain on phi = scale * a * s. Returns per-step s_0 s_x (n, 4, 4) and M^2 of phi = a * s (n,)."""
    phi = scale * a * np.where(rng.random((L, L)) < 0.5, -1.0, 1.0)
    corr, M2 = np.empty((n_steps, L, L)), np.empty(n_steps)
    for t in range(n_steps):
        phi, _ = sw_step(phi, rng)
        s = np.sign(phi)
        corr[t], M2[t] = s[0, 0] * s, np.mean(a * s) ** 2
    return corr, M2


for scale, label in ((1.0, "SW"), (2**-0.5, "control, J/2")):
    corr, M2 = sw_only(scale, 100_000)
    (c, dc), (m, dm) = binned_mean_err(corr), binned_mean_err(M2)
    z_corr = ((c - corr_exact) / np.where(dc > 0, dc, np.inf))  # dc = 0 only at x = 0, where s_0 s_0 = 1 exactly
    z_M2 = (m - M2_exact) / dm
    zmax = max(np.max(np.abs(z_corr)), abs(z_M2))
    print(f"[2] {label:13s} <M^2> = {m:.5f} ± {dm:.5f} (exact {M2_exact:.5f}, z = {z_M2:+.1f});   "
          f"max |z| over the 15 <s_0 s_x> = {np.max(np.abs(z_corr)):.1f}")
    assert zmax < 4 if label == "SW" else zmax > 10


# 3. Full update (Metropolis + SW) in the free theory near criticality, lam = 0, m2 = 0.1 (correlation length ~3 on L = 8):
#    <|phi_k|^2>/V = 1/(m2 + khat^2) at every k. SW is exact for any even potential, including lam = 0.
L, m2_free = 8, 0.1
P, _, acc = cluster_chain(L, m2_free, 0.0, n_updates=40_000, n_therm=1000, step=1.0, rng=rng,
                          measure=lambda phi: momentum_power(phi[None])[0])
mean, err = binned_mean_err(P)
z = (mean - exact_propagator(L, m2_free)) / err
print(f"[3] free L={L}, m2={m2_free}: acceptance {acc:.3f};  k = 0: {mean[0, 0]:.3f} ± {err[0, 0]:.3f} "
      f"(exact {exact_propagator(L, m2_free)[0, 0]:.3f});  max |z| over all {L * L} momenta = {np.max(np.abs(z)):.2f}")
assert np.max(np.abs(z)) < 4.5


# 4. Interacting theory at the 6b crossing, L = 8, m2 = -3.80 (6b grid index 6): U4 and chi agree with the 6b Metropolis
#    chain. tau_int(M^2) is reported, not asserted: at L = 8 the gain over Metropolis is only ~3.6x (the cluster step
#    cannot move |phi|, which only the local sweep updates). What matters is how tau grows with L (pilot in cluster_scan.py).
scan = np.load(RUNS / "scan_fine.npz")
L, i = 8, 6
m2 = float(scan["m2"][i])
rec, _, acc = cluster_chain(L, m2, 1.0, n_updates=60_000, n_therm=1000, step=1.5, rng=rng, measure=m_and_phi2)
M_met, M_cl = scan[f"M_{L}"][i], rec[:, 0]
tau_cl, tau_met = tau_int(M_cl**2), tau_int(M_met.astype(float) ** 2)
print(f"[4] L={L}, m2={m2}: acceptance {acc:.3f};  tau_int(M^2) = {tau_cl:.2f} updates (cluster) vs {tau_met:.1f} sweeps (Metropolis)")
for name, (cv, ce), (mv, me) in zip(("U4", "chi"), binder_chi(M_cl, L), binder_chi(M_met, L)):
    zz = (cv - mv) / np.hypot(ce, me)
    print(f"    {name:4s} cluster {cv:8.4f} ± {ce:.4f}   Metropolis {mv:8.4f} ± {me:.4f}   z = {zz:+.1f}")
    assert abs(zz) < 3


# 5. Checkpointing (cluster_scan.py): a chain saved to disk and reloaded after every 70 updates, crossing the end of
#    thermalisation, must equal one uninterrupted cluster_chain run with the same seed, bit for bit (records, final field,
#    acceptance count). A resume that dropped the RNG state, the field or the update counter would break this.
import tempfile
from cluster_scan import advance, load_state, new_state, save_state

L, m2, lam, n_upd, n_th, step, seed = 8, -3.8, 1.0, 300, 40, 1.5, 7
ref, ref_phi, ref_acc = cluster_chain(L, m2, lam, n_upd, n_th, step, np.random.default_rng([seed, L, round(-m2 * 1e5)]),
                                      m_and_phi2)
params = (m2, lam, n_upd, n_th, step, seed)
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "chain.ckpt.npz"
    state, n_reloads = new_state(L, m2, n_upd, seed), 0
    while state["t"] < n_th + n_upd:
        advance(state, L, m2, lam, n_upd, n_th, step, 70)
        save_state(path, state, params)
        state, n_reloads = load_state(path, params, n_upd), n_reloads + 1
acc = state["n_accept"] / (n_upd * L * L)
print(f"[5] checkpointed chain ({n_reloads} save/reload cycles) vs uninterrupted: records identical "
      f"{np.array_equal(state['rec'], ref)}, final field identical {np.array_equal(state['phi'], ref_phi)}, "
      f"acceptance {acc:.6f} vs {ref_acc:.6f}")
assert np.array_equal(state["rec"], ref) and np.array_equal(state["phi"], ref_phi) and acc == ref_acc

print("All step-7d cluster checks passed.")
