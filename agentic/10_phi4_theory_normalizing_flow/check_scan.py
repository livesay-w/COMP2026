"""Step 6 checks: the memory-light sampler, the Binder limits, and the coarse (6a) and fine (6b) scans.

Run after: python scan.py coarse      ->  python check_scan.py coarse   (checks [1]-[5])
           python scan.py fine        ->  python check_scan.py fine     (checks [1]-[4] on the fine scan, then [6]-[8])
"""
import sys

import numpy as np

from phi4 import magnetization, metropolis, metropolis_magnetization, tau_int
from scan import U4_ISING, binder_chi, crossing, doubling_pairs, load_scan, pair_crossing

N_BLOCKS = 50  # jackknife blocks used by binder_chi (phi4.jackknife default)
name = sys.argv[1]

# [1] metropolis_magnetization is the SAME chain as metropolis, not merely a similar one.
#     Same seed -> identical M series and identical final configuration, to the last bit.
L, m2, lam = 8, -4.0, 1.0
configs, acc_a = metropolis(L, m2, lam, 2000, 200, 1.5, np.random.default_rng(1), flip=True)
M, last, acc_b = metropolis_magnetization(L, m2, lam, 2000, 200, 1.5, np.random.default_rng(1), flip=True)
assert np.array_equal(M, magnetization(configs)) and np.array_equal(last, configs[-1]) and acc_a == acc_b
print("[1] metropolis_magnetization reproduces metropolis bit for bit")

# [2] Binder limits, far from the transition where the answer is known without any simulation.
#     Symmetric phase (m2 = +1): M is a sum of many weakly correlated sites -> Gaussian -> <M^4> = 3<M^2>^2 -> U4 = 0.
#     Deep broken phase (m2 = -8, wells at phi^2 = -m2/(4 lam) = 2): M sits at +-|M0| -> <M^4> = <M^2>^2 -> U4 = 2/3.
for m2, target, tol in [(+1.0, 0.0, 0.05), (-8.0, 2 / 3, 0.01)]:
    M, _, _ = metropolis_magnetization(8, m2, 1.0, 40000, 2000, 1.5, np.random.default_rng(2), flip=True)
    (U4, dU4), _ = binder_chi(M, 8)
    print(f"[2] m2={m2:+.0f}: U4 = {U4:.4f} +- {dU4:.4f}   (expected {target:.4f}, tol {tol})")
    assert abs(U4 - target) < tol

# ---- the scan itself
d, U4, dU4, chi, dchi = load_scan(name)
Ls, m2 = d["Ls"], d["m2"]

# [3] Error bars are honest: every jackknife block must be >= 10 tau_int(M^2) long, or blocks are correlated
#     and the errors on U4 and chi are underestimated. Checked per L, since sweeps (and tau) grow with L.
for L in Ls:
    M_L = d[f"M_{L}"]
    taus = np.array([tau_int(M_L[i].astype(np.float64) ** 2) for i in range(len(m2))])
    block = M_L.shape[1] // N_BLOCKS
    print(f"[3] L={L:2d}: block length {block:6d} sweeps; max tau_int(M^2) = {taus.max():6.1f} "
          f"at m2={m2[taus.argmax()]:+.3f}  -> {block / taus.max():.1f} tau per block")
    assert block >= 10 * taus.max()

# [4] U4 grows monotonically as m2 is lowered (deeper wells -> more ordered), for each L, within 3 sigma.
for a, L in enumerate(Ls):
    step = np.diff(U4[a]) / np.hypot(dU4[a, 1:], dU4[a, :-1])
    print(f"[4] L={L:2d}: worst step in U4 = {step.min():+.1f} sigma (must be > -3)")
    assert step.min() > -3

if name == "coarse":
    # [5] The curves cross exactly once: the small lattice is MORE ordered in the symmetric phase
    #     (finite-size tail) and LESS ordered in the broken phase (larger L approaches 2/3 faster).
    D, dD = U4[1] - U4[0], np.hypot(dU4[1], dU4[0])
    sig = np.abs(D) > 2 * dD
    print(f"[5] U4({Ls[1]}) - U4({Ls[0]}): {D[0] / dD[0]:+.1f} sigma at m2={m2[0]:+.2f}, "
          f"{D[-1] / dD[-1]:+.1f} sigma at m2={m2[-1]:+.2f}")
    assert D[0] < -3 * dD[0] and D[-1] > 3 * dD[-1]
    m2c, _ = crossing(m2[sig], D[sig])
    print(f"    one significant sign change -> crossing at m2 ~ {m2c:+.3f}")

    # Not asserted (coarse grid, only L = 8 and 16): where the crossing sits, as a preview of 6b and 7.
    U4c = 0.5 * (np.interp(-m2c, -m2, U4[0]) + np.interp(-m2c, -m2, U4[1]))  # np.interp needs increasing x
    ratio = np.interp(-m2c, -m2, chi[1] / chi[0])
    print(f"    U4 at crossing ~ {U4c:.3f}   (2D Ising: {U4_ISING})")
    print(f"    chi({Ls[1]})/chi({Ls[0]}) at crossing ~ {ratio:.2f}   (2D Ising: 2^(7/4) = {2 ** 1.75:.2f}; "
          f"symmetric phase -> 1, broken -> 4)")
    print("All step-6a checks passed.")
    sys.exit()

# [6] Reproducibility: the fine scan (seed 1) re-measures L = 8, 16 at the m2 values it shares with the coarse
#     scan (seed 0). Independent chains of the same physics must agree within their error bars (|z| < 3);
#     a failure means either the error bars are too small or one of the chains is not equilibrated.
dc, U4c_, dU4c_, chic_, dchic_ = load_scan("coarse")
for b, L in enumerate(dc["Ls"]):
    a = list(Ls).index(L)
    for i_c, x in enumerate(dc["m2"]):
        i_f = np.flatnonzero(np.isclose(m2, x))
        if len(i_f) == 0:
            continue
        i_f = i_f[0]
        zU = (U4[a, i_f] - U4c_[b, i_c]) / np.hypot(dU4[a, i_f], dU4c_[b, i_c])
        zc = (chi[a, i_f] - chic_[b, i_c]) / np.hypot(dchi[a, i_f], dchic_[b, i_c])
        print(f"[6] L={L:2d} m2={x:+.3f}: U4 {U4[a, i_f]:.4f} vs {U4c_[b, i_c]:.4f} (z={zU:+.1f})   "
              f"chi {chi[a, i_f]:7.2f} vs {chic_[b, i_c]:7.2f} (z={zc:+.1f})")
        assert abs(zU) < 3 and abs(zc) < 3

# Errors on crossings and exponents: parametric bootstrap. Every (L, m2) point is an independent chain, so
# each replica redraws every U4 and chi from N(value, jackknife error) and redoes the whole analysis.
rng = np.random.default_rng(0)
N_BOOT = 300
replicas = [(U4 + dU4 * rng.normal(size=U4.shape), chi + dchi * rng.normal(size=chi.shape)) for _ in range(N_BOOT)]


def chi_at(m2c, chi_):
    """chi(L) at m2c for every L, interpolated in log chi (chi is ~ exponential in m2 here)."""
    return np.array([np.exp(np.interp(-m2c, -m2, np.log(chi_[a]))) for a in range(len(Ls))])


# [7] Binder crossings for each (L, 2L) pair. U4 at the crossing is universal: for the 2D Ising class on
#     periodic squares U4* = 0.61069. Corrections shrink with L, so the largest pair must be within 0.02.
pairs = doubling_pairs(Ls)
for L1, L2 in pairs:
    m2c, U4x = pair_crossing(m2, U4, dU4, Ls, L1, L2)
    boot = np.array([pair_crossing(m2, u, dU4, Ls, L1, L2) for u, _ in replicas])
    print(f"[7] L={L1:2d},{L2:2d}: crossing m2 = {m2c:+.4f} +- {boot[:, 0].std():.4f}   "
          f"U4 = {U4x:.4f} +- {boot[:, 1].std():.4f}   (Ising {U4_ISING})")
L1, L2 = pairs[-1]
m2c, U4x = pair_crossing(m2, U4, dU4, Ls, L1, L2)
assert abs(U4x - U4_ISING) < 0.02

# [8] At m2_c, chi = V<M^2> ~ L^(gamma/nu) with gamma/nu = 2 - 2 Delta_sigma = 7/4 for 2D Ising.
#     Fit log chi vs log L over all five sizes at the m2_c of the largest pair. The bootstrap error includes
#     the uncertainty in m2_c itself (each replica recomputes its own crossing), which dominates: chi changes by
#     a factor of ~4 across the transition.
def slope(u, c):
    m2c_r, _ = pair_crossing(m2, u, dU4, Ls, L1, L2)
    return np.polyfit(np.log(Ls), np.log(chi_at(m2c_r, c)), 1)[0]


g = slope(U4, chi)
dg = np.std([slope(u, c) for u, c in replicas])
print(f"[8] chi(L) at m2_c = {m2c:+.4f}: " + "  ".join(f"L={L}: {c:.1f}" for L, c in zip(Ls, chi_at(m2c, chi))))
print(f"    gamma/nu = {g:.3f} +- {dg:.3f}   (2D Ising: 7/4 = 1.75)")
assert abs(g - 1.75) < 0.1

# Not asserted (that is step 7): the effective exponent from each doubling pair at its own crossing.
for L1_, L2_ in pairs:
    m2c_, _ = pair_crossing(m2, U4, dU4, Ls, L1_, L2_)
    c = chi_at(m2c_, chi)
    ratio = c[list(Ls).index(L2_)] / c[list(Ls).index(L1_)]
    print(f"    L={L1_:2d},{L2_:2d}: chi ratio {ratio:.3f} (2^(7/4) = {2 ** 1.75:.3f}) -> "
          f"Delta_sigma_eff = {1 - 0.5 * np.log2(ratio):.4f} (exact 1/8)")
print("All step-6b checks passed.")
