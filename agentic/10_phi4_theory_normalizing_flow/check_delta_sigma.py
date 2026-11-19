"""Step 7 checks: the fixed-U4 pipeline (7a, delta_sigma.py) and the flow at m2*(8) (7b, flow_fixed_u4.py).

Run after: python delta_sigma.py  and  python -u flow_fixed_u4.py
"""
from pathlib import Path

import numpy as np

from delta_sigma import N_BLOCKS, OMEGAS, analyse, extrapolate, jk_err, replicas, target_scan
from phi4 import mean_err, tau_int
from scan import U4_ISING, binder_chi, doubling_pairs

RUNS = Path(__file__).parent / "runs"
scan = np.load(RUNS / "scan_fine.npz")
Ls, m2 = [int(L) for L in scan["Ls"]], scan["m2"]
pairs = doubling_pairs(Ls)

# 1. Synthetic data with a known answer, on the 6b grid. U4 linear in m2, crossing U4* at m2_c(L) = -3.80 - 1.1/L
#    (between grid points), and chi = L^(7/4) e^(3 U4): an exact power law, Delta_sigma = 1/8, no corrections.
#    The pipeline must return m2* = m2_c(L), Delta_sigma_eff = 1/8 for every pair, and 1/8 with a = 0 for every omega.
m2c_syn = {L: -3.80 - 1.1 / L for L in Ls}
U4_syn = {L: U4_ISING + 0.5 * (m2c_syn[L] - m2) for L in Ls}
chi_syn = {L: L**1.75 * np.exp(3 * U4_syn[L]) for L in Ls}
syn = analyse(m2, U4_syn, chi_syn, Ls)
print(f"[1] synthetic: max |m2* - m2_c| = {np.max(np.abs(syn['m2_star'] - [m2c_syn[L] for L in Ls])):.1e}   "
      f"max |dse - 1/8| = {np.max(np.abs(np.r_[syn['dse'], syn['dse_quad']] - 0.125)):.1e}   "
      f"max |Delta_sigma - 1/8| = {np.max(np.abs(syn['delta_sigma'] - 0.125)):.1e}   max |a| = {np.max(np.abs(syn['a'])):.1e}")
assert np.allclose(syn["m2_star"], [m2c_syn[L] for L in Ls], atol=1e-12)
assert np.allclose(np.r_[syn["dse"], syn["dse_quad"], syn["delta_sigma"]], 0.125, atol=1e-12)
assert np.allclose(syn["a"], 0, atol=1e-10)
#    ... and the fit alone recovers a known correction amplitude.
Lp = np.array([a for a, _ in pairs], dtype=float)
fit = extrapolate(Lp, 0.125 + 0.7 * Lp**-1.75, 1.75)
print(f"    fit of 1/8 + 0.7 L^(-7/4): Delta_sigma = {fit[0]:.12f}, a = {fit[1]:.12f}")
assert np.allclose(fit, [0.125, 0.7], atol=1e-10)

# 2. The joint jackknife reproduces the per-point values and errors of scan.binder_chi (the 6b analysis).
for L in (8, 32):
    U4r, chir = replicas(scan[f"M_{L}"], L)
    ref = np.array([np.ravel(binder_chi(M, L)) for M in scan[f"M_{L}"]])  # columns U4, dU4, chi, dchi
    ours = np.stack([U4r[0], jk_err(U4r), chir[0], jk_err(chir)], axis=1)
    print(f"[2] L={L}: max relative difference to binder_chi (values and errors) = {np.max(np.abs(ours / ref - 1)):.1e}")
    assert np.allclose(ours, ref, rtol=1e-8)

# Real data.
ds = np.load(RUNS / "delta_sigma.npz")
val = {k[4:]: ds[k][0] for k in ds.files if k.startswith("rep_")}
err = {k[4:]: jk_err(ds[k]) for k in ds.files if k.startswith("rep_")}

# 3. Interpolation systematic: a quadratic fit to the 4 nearest points moves each Delta_sigma_eff by < 1 sigma.
#    (chi* itself is listed too; L = 32 has the coarsest grid relative to how fast chi changes.)
print("[3] chi*: quadratic - linear = " + "  ".join(
    f"L={L}: {(val['chi_star_quad'][j] - val['chi_star'][j]) / err['chi_star'][j]:+.2f} sigma" for j, L in enumerate(Ls)))
shift = (val["dse_quad"] - val["dse"]) / err["dse"]
print("    Delta_sigma_eff: quadratic - linear = " + "  ".join(f"{a},{b}: {s:+.2f} sigma" for (a, b), s in zip(pairs, shift)))
assert np.all(np.abs(shift) < 1)

# 4. Not asserted: m2*(L) should drift toward the 6b crossing m2_c = -3.806 as L grows.
print("[4] m2*(L): " + "  ".join(f"L={L}: {v:.4f} ± {e:.4f}" for L, v, e in zip(Ls, val["m2_star"], err["m2_star"]))
      + "   (6b crossing: -3.806 ± 0.011)")

# 5. Goal: the extrapolated Delta_sigma (omega = 7/4) within 3 sigma of 1/8. Also reported: sensitivity to omega and
#    to the U4 target. The 6b crossings sit at U4 = 0.590-0.596, below U4* = 0.611, so corrections to scaling in U4
#    itself feed into chi*. Targets above 0.62 leave the L = 8 grid.
k = OMEGAS.index(1.75)
z = (val["delta_sigma"][k] - 0.125) / err["delta_sigma"][k]
print(f"[5] Delta_sigma (omega = 7/4) = {val['delta_sigma'][k]:.4f} ± {err['delta_sigma'][k]:.4f}   (1/8: z = {z:+.1f});   "
      + "  ".join(f"omega = {w}: {v:.4f}" for w, v in zip(OMEGAS, val["delta_sigma"])))
U4r = {L: replicas(scan[f"M_{L}"], L) for L in Ls}
for target in (0.58, 0.59, 0.60, U4_ISING):
    reps = target_scan(m2, U4r, Ls, target)
    e = jk_err(reps)
    print(f"    U4 target {target:.5f}: Delta_sigma_eff " + "  ".join(f"{v:.3f}±{s:.3f}" for v, s in zip(reps[0, :3], e[:3]))
          + f"   extrapolated {reps[0, 3]:.3f} ± {e[3]:.3f}")
assert abs(z) < 3

# 6. Step 7b: flow MCMC at m2*(8) agrees with the 6b Metropolis chains interpolated (linearly in m2) to the same m2.
f = np.load(RUNS / "fixed_u4_L8.npz")
L, m2f = int(f["L"]), float(f["m2"])
M_flow = f["M"][f["idx"]]
U4r8, chir8 = U4r[L]
met = [np.array([np.interp(-m2f, -m2, row) for row in q]) for q in (U4r8, chir8)]  # np.interp needs increasing x
flow = binder_chi(M_flow, L)
print(f"[6] L={L}, m2 = {m2f}: flow-MCMC acceptance {f['accepted'][1:].mean():.3f}")
for (name, mq), (fv, fe) in zip((("U4", met[0]), ("chi", met[1])), flow):
    zf = (fv - mq[0]) / np.hypot(fe, jk_err(mq))
    print(f"    {name:4s} Metropolis (interpolated) {mq[0]:8.4f} ± {jk_err(mq):.4f}   flow MCMC {fv:8.4f} ± {fe:.4f}   z = {zf:+.1f}")
    assert abs(zf) < 3
print(f"    flow-MCMC U4 = {flow[0][0]:.4f} vs the target U4* = {U4_ISING} (m2 was set from the Metropolis data)")
per_block = len(M_flow) / N_BLOCKS / tau_int(M_flow**2)
frac, ferr = mean_err((M_flow > 0).astype(float))
print(f"    {per_block:.0f} tau_int per jackknife block (need >= 10);   fraction M > 0 = {frac:.3f} ± {ferr:.3f}")
assert per_block >= 10 and abs(frac - 0.5) < 4 * ferr

print("All step-7 checks passed.")
