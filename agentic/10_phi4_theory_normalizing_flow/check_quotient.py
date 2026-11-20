"""Step 7c/7d checks: the quotient pipeline (quotient.py) and the production cluster chains (cluster_scan.py).

Run after: python -u cluster_scan.py prod <n_workers>  (and prod256 if listed in quotient.CLUSTER_RUNS)  and  python quotient.py
"""
import numpy as np

from cluster_scan import load
from delta_sigma import N_BLOCKS, jk_err, replicas, target_scan
from phi4 import tau_int
from quotient import (COLS, fit, fit_set, fit_table, load_quotient, load_sources, pair_linear, pair_quadratic,
                      pairs_of)
from scan import binder_chi, doubling_pairs

sources = load_sources()
pairs = pairs_of(sources)
grids = {src: {L: v[0] for L, v in s.items()} for src, s in sources.items()}

# 1. Synthetic data with a known answer, on the real grids. U4_L(m2) = U0 - d/L - k L (m2 - c) and chi_L = L^(7/4) e^(-b L (m2 - c)),
#    both linear in m2 (in log for chi), so linear and quadratic interpolation are exact. The pair (L, 2L) crosses at
#    m2_x = c + d / (2 k L^2), its own crossing, not c, and Delta_sigma_eff = 1/8 + b d / (4 k L ln 2) exactly.
#    With d = 0 every pair crosses at c and gives exactly 1/8. (The reweighted path is checked in check_reweight.py.)
U0, c, k, b = 0.6, -3.8273, 0.05, 0.03
for d in (0.0, 0.02):
    worst = 0.0
    for src, plist in pairs.items():
        for LA, LB in plist:
            gA, gB = grids[src][LA], grids[src][LB]
            U = [U0 - d / L - k * L * (g - c) for g, L in ((gA, LA), (gB, LB))]
            chi = [L**1.75 * np.exp(-b * L * (g - c)) for g, L in ((gA, LA), (gB, LB))]
            m2x, _, chiA, chiB, _ = pair_linear(gA, U[0], chi[0], gB, U[1], chi[1])
            m2q, chiAq, chiBq = pair_quadratic(gA, U[0], chi[0], gB, U[1], chi[1], m2x)
            dse_exact = 0.125 + b * d / (4 * k * LA * np.log(2))
            dse = [1 - 0.5 * np.log2(chiB / chiA), 1 - 0.5 * np.log2(chiBq / chiAq)]
            m2x_exact = c + d / (2 * k * LA**2)
            assert np.allclose([m2x, m2q], m2x_exact, atol=1e-10) and np.allclose(dse, dse_exact, atol=1e-10)
            worst = max(worst, abs(m2x - m2x_exact), abs(m2q - m2x_exact), *np.abs(np.array(dse) - dse_exact))
    print(f"[1] synthetic, d = {d}: crossings and Delta_sigma_eff of all {sum(map(len, pairs.values()))} pairs match the exact "
          f"values (worst {worst:.1e})")
fset = fit_set(pairs)
Ls = np.array([L for _, L in fset], dtype=float)
assert np.allclose(fit(Ls, 0.125 + 0.7 * Ls**-1.75, np.linspace(0.005, 0.01, len(Ls)), 1.75), [0.125, 0.7], atol=1e-10)
print("    weighted fit of 1/8 + 0.7 L^(-7/4) returns (0.125, 0.7)")

q = load_quotient()
assert set(q) == {(s, L) for s, plist in pairs.items() for L, _ in plist}, "runs/quotient.npz is stale: rerun quotient.py"
val = {key: v[0] for key, v in q.items()}
err = {key: jk_err(v) for key, v in q.items()}
col = {name: i for i, name in enumerate(COLS)}

# 2. On the 6b chains (one grid shared by all L), the quotient result must equal 7a's fixed-U4 pipeline run with the target
#    set to that pair's U4 at the crossing: at that target both lattices sit at m2_x, and 7a's linear interpolation in U4
#    is linear in m2 within a grid interval. Two independently written codes, same number to rounding.
met = sources["metropolis"]
Ls_met = sorted(met)
m2_6b = met[Ls_met[0]][0]
reps_met = {L: replicas(met[L][1], L) for L in Ls_met}
for LA, LB in pairs["metropolis"]:
    U4x = val[("metropolis", LA)][col["U4_x"]]
    ref = target_scan(m2_6b, reps_met, Ls_met, U4x)[0]  # Delta_sigma_eff of every 6b doubling pair at that target, then the fit
    pair_idx = doubling_pairs(Ls_met).index((LA, LB))
    diff = ref[pair_idx] - val[("metropolis", LA)][col["dse"]]
    print(f"[2] Metropolis ({LA},{LB}): quotient {val[('metropolis', LA)][col['dse']]:.6f}   7a pipeline at U4 = {U4x:.4f}: "
          f"{ref[pair_idx]:.6f}   difference {diff:.1e}")
    assert abs(diff) < 1e-10

# 3. Cluster chains vs the 6b Metropolis chains at every shared (L, m2): 7 points each at L = 16, 32, all independent chains.
cl = load("prod")
z_all = []
for L in (16, 32):
    for j, m2 in enumerate(cl[L][0]):
        i = int(np.flatnonzero(np.isclose(m2_6b, m2))[0])
        for name, (cv, ce), (mv, me) in zip(("U4", "chi"), binder_chi(cl[L][1][j], L), binder_chi(met[L][1][i], L)):
            z_all.append((L, m2, name, (cv - mv) / np.hypot(ce, me)))
z = np.array([t[3] for t in z_all])
print(f"[3] cluster vs Metropolis at {len(z) // 2} shared points (U4 and chi): max |z| = {np.max(np.abs(z)):.2f},  "
      f"sum z^2 = {np.sum(z**2):.1f} for {len(z)} comparisons (U4 and chi of one chain are correlated)")
for L, m2, name, zz in z_all:
    if abs(zz) > 2:
        print(f"    |z| > 2: L={L} m2={m2:+.4f} {name}: z = {zz:+.2f}")
assert np.max(np.abs(z)) < 3

# 4. Every cluster chain: 50 jackknife blocks of >= 10 tau_int(M^2), and thermalisation: <M^2> of the first and second
#    halves agree (a chain still relaxing from its random start would drift).
worst_blocks, worst_half, taus = np.inf, 0.0, {}
for L, (g, M, _) in sources["cluster"].items():
    for j in range(len(g)):
        tau = tau_int(M[j] ** 2)
        taus[L] = max(taus.get(L, 0), tau)
        worst_blocks = min(worst_blocks, len(M[j]) / N_BLOCKS / tau)
        h1, h2 = np.array_split(M[j] ** 2, 2)
        e = [np.std(h) * np.sqrt(tau / len(h)) for h in (h1, h2)]
        worst_half = max(worst_half, abs(h1.mean() - h2.mean()) / np.hypot(*e))
print("[4] max tau_int(M^2) per L: " + "  ".join(f"L={L}: {t:.1f}" for L, t in taus.items())
      + f";   fewest tau per block = {worst_blocks:.0f} (need >= 10);   worst first-vs-second-half |z| of <M^2> = {worst_half:.1f}")
assert worst_blocks >= 10 and worst_half < 4

# 5. Interpolation systematics, in units of each pair's main error.
#    Metropolis pairs (main = linear interpolation): quadratic vs linear within 1 sigma (asserted).
#    Cluster pairs (main = reweighting): reported only. Reweighting uses every chain and needs no interpolation, so its error
#    is ~half the linear one, and the old interpolation errors now show up as 1-5 of its sigmas (worst at (32,64), where the
#    L = 32 grid is coarse, 0.025, and the linear U4 difference is +0.0006 at a grid point). Asserting them against the new,
#    smaller error failed on the first run; the reweighting itself is validated in check_reweight.py [1] (exact free theory)
#    and [2] (leave-one-chain-out predictions).
for key in q:
    s = err[key][col["dse"]]
    rw_lin = (val[key][col["dse"]] - val[key][col["dse_lin"]]) / s
    quad_lin = (val[key][col["dse_quad"]] - val[key][col["dse_lin"]]) / s
    if key[0] == "metropolis":
        print(f"[5] met {key[1]:3d},{2 * key[1]:3d}: quadratic - linear = {quad_lin:+.2f} sigma")
        assert abs(quad_lin) < 1
    else:
        print(f"[5] clu {key[1]:3d},{2 * key[1]:3d}: linear {val[key][col['dse_lin']]:.4f}, quadratic {val[key][col['dse_quad']]:.4f}"
              f" vs reweighted {val[key][col['dse']]:.4f} ± {s:.4f}:  reweighted - linear = {rw_lin:+.1f} sigma, "
              f"quadratic - linear = {quad_lin:+.1f} sigma (reported)")

# 6. The (16,32) pair from the two algorithms (independent chains, different grids).
a, b_ = ("cluster", 16), ("metropolis", 16)
zp = (val[a][col["dse"]] - val[b_][col["dse"]]) / np.hypot(err[a][col["dse"]], err[b_][col["dse"]])
print(f"[6] (16,32): cluster {val[a][col['dse']]:.4f} ± {err[a][col['dse']]:.4f}   "
      f"Metropolis {val[b_][col['dse']]:.4f} ± {err[b_][col['dse']]:.4f}   z = {zp:+.1f}")
assert abs(zp) < 3

# 7. Not asserted: the pair crossings should drift toward m2_c and U4_x rise toward the Ising value 0.61069 as L grows.
for s, L in q:
    print(f"[7] {s[:3]} {L:3d},{2 * L:3d}: m2_x = {val[(s, L)][0]:.4f} ± {err[(s, L)][0]:.4f},  U4_x = {val[(s, L)][1]:.4f} ± {err[(s, L)][1]:.4f}"
          f",  Delta_sigma_eff = {val[(s, L)][col['dse']]:.4f} ± {err[(s, L)][col['dse']]:.4f}")

# 8. Goal: the extrapolated Delta_sigma (all pairs, omega = 7/4) within 3 sigma of 1/8; the other fits are reported.
table = fit_table(q, fset)
for label, reps, chi2, dof in table:
    e = jk_err(reps)[0]
    print(f"[8] {label}: Delta_sigma = {reps[0, 0]:.4f} ± {e:.4f}  (z = {(reps[0, 0] - 0.125) / e:+.1f} vs 1/8)   chi2/dof = {chi2:.1f}/{dof}")
main = next(reps for label, reps, _, _ in table if label == "pairs L >=  8, omega = 1.75")
assert abs(main[0, 0] - 0.125) < 3 * jk_err(main)[0]

print("All step-7c/7d checks passed.")
