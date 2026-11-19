"""Step 6c checks: flow-MCMC at m2 = -3.80 (next to m2_c = -3.806) against the 6b Metropolis chains, L = 8 and 16.
Uses the Z2-symmetrized proposal (runs/crit_sym_samples_L<L>_N<N>.npz from critical.py, N = critical.N_SAMPLES).

The 6b chain at grid index 6 has exactly the same couplings (m2 = -3.80, lam = 1), different algorithm, different
seed: any disagreement in U4 or chi beyond the errors means the flow-MCMC chain is not sampling e^{-S}/Z.

Run after: python critical.py
"""
from pathlib import Path

import numpy as np
import torch

from critical import samples_file
from phi4 import mean_err, tau_int
from scan import binder_chi
from train import ess

RUNS = Path(__file__).parent / "runs"
N_BLOCKS = 50  # jackknife blocks used by binder_chi

scan = np.load(RUNS / "scan_fine.npz")
i_c = 6
assert np.isclose(scan["m2"][i_c], -3.80) and scan["lam"] == 1.0

for L in (8, 16):
    d = np.load(samples_file(L))
    assert np.isclose(d["m2"], scan["m2"][i_c]) and d["lam"] == scan["lam"]
    M_raw, idx, accepted = d["M"], d["idx"], d["accepted"]
    M_flow, M_met = M_raw[idx], scan[f"M_{L}"][i_c].astype(np.float64)
    est = {"Metropolis 6b": binder_chi(M_met, L), "flow raw": binder_chi(M_raw, L), "flow MCMC": binder_chi(M_flow, L)}

    print(f"--- L = {L}, m2 = {float(d['m2']):.3f}: flow-MCMC acceptance {accepted[1:].mean():.3f}, "
          f"ESS of the proposals {ess(torch.from_numpy(d['logw'])):.3f}")
    for k, name in enumerate(("U4", "chi")):
        (a, ea), (b, eb), (c, ec) = (est[key][k] for key in est)
        z_raw, z_mcmc = (b - a) / np.hypot(ea, eb), (c - a) / np.hypot(ea, ec)
        print(f"    {name:4s} Metropolis {a:9.4f} ± {ea:.4f}   flow raw {b:9.4f} ± {eb:.4f} (z {z_raw:+6.1f})   "
              f"flow MCMC {c:9.4f} ± {ec:.4f} (z {z_mcmc:+5.1f})")
        # 1. Exactness: flow MCMC reproduces Metropolis. (The raw flow is biased and is only reported.)
        assert abs(z_mcmc) < 3, f"L={L} {name}: flow MCMC disagrees with Metropolis"

    # 2. The flow-chain errors are trustworthy only if each jackknife block spans >= 10 autocorrelation times.
    tau_flow, tau_met = tau_int(M_flow**2), tau_int(M_met**2)
    per_block = len(M_flow) / N_BLOCKS / tau_flow
    print(f"[2] tau_int(M^2): flow MCMC {tau_flow:.1f} proposals -> {per_block:.0f} tau per block (need >= 10);   "
          f"Metropolis {tau_met:.1f} sweeps")
    assert per_block >= 10

    # 3. phi -> -phi balance of the chain. The plain flow collapsed onto M > 0 (100% of raw samples), and that also
    #    under-covered the M ~ 0 configurations between the wells: L = 16 failed [1] with U4 at z = +9. The
    #    symmetrized proposal is balanced by construction, so this is a sanity check; [1] is the real test.
    frac, err = mean_err((M_flow > 0).astype(float))
    print(f"[3] fraction with M > 0: proposals {np.mean(M_raw > 0):.3f}   flow MCMC {frac:.3f} ± {err:.3f}   (exact 0.5)")
    assert abs(frac - 0.5) < 4 * err

print("All step-6c checks passed.")
