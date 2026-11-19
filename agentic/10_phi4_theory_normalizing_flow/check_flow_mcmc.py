"""Step 5 checks: the flow on the interacting theory, against local Metropolis (+ global flip).

Three estimators of the same observables:
  Metropolis + flip   reference (exact, autocorrelated)
  flow raw            plain average over flow samples: BIASED unless q = p exactly
  flow MCMC           independence Metropolis with the flow as proposal: exact

Run after: python train.py sym  and  python flow_mcmc.py sym
"""
from pathlib import Path

import numpy as np

from flow_mcmc import estimates
from phi4 import magnetization, mean_err, tau_int

d = np.load(Path(__file__).parent / "runs" / "sym_samples.npz")
m2, lam, L = float(d["m2"]), float(d["lam"]), int(d["L"])
met, flow, idx, accepted = d["met_phi"], d["flow_phi"], d["idx"], d["accepted"]
N = len(idx)
chain = flow[idx]

est = {"Metropolis + flip": estimates(met, m2, lam), "flow raw": estimates(flow, m2, lam),
       "flow MCMC": estimates(chain, m2, lam)}

print(f"L = {L}, m2 = {m2}, lam = {lam}, N = {N}.  Flow-MCMC acceptance = {accepted[1:].mean():.3f}")
print(f"{'observable':14s} {'Metropolis + flip':>20s} {'flow raw':>20s} {'z raw':>7s} {'flow MCMC':>20s} {'z MCMC':>7s}")
worst = 0.0
for name in est["Metropolis + flip"]:
    (a, ea), (b, eb), (c, ec) = (est[k][name] for k in est)
    z_raw, z_mcmc = (b - a) / np.hypot(ea, eb), (c - a) / np.hypot(ea, ec)
    worst = max(worst, abs(z_mcmc))
    print(f"{name:14s} {a:12.5f} ±{ea:7.5f} {b:12.5f} ±{eb:7.5f} {z_raw:+7.1f} {c:12.5f} ±{ec:7.5f} {z_mcmc:+7.1f}")

# 1. Exactness: every flow-MCMC observable agrees with Metropolis.
print(f"[1] worst |z| flow MCMC vs Metropolis = {worst:.2f}")
assert worst < 4

# 2. phi -> -phi symmetry. Nothing in the architecture enforces it (conv biases can favour one sign), so the
#    raw flow may be lopsided -- that is reported, not asserted. What IS guaranteed: the Metropolis-corrected
#    chain samples p exactly, so its fraction of M > 0 must be 1/2.
frac_raw = np.mean(magnetization(flow) > 0)
frac_chain, err_chain = mean_err((magnetization(chain) > 0).astype(float))
print(f"[2] fraction with M > 0:  flow raw = {frac_raw:.4f} (+- {0.5 / np.sqrt(N):.4f}; not guaranteed)   "
      f"flow MCMC = {frac_chain:.4f} +- {err_chain:.4f}   (exact 0.5)")
assert abs(frac_chain - 0.5) < 4 * err_chain

# 3. The chain really is a chain: rejections repeat states, so tau_int > 1, consistent with acceptance.
tau_flow = tau_int(np.abs(magnetization(chain)))
tau_met = tau_int(np.abs(magnetization(met)))
print(f"[3] tau_int(|M|): flow MCMC = {tau_flow:.2f}   Metropolis + flip = {tau_met:.2f}")
print(f"    effective samples / second: flow MCMC {N / tau_flow / float(d['t_flow']):.0f}   "
      f"Metropolis {N / tau_met / float(d['t_met']):.0f}")
assert tau_flow >= 1 - 0.1

print("All step-5 checks passed.")
