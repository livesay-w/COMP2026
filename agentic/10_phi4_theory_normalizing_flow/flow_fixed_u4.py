"""Step 7b: the flow at the fixed-U4 point of L = 8, m2*(8) from step 7a (where U4 = 0.61069).

Warm-start from the 6c flow (runs/crit_sym_L8_m-3.80.pt), train one more Z2-symmetrized stage at m2*(8), then
independence Metropolis with 500k proposals. check_delta_sigma.py compares U4 and chi with the 6b Metropolis chains
interpolated to the same m2. (L = 16 is out of reach for this flow: see "Later: 6c L = 16" in Phi4_Normalizing_Flow.md.)

Lattice units, lam = 1.  Run after delta_sigma.py: python -u flow_fixed_u4.py   (writes runs/fixed_u4_L8.pt, .npz)
"""
import time
from pathlib import Path

import numpy as np
import torch

from critical import BATCH, LAM, LR, Z2Symmetrized, flow_m_logw, transfer
from flow_mcmc import independence_metropolis
from train import train

RUNS = Path(__file__).parent / "runs"
L, N_STEPS, N_SAMPLES = 8, 1500, 500_000

ds = np.load(RUNS / "delta_sigma.npz")
m2 = round(float(ds["rep_m2_star"][0, list(ds["Ls"]).index(L)]), 3)  # 0.001 is far below its jackknife error
print(f"L={L}: training at m2*(8) = {m2} (lam = {LAM})", flush=True)

torch.manual_seed(30)
flow = transfer(torch.load(RUNS / "crit_sym_L8_m-3.80.pt")["state_dict"], L)
t0 = time.time()
history = train(Z2Symmetrized(flow), m2, LAM, N_STEPS, BATCH, LR, log_every=250, log_all_until=0)
torch.save({"L": L, "m2": m2, "lam": LAM, "state_dict": flow.state_dict(), "history": history}, RUNS / "fixed_u4_L8.pt")
print(f"  saved runs/fixed_u4_L8.pt  ({time.time() - t0:.0f} s)", flush=True)

torch.manual_seed(31)
t0 = time.time()
M, logw = flow_m_logw(Z2Symmetrized(flow), N_SAMPLES, m2, LAM)
idx, accepted = independence_metropolis(logw, np.random.default_rng(31))
print(f"  {N_SAMPLES} proposals + chain in {time.time() - t0:.0f} s, acceptance {accepted[1:].mean():.3f}")
np.savez(RUNS / "fixed_u4_L8.npz", M=M, logw=logw, idx=idx, accepted=accepted, L=L, m2=m2, lam=LAM)
