"""Step 6c: warm-start the step-5 flow toward the critical point, then flow-MCMC at m2_c for L = 8 and 16.

Lattice units (a = 1), lam = 1. Target m2 = -3.80: the 6b grid point nearest m2_c = -3.806, so the flow-MCMC
result can be compared with an existing 6b Metropolis chain at exactly the same couplings (check_critical.py).

Warm start: begin from runs/sym.pt (L = 8, m2 = -2) and retrain in stages, moving m2 a little each time, so the
flow never has to jump far from what it already represents. The CNN kernels are 3x3 with circular padding, so
the same weights define a flow on any even L: the L = 16 stage starts from the final L = 8 kernels.

Z2 symmetrization: the first attempt (plain q) collapsed onto M > 0 and failed at L = 16 (U4 z = +9). Now both
training and MCMC use the mixture q_sym(phi) = [q(phi) + q(-phi)] / 2, which is even in phi like e^{-S}, so
the flow gains nothing by covering only one well (Phi4_Normalizing_Flow.md, row 6c).

Memory: one process, and only M and log w are kept from the flow samples, never the configurations. Every
stage is saved to runs/crit_sym_*.pt and skipped when rerun, so an interrupted run loses at most one stage.

Run: python -u critical.py        (writes runs/crit_sym_L<L>_m<m2>.pt and runs/crit_sym_samples_L<L>_N<N>.npz)
"""
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from flow import Flow
from phi4 import action
from train import load_flow, train

RUNS = Path(__file__).parent / "runs"
LAM, M2_C = 1.0, -3.80
N_LAYERS, HIDDEN, BATCH, LR = 16, 8, 64, 1e-3  # same architecture and optimiser as step 5 (train.py "sym")
# (L, m2, training steps). L = 8 first, so its result is complete even if the L = 16 stage is cut short.
STAGES = [(8, -2.6, 800), (8, -3.2, 800), (8, M2_C, 1500), (16, M2_C, 3000)]
# Flow proposals per L for independence Metropolis. 50k left tau_int(M^2) = 366 proposals at L = 8, i.e. 3 tau per
# jackknife block (check_critical.py [2] needs 10), so now 500k. Same seed: the first 50k proposals are the old ones.
N_SAMPLES = 500_000


def samples_file(L):
    return RUNS / f"crit_sym_samples_L{L}_N{N_SAMPLES}.npz"


class Z2Symmetrized(nn.Module):
    """Proposal q_sym(phi) = [q(phi) + q(-phi)] / 2: draw phi ~ q, then flip its sign with probability 1/2.

    q_sym(phi) = q_sym(-phi), so log q_sym is the same for phi and its flip and needs one extra reverse pass.
    Has .sample and .parameters like Flow, so train() and flow_m_logw() take it unchanged. In training the
    random sign does not change the loss (S and q_sym are both even), only which well the sample shows.
    """

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def sample(self, batch_size):
        phi, log_q = self.flow.sample(batch_size)
        log_q_sym = torch.logaddexp(log_q, self.flow.log_prob(-phi)) - math.log(2)
        sign = (2 * torch.randint(0, 2, (batch_size,)) - 1).to(phi.dtype)
        return sign[:, None, None] * phi, log_q_sym


def stage_file(L, m2):
    return RUNS / f"crit_sym_L{L}_m{m2:+.2f}.pt"


def transfer(state, L):
    """A flow on an L x L lattice with the kernels from `state`. Only the checkerboard masks depend on L."""
    flow = Flow(L, N_LAYERS, HIDDEN)
    kernels = {k: v for k, v in state.items() if not k.endswith("frozen")}
    missing, unexpected = flow.load_state_dict(kernels, strict=False)
    assert all(k.endswith("frozen") for k in missing) and not unexpected
    return flow


def run_stages():
    """Train every stage that has no saved file yet, each warm-started from the one before. Returns the final
    flow of each L."""
    sym, cfg, _ = load_flow("sym")
    assert (cfg["L"], cfg["m2"], cfg["lam"], cfg["n_layers"], cfg["hidden"]) == (8, -2.0, LAM, N_LAYERS, HIDDEN)
    state = sym.state_dict()
    final = {}
    for i, (L, m2, n_steps) in enumerate(STAGES):
        out = stage_file(L, m2)
        if out.exists():
            print(f"stage {i}: L={L} m2={m2:+.2f} already done ({out.name})", flush=True)
            state = torch.load(out)["state_dict"]
        else:
            print(f"stage {i}: L={L} m2={m2:+.2f}, {n_steps} steps", flush=True)
            torch.manual_seed(10 + i)
            flow = transfer(state, L)
            t0 = time.time()
            history = train(Z2Symmetrized(flow), m2, LAM, n_steps, BATCH, LR, log_every=250, log_all_until=0)
            state = flow.state_dict()
            torch.save({"L": L, "m2": m2, "lam": LAM, "state_dict": state, "history": history,
                        "seconds": time.time() - t0}, out)
            print(f"  saved {out.name}  ({time.time() - t0:.0f} s)", flush=True)
        final[L] = transfer(state, L)
    return final


def flow_m_logw(proposal, n, m2, lam, batch=1000):
    """M and log w = -S - log q of n independent proposals. Configurations are not kept (memory)."""
    Ms, logws = [], []
    with torch.no_grad():
        for _ in range(n // batch):
            phi, log_q = proposal.sample(batch)
            Ms.append(phi.mean(dim=(1, 2)).double())
            logws.append((-action(phi, m2, lam) - log_q).double())
    return torch.cat(Ms).numpy(), torch.cat(logws).numpy()


if __name__ == "__main__":
    from flow_mcmc import independence_metropolis

    final = run_stages()
    for L, flow in final.items():
        out = samples_file(L)
        if out.exists():
            print(f"L={L}: samples already done ({out.name})")
            continue
        torch.manual_seed(20 + L)
        t0 = time.time()
        M, logw = flow_m_logw(Z2Symmetrized(flow), N_SAMPLES, M2_C, LAM)
        idx, accepted = independence_metropolis(logw, np.random.default_rng(20 + L))
        print(f"L={L}: {N_SAMPLES} proposals + chain in {time.time() - t0:.0f} s, acceptance {accepted[1:].mean():.3f}")
        np.savez(out, M=M, logw=logw, idx=idx, accepted=accepted, L=L, m2=M2_C, lam=LAM)
