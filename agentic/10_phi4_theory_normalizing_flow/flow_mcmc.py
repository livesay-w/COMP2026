"""Step 5: exact sampling with a trained flow, and a matched local-Metropolis baseline.

Independence Metropolis: propose phi' ~ q (fresh, independent of the current state) and accept with
    min(1, w'/w),   w = e^{-S} / q.
Detailed balance holds for p = e^{-S}/Z for ANY q that covers p, so the chain is exact even when the
flow is imperfect; a worse flow just rejects more often (and repeats the current state).

Run: python flow_mcmc.py sym      (needs runs/sym.pt; writes runs/sym_samples.npz)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

from phi4 import action, correlator, jackknife, magnetization, metropolis
from train import load_flow

RUNS = Path(__file__).parent / "runs"


def estimates(phi, m2, lam):
    """Observables with jackknife errors from a sequence of configurations (N, L, L). Dict name -> (value, err)."""
    V = phi.shape[-1] * phi.shape[-2]
    M = magnetization(phi)
    S = action(torch.from_numpy(np.asarray(phi, dtype=np.float64)), m2, lam).numpy()
    out = {
        "<|M|>": jackknife(lambda a: a, np.abs(M)),
        "chi = V<M^2>": jackknife(lambda a: V * a, M**2),
        "Binder U4": jackknife(lambda a, b: 1 - b / (3 * a**2), M**2, M**4),
        "<S>/V": jackknife(lambda a: a / V, S),
    }
    C = jackknife(lambda a: a, correlator(phi))
    for r in range(C[0].shape[0]):
        out[f"C({r})"] = (C[0][r], C[1][r])
    return out


def flow_proposals(flow, n, m2, lam, batch=1000):
    """n independent flow samples and their log importance weights log w = -S - log q."""
    phis, logws = [], []
    with torch.no_grad():
        for _ in range(n // batch):
            phi, log_q = flow.sample(batch)
            phis.append(phi)
            logws.append(-action(phi, m2, lam) - log_q)
    return torch.cat(phis).double().numpy(), torch.cat(logws).double().numpy()


def independence_metropolis(logw, rng):
    """Chain over pre-drawn proposals. Returns (index of the state held at each step, accepted flags)."""
    n = len(logw)
    idx = np.empty(n, dtype=int)
    accepted = np.zeros(n, dtype=bool)
    log_u = np.log(rng.random(n))
    cur = 0
    idx[0] = 0
    for i in range(1, n):
        if log_u[i] < logw[i] - logw[cur]:
            cur = i
            accepted[i] = True
        idx[i] = cur
    return idx, accepted


if __name__ == "__main__":
    name = sys.argv[1]
    N = 50_000
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    flow, cfg, _ = load_flow(name)
    L, m2, lam = cfg["L"], cfg["m2"], cfg["lam"]

    t0 = time.time()
    flow_phi, logw = flow_proposals(flow, N, m2, lam)
    idx, accepted = independence_metropolis(logw, rng)
    t_flow = time.time() - t0
    print(f"flow: {N} proposals + chain in {t_flow:.1f} s,  acceptance {accepted[1:].mean():.3f}")

    t0 = time.time()
    met_phi, met_acc = metropolis(L, m2, lam, n_sweeps=N, n_therm=2000, step=1.5, rng=rng, flip=True)
    t_met = time.time() - t0
    print(f"Metropolis + flip: {N} sweeps in {t_met:.1f} s,  local acceptance {met_acc:.3f}")

    np.savez_compressed(RUNS / f"{name}_samples.npz", flow_phi=flow_phi.astype(np.float32), logw=logw, idx=idx,
                        accepted=accepted, met_phi=met_phi.astype(np.float32), t_flow=t_flow, t_met=t_met,
                        L=L, m2=m2, lam=lam)
    print(f"saved runs/{name}_samples.npz")
