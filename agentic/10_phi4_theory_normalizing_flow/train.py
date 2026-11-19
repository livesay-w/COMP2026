"""Step 4+: train the flow by reverse KL against the action. No data: the flow learns from S alone.

    loss = E_{phi ~ q} [ log q(phi) + S(phi) ]  =  KL(q || p) - log Z   >=   -log Z

One training step: draw fresh noise, push it through the flow (every coupling layer applies its
CNN kernels once), score the resulting fields with the action, and update the kernel weights by
gradient descent. The fields are NOT iterated; the kernels are.

Run: python train.py free          (configs below; trained weights go to runs/<name>.pt)
"""
import sys
import time
from pathlib import Path

import torch

from flow import Flow
from phi4 import action

RUNS = Path(__file__).parent / "runs"

CONFIGS = {
    # Free theory: exactly solvable, used to validate training (check_train_free.py).
    "free": dict(L=8, m2=1.0, lam=0.0, n_layers=16, hidden=8, batch_size=64, lr=1e-3, n_steps=4000, seed=0),
    # Interacting, symmetric phase (transition at lam = 1 is near m2 ~ -3.5 to -4). Step 5.
    "sym": dict(L=8, m2=-2.0, lam=1.0, n_layers=16, hidden=8, batch_size=64, lr=1e-3, n_steps=4000, seed=0),
}


def ess(logw):
    """Normalised effective sample size of importance weights w = p~/q, in [1/N, 1]. 1 means q = p."""
    return (torch.exp(2 * torch.logsumexp(logw, 0) - torch.logsumexp(2 * logw, 0)) / len(logw)).item()


def train(flow, m2, lam, n_steps, batch_size, lr, log_every=100, log_all_until=100):
    """Logged: step 0 (the untrained flow), every step up to log_all_until (where the big drop happens),
    then every log_every steps. Each logged loss is that step's batch, before the update."""
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    history = {"step": [], "loss": [], "spread": [], "ess": []}
    t0 = time.time()
    for step in range(n_steps + 1):
        phi, log_q = flow.sample(batch_size)
        logw = -action(phi, m2, lam) - log_q      # log of the importance weight e^{-S} / q
        loss = (-logw).mean()                     # = mean(log q + S)
        if step <= log_all_until or step % log_every == 0:
            lw = logw.detach()
            history["step"].append(step)
            history["loss"].append(loss.item())
            history["spread"].append(lw.std().item())  # -> 0 for a perfect flow
            history["ess"].append(ess(lw))
            if step % log_every == 0:
                print(f"step {step:5d}   loss {loss.item():9.4f}   std(log q + S) {lw.std().item():7.4f}   "
                      f"ESS {history['ess'][-1]:.3f}   ({time.time() - t0:.0f} s)")
        if step == n_steps:
            break
        opt.zero_grad()
        loss.backward()
        opt.step()
    return history


def load_flow(name):
    """Rebuild a trained flow from runs/<name>.pt. Returns (flow, config, history)."""
    saved = torch.load(RUNS / f"{name}.pt")
    cfg = saved["config"]
    flow = Flow(cfg["L"], cfg["n_layers"], cfg["hidden"])
    flow.load_state_dict(saved["state_dict"])
    return flow, cfg, saved["history"]


if __name__ == "__main__":
    name = sys.argv[1]
    cfg = CONFIGS[name]
    torch.manual_seed(cfg["seed"])
    flow = Flow(cfg["L"], cfg["n_layers"], cfg["hidden"])
    print(f"training '{name}': {cfg}")
    print(f"trainable parameters (all CNN kernel weights + biases): {sum(p.numel() for p in flow.parameters())}")
    history = train(flow, cfg["m2"], cfg["lam"], cfg["n_steps"], cfg["batch_size"], cfg["lr"])
    RUNS.mkdir(exist_ok=True)
    torch.save({"config": cfg, "state_dict": flow.state_dict(), "history": history}, RUNS / f"{name}.pt")
    print(f"saved runs/{name}.pt")
