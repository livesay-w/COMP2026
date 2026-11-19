"""Steps 2-3 checks: one coupling layer, then the full flow.

All checks use RANDOM nonzero kernel weights. At initialisation the last conv layer is zero,
so the flow is the identity and most of these checks would pass trivially.

Run: python check_flow.py
"""
import torch
from torch.autograd.functional import jacobian

from flow import AffineCoupling, Flow, log_prior

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)


def randomize(model, scale=0.3):
    for p in model.parameters():
        p.data = scale * torch.randn_like(p)


def brute_logdet(f, x):
    """log|det| of the full V x V Jacobian of f at one configuration x (L, L)."""
    L = x.shape[0]
    J = jacobian(lambda v: f(v.reshape(1, L, L)).reshape(L * L), x.reshape(L * L))
    return torch.linalg.slogdet(J)[1]


L, hidden = 4, 4
x = torch.randn(5, L, L)

# ---------------------------------------------------------------- one coupling layer
layer = AffineCoupling(L, parity=0, hidden=hidden)
randomize(layer)
y, logdet = layer(x)
x_back, logdet_rev = layer.reverse(y)

# 1. Invertibility, and the reverse log-det is minus the forward one.
print(f"[1] layer: max |reverse(forward(x)) - x| = {(x_back - x).abs().max():.2e}")
assert torch.allclose(x_back, x, atol=1e-12)
assert torch.allclose(logdet_rev, -logdet, atol=1e-12)

# 2. Frozen sites pass through exactly; active sites actually change (the layer is not the identity).
frozen = layer.frozen.bool()
print(f"[2] layer: frozen sites changed by {(y - x)[:, frozen].abs().max():.1e}, "
      f"active sites by up to {(y - x)[:, ~frozen].abs().max():.2f}")
assert torch.equal(y[:, frozen], x[:, frozen])
assert (y - x)[:, ~frozen].abs().max() > 1e-2

# 3. Cheap log-det (sum of s over active sites) equals the brute-force log|det J| of the full 16x16 Jacobian.
brute = torch.stack([brute_logdet(lambda v: layer(v)[0], xi) for xi in x])
print(f"[3] layer: max |logdet - brute-force slogdet| = {(logdet - brute).abs().max():.2e}")
assert torch.allclose(logdet, brute, atol=1e-10)

# 4. Translation equivariance. Shifting by (1, 1) keeps the checkerboard pattern, so the layer must commute
#    with it. Shifting by (1, 0) swaps the colours, so it must NOT commute -- this shows the test has teeth.
def shift(v, a, b):
    return torch.roll(v, shifts=(a, b), dims=(1, 2))

err_11 = (layer(shift(x, 1, 1))[0] - shift(y, 1, 1)).abs().max()
err_10 = (layer(shift(x, 1, 0))[0] - shift(y, 1, 0)).abs().max()
print(f"[4] layer: equivariance error, shift (1,1) = {err_11:.1e}   shift (1,0) = {err_10:.2f}")
assert err_11 < 1e-12
assert err_10 > 1e-2

# ---------------------------------------------------------------- full flow
flow = Flow(L, n_layers=4, hidden=hidden)
randomize(flow)
z = torch.randn(5, L, L)
phi, logdet = flow(z)
z_back, logdet_rev = flow.reverse(phi)

# 5. Invertibility and log-det of the whole stack (catches parity / accumulation-sign bugs that one layer can't).
brute = torch.stack([brute_logdet(lambda v: flow(v)[0], zi) for zi in z])
print(f"[5] flow:  max |z_back - z| = {(z_back - z).abs().max():.2e}   "
      f"max |logdet - brute| = {(logdet - brute).abs().max():.2e}")
assert torch.allclose(z_back, z, atol=1e-10)
assert torch.allclose(logdet, brute, atol=1e-10)
assert torch.allclose(logdet_rev, -logdet, atol=1e-10)

# 6. Two routes to log q agree: sampling forward vs evaluating backward on the same phi.
phi, log_q_sample = flow.sample(5)
log_q_eval = flow.log_prob(phi)
print(f"[6] flow:  max |log q (sample) - log q (log_prob)| = {(log_q_sample - log_q_eval).abs().max():.2e}")
assert torch.allclose(log_q_sample, log_q_eval, atol=1e-10)

# 7. A freshly built flow is exactly the identity, so q is exactly the Gaussian prior.
fresh = Flow(L, n_layers=4, hidden=hidden)
phi, log_q = fresh(z)[0], fresh.log_prob(z)
print(f"[7] fresh flow: phi == z: {torch.equal(phi, z)},  log q == log prior: {torch.equal(log_q, log_prior(z))}")
assert torch.equal(phi, z) and torch.equal(log_q, log_prior(z))

print("All step-2/3 checks passed.")
