"""Step 1 checks: the action, the Metropolis baseline, and the exact free-theory formulas.

Run: python check_phi4.py
"""
import numpy as np
import torch

from phi4 import (action, binned_mean_err, correlator, exact_logZ, exact_propagator, jackknife, khat2,
                  local_action, magnetization, mean_err, metropolis, momentum_power, neighbor_sum, tau_int)

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(0)
m2, lam = -1.3, 0.7  # arbitrary interacting values for the algebraic checks


# 1. Local action vs full action: changing one site must change S by exactly the local difference.
#    Compares the torch action against the independent numpy local formula used by Metropolis.
L = 6
phi = rng.normal(size=(L, L))
new = phi.copy()
new[2, 3] += 0.8
dS_full = (action(torch.tensor(new)[None], m2, lam) - action(torch.tensor(phi)[None], m2, lam)).item()
nbr = neighbor_sum(phi)[2, 3]
dS_local = local_action(new[2, 3], nbr, m2, lam) - local_action(phi[2, 3], nbr, m2, lam)
print(f"[1] dS full = {dS_full:.12f}   dS local = {dS_local:.12f}")
assert abs(dS_full - dS_local) < 1e-10


# 2. Free theory as a quadratic form on 4x4: S = 1/2 phi^T K phi with K = (4 + m2) I - adjacency.
#    K's eigenvalues must be exactly m2 + khat^2, which is what exact_propagator / exact_logZ assume.
L, m2_free = 4, 1.0
V = L * L
K = (4 + m2_free) * np.eye(V)
for i in range(L):
    for j in range(L):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            K[i * L + j, ((i + di) % L) * L + (j + dj) % L] -= 1
phi = rng.normal(size=(L, L))
S_torch = action(torch.tensor(phi)[None], m2_free, 0.0).item()
S_quad = 0.5 * phi.ravel() @ K @ phi.ravel()
eig_K = np.sort(np.linalg.eigvalsh(K))
eig_exact = np.sort((m2_free + khat2(L)).ravel())
logZ_matrix = 0.5 * V * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(K)[1]
print(f"[2] S torch = {S_torch:.12f}   1/2 phi K phi = {S_quad:.12f}")
print(f"    max |eig K - (m2 + khat^2)| = {np.max(np.abs(eig_K - eig_exact)):.2e}")
print(f"    log Z from det K = {logZ_matrix:.10f}   exact_logZ = {exact_logZ(L, m2_free):.10f}")
assert abs(S_torch - S_quad) < 1e-10
assert np.allclose(eig_K, eig_exact, atol=1e-10)
assert abs(logZ_matrix - exact_logZ(L, m2_free)) < 1e-10


# 3. Metropolis reproduces the exact free propagator <|phi_k|^2>/V = 1/(m2 + khat^2) at every k.
L, m2_free = 8, 1.0
configs, acc = metropolis(L, m2_free, 0.0, n_sweeps=20000, n_therm=1000, step=1.0, rng=rng, flip=False)
mean, err = binned_mean_err(momentum_power(configs))
z = (mean - exact_propagator(L, m2_free)) / err
print(f"[3] Metropolis L={L}, m2={m2_free}, lam=0: acceptance = {acc:.3f}")
print(f"    k = 0:  measured {mean[0, 0]:.4f} +- {err[0, 0]:.4f}   exact {exact_propagator(L, m2_free)[0, 0]:.4f}")
print(f"    max |z-score| over all {L * L} momenta = {np.max(np.abs(z)):.2f}")
assert np.max(np.abs(z)) < 4.5


# 4. tau_int on a series with a known answer. AR(1): x_t = rho x_{t-1} + noise has rho(t) = rho^t,
#    so tau = 1 + 2 sum rho^t = (1 + rho) / (1 - rho). Independent samples (rho = 0) give exactly 1.
for rho in (0.0, 0.5, 0.9):
    n = 200_000
    x = np.empty(n)
    x[0] = rng.normal()
    noise = rng.normal(size=n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    exact = (1 + rho) / (1 - rho)
    tau = tau_int(x)
    sigma = exact * np.sqrt(2 * (2 * 5 * exact + 1) / n)  # Madras-Sokal error with window W ~ 5 tau
    print(f"[4] AR(1) rho = {rho}: tau_int = {tau:.3f}   exact = {exact:.3f}   (+- {sigma:.3f})")
    assert abs(tau - exact) < 4 * sigma


# 5. Global flip phi -> -phi. First, the premise: S is exactly even in phi, so dS = 0 and the move is always accepted.
phi = torch.randn(3, 6, 6)
assert torch.allclose(action(-phi, -4.5, 1.0), action(phi, -4.5, 1.0), atol=1e-12)
#    Then, in the broken phase: without the flip the chain is stuck (<M> far from 0); with it <M> = 0 within
#    errors. <|M|> and chi do not care which well the chain is in, so both runs must agree on them.
L, m2_b, lam_b = 16, -4.5, 1.0
stuck, _ = metropolis(L, m2_b, lam_b, n_sweeps=20000, n_therm=2000, step=1.5, rng=rng, flip=False)
mixed, _ = metropolis(L, m2_b, lam_b, n_sweeps=20000, n_therm=2000, step=1.5, rng=rng, flip=True)
M_stuck, M_mixed = magnetization(stuck), magnetization(mixed)
m_mixed, e_mixed = mean_err(M_mixed)
print(f"[5] broken phase L={L}, m2={m2_b}, lam={lam_b}:")
print(f"    <M>   local only = {M_stuck.mean():+.4f}      with flip = {m_mixed:+.4f} +- {e_mixed:.4f}")
print(f"    fraction of sweeps with M > 0:  local only = {np.mean(M_stuck > 0):.3f}   with flip = {np.mean(M_mixed > 0):.3f}")
assert abs(M_stuck.mean()) > 0.5            # the test has teeth: local-only chain really is stuck
assert abs(m_mixed) < 4 * e_mixed           # the flip restores <M> = 0
for name, f in (("<|M|>", np.abs), ("V M^2", lambda m: L * L * m**2)):
    (a, ea), (b, eb) = mean_err(f(M_stuck)), mean_err(f(M_mixed))
    zscore = (a - b) / np.hypot(ea, eb)
    print(f"    {name:6s} local only = {a:.4f} +- {ea:.4f}   with flip = {b:.4f} +- {eb:.4f}   z = {zscore:+.2f}")
    assert abs(zscore) < 4

# 6. correlator(): FFT version vs a brute-force double loop, C(r) = (1/V) sum_y phi_y phi_{y + r e_mu}.
L = 6
phi = rng.normal(size=(2, L, L))
brute = np.array([[np.mean([0.5 * (phi[n, i, j] * phi[n, (i + r) % L, j] + phi[n, i, j] * phi[n, i, (j + r) % L])
                            for i in range(L) for j in range(L)]) for r in range(L // 2 + 1)] for n in range(2)])
print(f"[6] correlator: max |FFT - brute force| = {np.max(np.abs(correlator(phi) - brute)):.2e}")
assert np.allclose(correlator(phi), brute, atol=1e-12)

# 7. jackknife(): for a linear function it must return the plain mean, and for iid data the error must
#    match std/sqrt(N). For a nonlinear function (a ratio) the estimate must be consistent with the exact value.
x = rng.normal(loc=2.0, size=100_000)
val, err = jackknife(lambda a: a, x)
print(f"[7] jackknife mean: {val:.5f} (np.mean {x.mean():.5f})   err {err:.5f} vs std/sqrt(N) {x.std() / np.sqrt(len(x)):.5f}")
assert abs(val - x.mean()) < 1e-12 and abs(err / (x.std() / np.sqrt(len(x))) - 1) < 0.35
y = rng.normal(loc=4.0, size=100_000)
val, err = jackknife(lambda a, b: b / a**2, x, y)  # exact: 4 / 2^2 = 1
print(f"    jackknife ratio <y>/<x>^2 = {val:.4f} +- {err:.4f}   (exact 1)")
assert abs(val - 1) < 4 * err

print("All step-1 checks passed.")
