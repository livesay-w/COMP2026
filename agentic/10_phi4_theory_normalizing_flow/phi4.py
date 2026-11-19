"""Lattice phi^4 theory in 2D: action, local Metropolis, observables, exact free-theory results.

Lattice units (a = 1). Periodic L x L lattice with L even (needed for the checkerboard).

    S[phi] = sum_x [ 1/2 sum_mu (phi_{x+mu} - phi_x)^2 + 1/2 m2 phi_x^2 + lam phi_x^4 ]

Target distribution: p(phi) = exp(-S[phi]) / Z.
The torch action is used by the flow; the numpy code is the independent Metropolis baseline.
"""
import numpy as np


# ---------------------------------------------------------------- action (torch)

def action(phi, m2, lam):
    """Action of each configuration. phi: tensor (B, L, L) -> tensor (B,)."""
    # torch is imported here, not at module level, so numpy-only users (the parallel Metropolis workers in
    # scan.py) don't each load it: 12 workers x torch exhausted the 7 GB of RAM and killed the 6b scan.
    import torch
    kinetic = 0.5 * sum((torch.roll(phi, -1, dims=d) - phi) ** 2 for d in (1, 2))
    potential = 0.5 * m2 * phi**2 + lam * phi**4
    return (kinetic + potential).sum(dim=(1, 2))


# ---------------------------------------------------------------- local Metropolis (numpy)

def neighbor_sum(phi):
    """Sum of the 4 nearest neighbours of every site. phi: array (L, L)."""
    return sum(np.roll(phi, shift, axis=ax) for ax in (0, 1) for shift in (1, -1))


def local_action(phi_x, nbr, m2, lam):
    """Every term of S that involves site x, as a function of its value phi_x.

    Site x sits in 4 links, each contributing 1/2 (phi_x - phi_y)^2 -> 2 phi_x^2 - phi_x * sum_nn phi_y.
    """
    return (2 + 0.5 * m2) * phi_x**2 + lam * phi_x**4 - phi_x * nbr


def checkerboard_np(L, parity):
    i, j = np.indices((L, L))
    return (i + j) % 2 == parity


def metropolis(L, m2, lam, n_sweeps, n_therm, step, rng, flip):
    """Local Metropolis, vectorised over a checkerboard.

    Sites of one colour only couple to sites of the other colour, so all black sites
    can be updated at once given the white ones (and vice versa). One sweep = black + white.

    flip=True adds a global move phi -> -phi with probability 1/2 after every sweep. S is even in phi,
    so dS = 0 and the move is always accepted; it satisfies detailed balance and lets the chain jump
    between the two wells of the broken phase, which local updates almost never cross.

    Returns (configs of shape (n_sweeps, L, L), acceptance rate of the local updates).
    """
    assert L % 2 == 0, "checkerboard needs even L with periodic boundaries"
    phi = rng.normal(size=(L, L))
    masks = [checkerboard_np(L, 0), checkerboard_np(L, 1)]
    configs = np.empty((n_sweeps, L, L))
    n_accept = 0
    for sweep in range(n_therm + n_sweeps):
        phi, n_acc = sweep_once(phi, masks, m2, lam, step, rng, flip)
        if sweep >= n_therm:
            n_accept += n_acc
            configs[sweep - n_therm] = phi
    return configs, n_accept / (n_sweeps * L * L)


def sweep_once(phi, masks, m2, lam, step, rng, flip):
    """One checkerboard sweep (black, then white), then the optional global flip. Returns (phi, n_accepted)."""
    n_accept = 0
    for mask in masks:
        nbr = neighbor_sum(phi)
        proposal = phi + step * rng.uniform(-1, 1, size=phi.shape)
        dS = local_action(proposal, nbr, m2, lam) - local_action(phi, nbr, m2, lam)
        accept = mask & (np.log(rng.random(phi.shape)) < -dS)
        phi = np.where(accept, proposal, phi)
        n_accept += accept.sum()
    if flip and rng.random() < 0.5:
        phi = -phi
    return phi, n_accept


def metropolis_magnetization(L, m2, lam, n_sweeps, n_therm, step, rng, flip):
    """Same chain as metropolis(), but keeps only M per sweep (memory O(n_sweeps), not O(n_sweeps L^2)).

    Returns (M of shape (n_sweeps,), final configuration (L, L), acceptance rate of the local updates).
    With the same rng state it reproduces magnetization(metropolis(...)[0]) exactly (check_scan.py).
    """
    assert L % 2 == 0, "checkerboard needs even L with periodic boundaries"
    phi = rng.normal(size=(L, L))
    masks = [checkerboard_np(L, 0), checkerboard_np(L, 1)]
    M = np.empty(n_sweeps)
    n_accept = 0
    for sweep in range(n_therm + n_sweeps):
        phi, n_acc = sweep_once(phi, masks, m2, lam, step, rng, flip)
        if sweep >= n_therm:
            n_accept += n_acc
            M[sweep - n_therm] = phi.mean()
    return M, phi, n_accept / (n_sweeps * L * L)


# ---------------------------------------------------------------- observables (numpy, configs (N, L, L))

def magnetization(phi):
    """Site-averaged field M of each configuration."""
    return phi.mean(axis=(-2, -1))


def susceptibility(phi):
    """chi = V <M^2>. (<M> = 0 on a finite lattice by the phi -> -phi symmetry.)"""
    V = phi.shape[-1] * phi.shape[-2]
    return V * np.mean(magnetization(phi) ** 2)


def binder(phi):
    """Binder cumulant U4 = 1 - <M^4> / (3 <M^2>^2). Curves for different L cross at criticality."""
    M = magnetization(phi)
    return 1 - np.mean(M**4) / (3 * np.mean(M**2) ** 2)


def momentum_power(phi):
    """|phi_k|^2 / V for each configuration, phi_k = sum_x e^{-ikx} phi_x. Shape (N, L, L).

    Its average is the momentum-space propagator; in the free theory it equals 1 / (m2 + khat^2).
    """
    V = phi.shape[-1] * phi.shape[-2]
    return np.abs(np.fft.fft2(phi)) ** 2 / V


def two_point(phi):
    """Connected two-point function G(x) = <phi_y phi_{y+x}> - <M>^2, averaged over y. Shape (L, L)."""
    G = np.fft.ifft2(momentum_power(phi).mean(axis=0)).real
    return G - np.mean(magnetization(phi)) ** 2


def tau_int(x, c=5):
    """Integrated autocorrelation time of a 1D series, tau = 1 + 2 sum_{t>=1} rho(t).

    Convention: tau = 1 for independent samples, so the effective sample count is N / tau.
    The sum is cut at the first window W >= c * tau(W) (Sokal's automatic windowing), since
    rho(t) at large t is pure noise.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)  # zero-padded FFT -> linear (not circular) autocorrelation
    acf = np.fft.irfft(f * np.conj(f))[:n] / np.arange(n, 0, -1)
    rho = acf / acf[0]
    taus = 1 + 2 * np.cumsum(rho[1:])
    W = np.argmax(np.arange(1, n) >= c * taus)
    assert W > 0, "series too short to estimate tau_int"
    return taus[W]


def mean_err(x):
    """Mean and standard error of a Markov-chain series, inflated by sqrt(tau_int)."""
    x = np.asarray(x, dtype=float)
    return x.mean(), x.std(ddof=1) * np.sqrt(tau_int(x) / len(x))


def correlator(phi):
    """Per-configuration correlator C(r) = (1/V) sum_y phi_y phi_{y + r e_mu}, averaged over both axes.

    phi: (N, L, L) -> (N, L//2 + 1) for r = 0 .. L/2. In the symmetric phase <M> = 0, so <C(r)> is the
    connected two-point function.
    """
    V = phi.shape[-1] * phi.shape[-2]
    c = np.fft.ifft2(np.abs(np.fft.fft2(phi)) ** 2).real / V
    r = np.arange(phi.shape[-1] // 2 + 1)
    return 0.5 * (c[:, r, 0] + c[:, 0, r])


def jackknife(func, *series, n_blocks=50):
    """Value and error of func(<a>, <b>, ...) from Markov-chain series a, b, ... (axis 0 = time).

    Blocked jackknife: blocks absorb autocorrelation (they must be much longer than tau_int), and
    the leave-one-block-out estimates handle non-linear func such as the Binder ratio.
    """
    blocks = [np.array([b.mean(axis=0) for b in np.array_split(s, n_blocks)]) for s in series]
    value = func(*[b.mean(axis=0) for b in blocks])
    loo = np.array([func(*[(b.sum(axis=0) - b[i]) / (n_blocks - 1) for b in blocks]) for i in range(n_blocks)])
    err = np.sqrt((n_blocks - 1) * np.mean((loo - loo.mean(axis=0)) ** 2, axis=0))
    return value, err


def binned_mean_err(x, n_bins=20):
    """Mean and standard error of a Markov-chain series (axis 0), using n_bins bins to absorb autocorrelation."""
    bins = np.array_split(x, n_bins, axis=0)
    means = np.array([b.mean(axis=0) for b in bins])
    return means.mean(axis=0), means.std(axis=0, ddof=1) / np.sqrt(n_bins)


# ---------------------------------------------------------------- exact free theory (lam = 0, m2 > 0)

def khat2(L):
    """Lattice momentum squared, sum_mu 4 sin^2(k_mu / 2), in np.fft.fft2 ordering. Shape (L, L)."""
    k = 2 * np.pi * np.fft.fftfreq(L)
    s2 = 4 * np.sin(k / 2) ** 2
    return s2[:, None] + s2[None, :]


def exact_propagator(L, m2):
    return 1 / (m2 + khat2(L))


def exact_logZ(L, m2):
    """log Z for S = 1/2 phi^T K phi with eigenvalues m2 + khat^2:  (V/2) log 2 pi - 1/2 sum_k log(m2 + khat^2)."""
    return 0.5 * L * L * np.log(2 * np.pi) - 0.5 * np.sum(np.log(m2 + khat2(L)))
