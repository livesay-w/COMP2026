"""
Parallel tempering for the Edwards-Anderson model, vectorised over temperatures.

Plain Metropolis freezes below T_c: step 2 showed it failing on a *16-spin*
system at T=0.4.  Parallel tempering fixes this by running M copies at M fixed
temperatures at once and periodically swapping configurations between adjacent
rungs, accepted with

    P(swap) = min(1, exp[(beta_i - beta_j)(E_i - E_j)])

which preserves the product of all M Boltzmann distributions exactly.  A copy
trapped at low T is carried up the ladder, decorrelates where barriers are
irrelevant, and diffuses back down into a possibly different basin -- going
around barriers rather than over them.  This is an exact sampler, not a
heuristic optimiser.

Layout.  Spins live in one array of shape (2, M, L, ..., L):

    axis 0  the two independent replicas needed for the overlap q
    axis 1  the M temperature slots, s[:, i] always at temps[i]
    axis 2+ the lattice

Everything is updated in a single numpy call across both replicas and all
temperatures.  For these lattice sizes numpy's per-call overhead dominates the
arithmetic, so collapsing 2*M separate sweeps into one is worth more than any
other optimisation here -- roughly an order of magnitude.

Natural units: k_B = 1, <J^2> = 1.
"""

import numpy as np

from ea_model import local_fields  # noqa: F401  (single-replica version, step 0-2)


def make_ladder(t_min, t_max, n_temps):
    """Geometric temperature ladder, ascending.  Slot 0 is the coldest.

    Geometric spacing is the sensible starting point: swap acceptance depends
    on the overlap of adjacent energy histograms, whose width scales as
    sqrt(N), so constant *ratio* spacing keeps acceptance roughly uniform.
    Step 5's check is whether that actually happened.
    """
    return t_min * (t_max / t_min) ** (np.arange(n_temps) / (n_temps - 1))


def init_replicas(L, d, n_temps, rng):
    """Random start for both replicas at every temperature."""
    return (2 * rng.integers(0, 2, size=(2, n_temps) + (L,) * d) - 1).astype(np.int8)


def pt_local_fields(s, J):
    """h_i = sum_j J_ij s_j for every replica and temperature at once."""
    h = np.zeros(s.shape)
    for k in range(J.shape[0]):
        h += J[k] * np.roll(s, -1, axis=2 + k)
        h += np.roll(J[k], 1, axis=k) * np.roll(s, 1, axis=2 + k)
    return h


def pt_energies(s, J):
    """Total energy of each (replica, temperature), shape (2, M)."""
    d = J.shape[0]
    lat = tuple(range(2, 2 + d))
    e = np.zeros(s.shape[:2])
    for k in range(d):
        e -= np.sum(J[k] * s * np.roll(s, -1, axis=2 + k), axis=lat)
    return e


def pt_sweep(s, J, betas, rng, parity):
    """One checkerboard Metropolis sweep for all replicas and temperatures.

    Modifies s in place; returns the overall acceptance fraction.
    """
    b = betas.reshape((1, -1) + (1,) * (s.ndim - 2))
    accepted = 0
    for p in (0, 1):
        dE = 2.0 * s * pt_local_fields(s, J)
        # exp(-max(dE,0)*beta) is 1 where dE <= 0 and cannot overflow.
        accept = rng.random(s.shape) < np.exp(-np.maximum(dE, 0.0) * b)
        flip = accept & (parity == p)
        s[flip] *= -1
        accepted += int(flip.sum())
    return accepted / s.size


def pt_swap(s, J, betas, rng, labels, offset, n_try, n_acc):
    """Attempt swaps between adjacent temperature slots.

    Alternating even/odd pairs keeps every attempt independent.  Each of the
    two replicas swaps on its own, so they stay statistically independent --
    which is exactly what the overlap measurement requires.

    labels[r, i] identifies which walker currently occupies slot i, so that
    round trips through the ladder can be counted in Step 5.
    """
    e = pt_energies(s, J)
    for i in range(offset, s.shape[1] - 1, 2):
        delta = (betas[i] - betas[i + 1]) * (e[:, i] - e[:, i + 1])
        acc = rng.random(2) < np.exp(np.minimum(delta, 0.0))
        n_try[i] += 2
        n_acc[i] += int(acc.sum())
        for r in range(2):
            if acc[r]:
                s[r, [i, i + 1]] = s[r, [i + 1, i]]
                labels[r, [i, i + 1]] = labels[r, [i + 1, i]]
                e[r, [i, i + 1]] = e[r, [i + 1, i]]


def update_round_trips(labels, direction, trips):
    """Count cold -> hot -> cold traversals of the ladder.

    A walker reaching the coldest slot having last touched the hottest slot
    has completed a round trip.  This is the honest measure of whether the
    escape mechanism is working: a replica that never reaches the top of the
    ladder is still trapped, and parallel tempering has bought nothing for it.
    """
    top = labels.shape[1] - 1
    for r in range(labels.shape[0]):
        cold = labels[r, 0]
        if direction[r, cold] == -1:
            trips[r] += 1
        direction[r, cold] = 1
        direction[r, labels[r, top]] = -1


def make_phases(L, d):
    """exp(2 pi i x_k / L) for each lattice direction, shaped to broadcast."""
    out = []
    for k in range(d):
        shape = [1] * d
        shape[k] = L
        out.append(np.exp(2j * np.pi * np.arange(L) / L).reshape(shape))
    return out


def measure(s, J, phases):
    """Overlap observables at every temperature.

    Returns (e, q, chi0, chik, q_link), each of shape (M,) except e which is
    the mean energy per spin over both replicas.

    chi_SG(k) = <|sum_i q_i exp(i k.r_i)|^2> / N with q_i = s_i^(1) s_i^(2),
    averaged over the d equivalent directions.  chi(0) is just N q^2.  The
    ratio chi(0)/chi(k_min) gives the correlation length in step 7, and the
    averaging must be done on chi(0) and chi(k) *separately* before taking the
    ratio -- averaging the ratio itself is biased.
    """
    d = J.shape[0]
    a, b = s[0], s[1]
    n = a[0].size
    lat = tuple(range(1, 1 + d))

    q_field = (a * b).astype(np.float64)
    q = q_field.sum(axis=lat) / n
    chi0 = n * q ** 2

    chik = np.zeros(len(q))
    for ph in phases:
        chik += np.abs((q_field * ph).sum(axis=lat)) ** 2
    chik /= len(phases) * n

    # Link overlap: the two replicas' agreement bond by bond rather than site
    # by site.  For Gaussian couplings this is tied to the energy by an exact
    # identity (step 6), which is what makes it a thermalisation test.
    q_link = np.zeros(len(q))
    for k in range(d):
        q_link += (a * np.roll(a, -1, axis=1 + k)
                   * b * np.roll(b, -1, axis=1 + k)).sum(axis=lat)
    q_link /= d * n

    e = pt_energies(s, J).mean(axis=0) / n
    return e, q, chi0, chik, q_link


def binned_error(series, n_bins=20):
    """Standard error per temperature from bin means, series shape (n_rec, M).

    Bin means account for autocorrelation provided the bins are long compared
    with the correlation time; a naive std/sqrt(n) does not, and in a spin
    glass it underestimates badly.
    """
    n = (series.shape[0] // n_bins) * n_bins
    means = series[:n].reshape(n_bins, -1, series.shape[1]).mean(axis=1)
    return means.std(axis=0, ddof=1) / np.sqrt(n_bins)


def run_sample(J, L, d, temps, n_equil, n_meas, rng, stride=8, n_bins_q=101):
    """One disorder sample, start to finish.

    Equilibrate, then measure every `stride` sweeps.  Returns a dict of
    per-temperature results plus the diagnostics step 5 and 6 need.

    The P(q) histogram is accumulated here rather than the raw q series being
    returned, because the production run needs hundreds of samples and only
    the histogram has to survive.
    """
    n_temps = len(temps)
    betas = 1.0 / temps
    parity = (np.indices((L,) * d).sum(axis=0) % 2).astype(np.int8)
    assert L % 2 == 0, "checkerboard needs even L under periodic boundaries"
    phases = make_phases(L, d)

    s = init_replicas(L, d, n_temps, rng)
    labels = np.tile(np.arange(n_temps), (2, 1))
    direction = np.zeros((2, n_temps), dtype=np.int8)
    trips = np.zeros(2, dtype=np.int64)
    n_try = np.zeros(n_temps - 1)
    n_acc = np.zeros(n_temps - 1)

    for i in range(n_equil):
        pt_sweep(s, J, betas, rng, parity)
        pt_swap(s, J, betas, rng, labels, i % 2, n_try, n_acc)
        update_round_trips(labels, direction, trips)

    n_rec = n_meas // stride
    rec = {k: np.empty((n_rec, n_temps)) for k in
           ("e", "q", "q2", "q4", "chi0", "chik", "q_link")}
    hist = np.zeros((n_temps, n_bins_q))
    edges = np.linspace(-1.0, 1.0, n_bins_q + 1)

    for i in range(n_meas):
        pt_sweep(s, J, betas, rng, parity)
        pt_swap(s, J, betas, rng, labels, i % 2, n_try, n_acc)
        update_round_trips(labels, direction, trips)
        if i % stride == 0:
            j = i // stride
            e, q, chi0, chik, q_link = measure(s, J, phases)
            rec["e"][j], rec["q"][j] = e, q
            rec["q2"][j], rec["q4"][j] = q ** 2, q ** 4
            rec["chi0"][j], rec["chik"][j] = chi0, chik
            rec["q_link"][j] = q_link
            for t in range(n_temps):
                hist[t] += np.histogram(q[t], bins=edges)[0]

    out = {k: v.mean(axis=0) for k, v in rec.items()}
    for key in ("e", "q2", "q_link"):
        out[key + "_err"] = binned_error(rec[key])
    # Log-binning for the thermalisation test: the last half and the last
    # quarter must agree, or the run has not converged.
    half, quarter = n_rec // 2, 3 * n_rec // 4
    out["q2_half"] = rec["q2"][half:].mean(axis=0)
    out["q2_quarter"] = rec["q2"][quarter:].mean(axis=0)
    out["hist"] = hist
    out["swap_acc"] = n_acc / np.maximum(n_try, 1)
    out["trips"] = trips.copy()
    return out


def correlation_length(chi0, chik, L):
    """xi_L from the two lowest wavevectors.

    xi_L = sqrt(chi(0)/chi(k_min) - 1) / (2 sin(k_min/2)),  k_min = 2 pi / L.

    Above T_c the ratio can dip below 1 on noise alone, giving an imaginary
    length; those points are clipped to zero.  They sit far from the crossing
    that step 7 measures, so the clip cannot affect T_c -- but it is a clip,
    not a fit, and it is visible in the plots as xi_L/L = 0.
    """
    ratio = np.maximum(chi0 / chik - 1.0, 0.0)
    return np.sqrt(ratio) / (2.0 * np.sin(np.pi / L))
