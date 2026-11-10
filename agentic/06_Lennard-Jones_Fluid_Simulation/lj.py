"""
Lennard-Jones fluid in 3D -- Step 0: box, particles, and pair potential.

UNITS (reduced / "LJ units"), fixed throughout the project:
    epsilon = sigma = m = k_B = 1
    => energies are in units of epsilon, lengths in units of sigma,
       temperature T* = k_B T / epsilon, density rho* = N sigma^3 / V.
    No physical constants appear anywhere below. If you ever see a stray
    factor, it is a bug, not a unit conversion.

CONVENTIONS fixed in this file:
    - 3D cubic box of side L, fully periodic.
    - Cutoff r_c = 2.5, TRUNCATED AND SHIFTED:
          V_ts(r) = V_LJ(r) - V_LJ(r_c)   for r <  r_c
          V_ts(r) = 0                     for r >= r_c
      The shift makes the potential continuous at r_c, so a pair crossing
      the cutoff during an MC move causes no jump in the energy.
    - Minimum image convention, which is only valid while r_c < L/2.
"""

import numpy as np

# =====================================================================
# Section 1 -- The pair potential
# =====================================================================

R_CUT = 2.5
# Value of the raw LJ potential at the cutoff. Subtracting this is the "shift".
V_SHIFT = 4.0 * (R_CUT**-12 - R_CUT**-6)


def lj_raw(r):
    """Untruncated, unshifted Lennard-Jones potential 4(r^-12 - r^-6)."""
    inv_r6 = r**-6.0
    return 4.0 * (inv_r6 * inv_r6 - inv_r6)


def lj_potential(r):
    """Truncated-and-shifted LJ potential. Accepts a scalar or an array."""
    return np.where(r < R_CUT, lj_raw(r) - V_SHIFT, 0.0)


# =====================================================================
# Section 2 -- The periodic box
# =====================================================================

def box_length(N, rho):
    """Side of the cubic box holding N particles at number density rho."""
    return (N / rho) ** (1.0 / 3.0)


def check_box_is_legal(L):
    """Minimum image is only valid if the cutoff sphere fits in half the box."""
    assert R_CUT < L / 2.0, f"r_c={R_CUT} needs L>{2*R_CUT}, got L={L:.3f}"


def wrap(positions, L):
    """Fold coordinates back into [0, L)."""
    return positions % L


def min_image(dr, L):
    """
    Map displacement vectors to their nearest periodic image.
    Every returned component lies in [-L/2, L/2].
    """
    return dr - L * np.round(dr / L)


# =====================================================================
# Section 3 -- Particles: initial configuration
# =====================================================================

def fcc_lattice(N, L):
    """
    N particles on a face-centred cubic lattice filling the box.

    FCC (not simple cubic) because simple cubic is mechanically unstable at
    liquid densities. N must be 4*n^3: 4 basis atoms per cell, n cells a side.
    Returns an (N, 3) array.
    """
    n = round((N / 4) ** (1.0 / 3.0))
    assert 4 * n**3 == N, f"N={N} is not of the form 4n^3 (try 256, 500, 864)"

    a = L / n                      # conventional cubic cell edge
    basis = a * np.array([[0.0, 0.0, 0.0],
                          [0.5, 0.5, 0.0],
                          [0.5, 0.0, 0.5],
                          [0.0, 0.5, 0.5]])

    cells = a * np.array([[i, j, k]
                          for i in range(n)
                          for j in range(n)
                          for k in range(n)], dtype=float)

    # (n^3, 1, 3) + (1, 4, 3) -> (n^3, 4, 3) -> (N, 3)
    return (cells[:, None, :] + basis[None, :, :]).reshape(N, 3)


# =====================================================================
# Section 4 -- Energies
# =====================================================================

def pair_distances(positions, L):
    """Full (N, N) matrix of minimum-image pair distances. O(N^2)."""
    dr = min_image(positions[:, None, :] - positions[None, :, :], L)
    return np.sqrt((dr * dr).sum(axis=-1))


def total_energy(positions, L):
    """
    Total potential energy, summing each pair once. O(N^2).
    This is the slow REFERENCE implementation; the MC loop will use the
    O(N) single-particle version below and is checked against this one.
    """
    r = pair_distances(positions, L)
    i, j = np.triu_indices(len(positions), k=1)
    return lj_potential(r[i, j]).sum()


def particle_energy(i, positions, L):
    """Interaction energy of particle i with every other particle. O(N)."""
    dr = min_image(positions - positions[i], L)
    r = np.sqrt((dr * dr).sum(axis=-1))
    r[i] = np.inf          # exclude self-interaction (and avoid 1/0)
    return lj_potential(r).sum()


# =====================================================================
# Section 5 -- The Metropolis algorithm
#
# WHAT THIS DOES, AND WHAT IT DOES *NOT* DO
#
#   We are NOT minimizing the energy. Metropolis Monte Carlo samples
#   configurations from the Boltzmann distribution P(x) ~ exp(-E(x)/T).
#   At T* > 0 the system does not go to the lowest-energy configuration;
#   it goes to the one that balances low energy against high entropy
#   (there are vastly more disordered arrangements than ordered ones).
#
#   Concretely: we start on the FCC lattice at E/N = -5.92, which is near
#   the energy MINIMUM. Running at T* = 1.5 makes the energy go *UP* and
#   settle at some higher value. That rise is the lattice melting into a
#   fluid, and it is the correct behaviour. If the energy fell below the
#   lattice value, something would be wrong.
#
#   What we measure at the end is the *average* energy over many samples
#   once the system has stopped drifting -- a thermal average, not a
#   minimum.
#
# THE PARAMETERS
#
#   T     reduced temperature T* = k_B T / epsilon. This is the only place
#         temperature enters. It sets how willing we are to accept moves
#         that raise the energy: the acceptance weight is exp(-dE/T), so
#         large T accepts nearly everything (gas-like, disordered) and
#         small T accepts almost nothing uphill (frozen, lattice-like).
#
#   delta maximum trial displacement per coordinate, in units of sigma.
#         This is a purely numerical knob -- it does NOT change the
#         physics being sampled, only how efficiently we explore.
#           delta too small -> almost every move accepted, but the
#                              configuration barely changes: slow.
#           delta too large -> trial moves land on top of neighbours,
#                              dE is huge, almost everything rejected:
#                              also slow.
#         We tune delta for roughly 30-50% acceptance, which is the usual
#         rule of thumb for the best compromise.
#
#   rng   seeded generator, so runs are reproducible for grading.
#
# WHY THE MOVE IS CHEAP
#
#   Moving one particle only changes that particle's interactions, so
#   dE = E_i(new) - E_i(old) using the O(N) particle_energy(). We never
#   recompute the O(N^2) total inside the loop. Section 6 checks that this
#   shortcut stays exact.
# =====================================================================

def metropolis_sweep(positions, L, T, delta, rng):
    """
    One sweep = N attempted single-particle moves, so on average every
    particle is offered a move once. Sweeps (not individual moves) are the
    natural time unit, because they are independent of system size.

    MODIFIES positions IN PLACE.
    Returns (total energy change over the sweep, fraction of moves accepted).
    """
    N = len(positions)
    dE_sweep = 0.0
    n_accepted = 0

    for _ in range(N):
        i = rng.integers(N)                       # pick a particle at random

        old_position = positions[i].copy()        # keep it, so we can undo
        e_old = particle_energy(i, positions, L)

        # Trial move: uniform in a small cube of side 2*delta around the
        # particle. Symmetric (the reverse move is equally likely), which is
        # what lets us use the simple exp(-dE/T) acceptance rule.
        trial = old_position + rng.uniform(-delta, delta, size=3)
        positions[i] = wrap(trial, L)             # may cross a box face

        e_new = particle_energy(i, positions, L)
        dE = e_new - e_old

        # Metropolis acceptance: always accept downhill (dE <= 0), accept
        # uphill with probability exp(-dE/T). Testing dE <= 0 first also
        # avoids overflow in exp() for large negative dE.
        if dE <= 0.0 or rng.random() < np.exp(-dE / T):
            dE_sweep += dE
            n_accepted += 1
        else:
            positions[i] = old_position           # reject: put it back

    return dE_sweep, n_accepted / N


def run_sweeps(positions, L, T, delta, n_sweeps, rng):
    """
    Run n_sweeps sweeps, tracking the energy by accumulating dE only.

    Returns (energy trace of length n_sweeps+1, acceptance fraction per sweep).
    The trace starts with the initial energy, computed once with the O(N^2)
    routine; every later entry is the previous one plus that sweep's dE.
    """
    energy = total_energy(positions, L)
    trace = [energy]
    acceptance = []

    for _ in range(n_sweeps):
        dE, acc = metropolis_sweep(positions, L, T, delta, rng)
        energy += dE
        trace.append(energy)
        acceptance.append(acc)

    return np.array(trace), np.array(acceptance)


# =====================================================================
# Section 6 -- Consistency checks
#   Each one is self-contained and would genuinely fail if the code above
#   were wrong. Run this file directly to execute them.
# =====================================================================

def check_potential_landmarks():
    """The LJ potential has three exactly known features."""
    # Raw potential crosses zero at r = sigma = 1.
    assert abs(lj_raw(1.0)) < 1e-12

    # Raw minimum is exactly -1 at r = 2^(1/6).
    r_min = 2.0 ** (1.0 / 6.0)
    assert abs(lj_raw(r_min) + 1.0) < 1e-12

    # The shift does not move the minimum, only lowers the whole curve.
    assert abs(lj_potential(r_min) - (-1.0 - V_SHIFT)) < 1e-12

    # Shifted potential is continuous at the cutoff: V -> 0 from inside.
    assert abs(lj_potential(R_CUT - 1e-9)) < 1e-7
    assert lj_potential(R_CUT + 1e-9) == 0.0

    # Independent numerical confirmation of the minimum location.
    r = np.linspace(0.9, 2.0, 200001)
    assert abs(r[np.argmin(lj_potential(r))] - r_min) < 1e-4
    print(f"  potential landmarks ok   (V_shift = {V_SHIFT:+.6f})")


def check_min_image_against_brute_force(rng):
    """
    Minimum image must agree with an explicit search over all 27 neighbouring
    periodic images. This is the check that catches rounding/wrapping bugs.
    """
    L = 7.3
    a = wrap(rng.random((200, 3)) * L, L)
    b = wrap(rng.random((200, 3)) * L, L)

    fast = np.sqrt((min_image(a - b, L) ** 2).sum(axis=-1))

    offsets = L * np.array([[i, j, k]
                            for i in (-1, 0, 1)
                            for j in (-1, 0, 1)
                            for k in (-1, 0, 1)], dtype=float)
    # (200, 27, 3) -> distance to every image -> take the smallest
    brute = np.sqrt((((a - b)[:, None, :] + offsets[None, :, :]) ** 2).sum(-1)).min(axis=1)

    assert np.allclose(fast, brute, atol=1e-12)
    print(f"  min image == 27-image brute force   (max diff {np.abs(fast-brute).max():.1e})")


def check_min_image_bounds(rng):
    """No minimum-image component may exceed half the box."""
    L = 7.3
    dr = min_image(rng.random((500, 3)) * 10 * L - 5 * L, L)
    assert np.all(np.abs(dr) <= L / 2.0 + 1e-12)
    print("  all min-image components within [-L/2, L/2]")


def check_translation_invariance(positions, L, rng):
    """
    A periodic system has no origin: rigidly shifting every particle (and
    re-wrapping) must leave the total energy exactly unchanged.
    """
    e0 = total_energy(positions, L)
    shifted = wrap(positions + rng.random(3) * L, L)
    e1 = total_energy(shifted, L)
    assert abs(e0 - e1) < 1e-10 * max(1.0, abs(e0))
    print(f"  energy invariant under rigid shift   ({e0:.10f} vs {e1:.10f})")


def check_energy_decomposition(positions, L):
    """
    Summing the O(N) per-particle energies double counts every pair, so half
    that sum must equal the O(N^2) total. This validates the fast path the
    Metropolis loop will rely on.
    """
    e_total = total_energy(positions, L)
    e_sum = 0.5 * sum(particle_energy(i, positions, L) for i in range(len(positions)))
    assert abs(e_total - e_sum) < 1e-10 * max(1.0, abs(e_total))
    print(f"  0.5 * sum_i E_i == E_total   ({e_sum:.10f} vs {e_total:.10f})")


def check_lattice(positions, N, L, rho):
    """The FCC configuration must have the requested density and no overlaps."""
    assert positions.shape == (N, 3)
    assert np.all(positions >= 0) and np.all(positions < L)
    assert abs(N / L**3 - rho) < 1e-12

    n = round((N / 4) ** (1.0 / 3.0))
    a = L / n
    r = pair_distances(positions, L)
    r[np.diag_indices(N)] = np.inf
    nearest = r.min()
    # Nearest-neighbour separation in an FCC lattice is a/sqrt(2).
    assert abs(nearest - a / np.sqrt(2)) < 1e-10
    print(f"  FCC lattice ok   (a = {a:.4f}, nearest neighbour = {nearest:.4f} = a/sqrt2)")


def check_incremental_energy_is_exact(N, L, rng, n_sweeps=40, T=1.5, delta=0.15):
    """
    THE check for the whole project.

    During a run we never recompute the total energy -- we just add up the
    small dE of every accepted move. If those increments drifted even
    slightly from the truth, every energy, pressure and phase conclusion
    downstream would be quietly wrong, with nothing crashing to warn us.

    So: run a real MC trajectory tracking energy by increments only, then
    recompute the total from scratch with the O(N^2) routine and demand
    agreement to machine precision.
    """
    pos = fcc_lattice(N, L)
    trace, _ = run_sweeps(pos, L, T, delta, n_sweeps, rng)

    tracked = trace[-1]
    recomputed = total_energy(pos, L)
    err = abs(tracked - recomputed)

    assert err < 1e-8, f"incremental energy drifted by {err:.3e}"
    print(f"  incremental dE == full recompute after {n_sweeps} sweeps "
          f"({N*n_sweeps} moves):  error {err:.2e}")


def check_cold_runs_never_go_uphill(N, L, rng):
    """
    T -> 0 limit. At vanishing temperature exp(-dE/T) underflows for any
    uphill move, so Metropolis becomes pure downhill search and the energy
    must be monotonically non-increasing. A sign error in dE or in the
    acceptance test would break this immediately.
    """
    pos = fcc_lattice(N, L)
    trace, _ = run_sweeps(pos, L, T=1e-6, delta=0.1, n_sweeps=10, rng=rng)
    assert np.all(np.diff(trace) <= 1e-12), "energy rose at T -> 0"
    print(f"  T -> 0 run is monotonically downhill "
          f"({trace[0]/N:+.4f} -> {trace[-1]/N:+.4f} per particle)")


def check_acceptance_falls_with_delta(N, L, rng):
    """
    delta is the step size, and acceptance must decrease as it grows:
    tiny steps barely change the energy and are almost always accepted,
    huge steps drop particles onto their neighbours and are rejected.
    Also pins the two limits: delta -> 0 gives acceptance -> 1.
    """
    deltas = [0.01, 0.05, 0.2, 0.5, 1.5]
    rates = []
    for d in deltas:
        pos = fcc_lattice(N, L)
        _, acc = run_sweeps(pos, L, T=1.5, delta=d, n_sweeps=5, rng=rng)
        rates.append(acc.mean())

    assert np.all(np.diff(rates) < 0), f"acceptance not monotonic: {rates}"
    assert rates[0] > 0.95, f"tiny delta should accept nearly everything, got {rates[0]:.3f}"
    print("  acceptance falls monotonically with delta:  " +
          ", ".join(f"d={d}:{r:.0%}" for d, r in zip(deltas, rates)))


def check_hot_run_melts_the_lattice(N, L, rng):
    """
    A positive-temperature run must raise the energy above the lattice
    value -- we are sampling a thermal distribution, not minimizing.
    This is the check that would catch an inverted acceptance criterion
    (which would look superficially fine but drive the system to T = 0).
    """
    pos = fcc_lattice(N, L)
    e_lattice = total_energy(pos, L)
    trace, _ = run_sweeps(pos, L, T=1.5, delta=0.15, n_sweeps=60, rng=rng)
    assert trace[-1] > e_lattice, "T>0 run did not raise the energy"
    print(f"  T* = 1.5 run heats the lattice "
          f"({e_lattice/N:+.4f} -> {trace[-1]/N:+.4f} per particle)")


# =====================================================================
# Section 7 -- Where this goes next (notes, not code)
#
# Two directions the professor pointed at. Writing them down here so the
# next version of this file knows what it is being built toward.
#
# ---------------------------------------------------------------------
# A. BAYESIAN INFERENCE AND SIMULATION-BASED INFERENCE (SBI)
# ---------------------------------------------------------------------
#
#   Everything above solves the FORWARD problem: given (T*, rho*), produce
#   configurations and hence <E>/N, g(r), P. The inverse problem is the
#   one inference answers: given a measured observable, which (T*, rho*)
#   produced it, and how certain are we?
#
#   Bayesian inference answers that with a distribution rather than a fit:
#
#       P(theta | data)  ~  P(data | theta) * P(theta)
#        posterior          likelihood        prior
#
#   The posterior's WIDTH is the error bar, and its SHAPE carries the
#   degeneracies -- e.g. a single measured E/N does not pin down T* and
#   rho* separately, it only constrains a curve through the (T*, rho*)
#   plane, and the posterior shows that as a banana instead of a blob.
#
#   The obstacle: we have no tractable likelihood. We can SAMPLE from the
#   simulator at any theta, but we cannot write down the probability
#   density of g(r) given theta. The usual dodge is to crush the data into
#   one number, assume Gaussian errors, and chi-square it -- which throws
#   away most of the information in g(r).
#
#   SBI (a.k.a. likelihood-free inference) is Bayesian inference that only
#   ever calls the simulator:
#     - Draw theta_k from the prior, run the MC, record (theta_k, x_k).
#     - Train a density estimator (normalizing flow / neural net) on those
#       pairs to approximate either the posterior p(theta|x) directly
#       (neural posterior estimation), the likelihood p(x|theta), or just
#       the likelihood RATIO (neural ratio estimation).
#     - Evaluate the trained estimator on the one x we actually measured.
#     ABC is the primitive ancestor: keep every theta whose simulated x
#     landed within epsilon of the observed x. Correct, but the acceptance
#     rate collapses as the dimension of x grows, which is why the neural
#     versions exist.
#
#   Concretely for this project: theta = (T*, rho*), x = binned g(r) (or
#   (E/N, P)). A few thousand short runs over the prior is the training
#   set; this simulator is cheap enough for that.
#
#   CONSISTENCY CHECK for that step -- COVERAGE. Hold out simulations whose
#   true theta we know, build the 90% credible region from the posterior
#   for each, and count how often the truth falls inside. It must be ~90%.
#   Over-confident posteriors are THE standard SBI failure mode and this
#   test catches them; nothing else does, because a wrong posterior still
#   looks like a perfectly nice contour plot.
#
# ---------------------------------------------------------------------
# B. DO ERROR BARS REALLY FOLLOW A POWER LAW IN N?
# ---------------------------------------------------------------------
#
#   Short answer: -1/2 is the generic case, not a law. It is the NULL
#   HYPOTHESIS -- what you get when you average many weakly-correlated
#   contributions each with finite variance -- and the useful move is to
#   measure the exponent and treat a deviation as a signal.
#
#   First, two different N's, and conflating them is the trap:
#
#     n = number of samples (sweeps kept).  Error on a mean ~ n^(-1/2).
#         Pure central limit theorem.
#
#     N = number of particles.  For an INTENSIVE quantity (E/N), the
#         fluctuation width ~ N^(-1/2), again CLT, now over quasi-
#         independent subvolumes of the box. For an EXTENSIVE quantity
#         (total E), the absolute fluctuation GROWS like N^(+1/2) while
#         the relative one shrinks like N^(-1/2). Same exponent, opposite
#         sign, so "error bars shrink with N" is only true if you say
#         which quantity.
#
#   Where the -1/2 genuinely breaks:
#
#     1. CORRELATED SAMPLES (always true here). MC sweeps are not
#        independent, so
#              sigma_mean = sigma * sqrt(2 * tau_int / n).
#        Still n^(-1/2), but ignoring tau_int underestimates the error by
#        sqrt(2 tau), routinely a factor of a few. The fix is a blocking
#        analysis: bin the trace into blocks, plot the block-mean error vs
#        block size, and read off the plateau. The plateau value is the
#        honest error bar, and reaching a plateau at all is the evidence
#        that the samples have decorrelated.
#
#     2. NEAR A PHASE TRANSITION. The correlation length grows toward the
#        box size, so the subvolumes stop being independent and the
#        N^(-1/2) argument loses its premise. Fluctuation observables
#        (C_v, compressibility) then follow finite-size-scaling exponents
#        set by the universality class, not by counting, and tau_int
#        itself diverges (critical slowing down). LJ has a liquid-gas
#        critical point, so pushing toward it SHOULD break the -1/2 slope.
#        That break is physics, not a bug -- which is why measuring the
#        exponent is worth doing.
#
#     3. HEAVY TAILS. If the sampled quantity has a power-law tail heavy
#        enough that its variance diverges, the CLT is replaced by a
#        stable (Levy) limit and the error falls more slowly than n^(-1/2).
#        Unlikely for E/N with a truncated potential, but a live risk for
#        anything built on r^(-13), where one close pair dominates.
#
#     4. SYSTEMATIC ERROR DOESN'T SHRINK AT ALL. Cutoff truncation,
#        finite-box effects, leftover equilibration bias -- all constant in
#        n. Once the statistical error drops below them the log-log slope
#        flattens to 0 and more sampling buys nothing. Seeing that floor
#        is useful: it says where to stop running and start fixing physics.
#
#   So, to answer "is this fluctuation just statistical noise": plot the
#   error vs n on log-log (and separately vs N).
#        slope ~ -1/2  -> statistical noise, behaving;
#        shallower     -> correlated, unequilibrated, or hitting a
#                         systematic floor;
#        steeper/odd   -> suspect the estimator before celebrating.
#
#   This ties back to A: an SBI posterior is only as honest as the
#   simulations fed to it. Correlated training runs give an over-confident
#   posterior, and the coverage test above is what exposes it.
# =====================================================================


if __name__ == "__main__":
    N, RHO = 256, 0.8
    L = box_length(N, RHO)
    check_box_is_legal(L)
    rng = np.random.default_rng(0)
    pos = fcc_lattice(N, L)

    print(f"N = {N}, rho* = {RHO}, L = {L:.4f}, r_c = {R_CUT}  (L/2 = {L/2:.4f})\n")
    print("Consistency checks:")
    check_potential_landmarks()
    check_min_image_against_brute_force(rng)
    check_min_image_bounds(rng)
    check_lattice(pos, N, L, RHO)
    check_translation_invariance(pos, L, rng)
    check_energy_decomposition(pos, L)
    print(f"  FCC lattice energy per particle: {total_energy(pos, L) / N:+.6f}")

    print("\nMetropolis checks:")
    check_incremental_energy_is_exact(N, L, rng)
    check_cold_runs_never_go_uphill(N, L, rng)
    check_acceptance_falls_with_delta(N, L, rng)
    check_hot_run_melts_the_lattice(N, L, rng)
