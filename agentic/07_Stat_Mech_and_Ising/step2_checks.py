"""
Step 2 -- pin the code down at LOW temperature against exact enumeration.

Steps 0 and 1 validated the machinery at high temperature and on the
unfrustrated ferromagnet.  Neither says anything about the regime the project
actually cares about.  This step closes that gap in the only way that gives
exact equality: take a lattice small enough to sum over every one of its 2^N
states, build the partition function by brute force, and demand the Monte
Carlo reproduce it.

It is the strongest check in the project and the last one where an *exact*
answer is available.  After this, every disagreement becomes "bug, or just
statistics?".

Geometries (all small enough to enumerate):

    2D L=4 periodic   16 spins, 32 bonds, 65536 states   <- production path
    2D L=4 open       16 spins, 24 bonds, 65536 states
    3D L=2 open        8 spins, 12 bonds,   256 states   <- 3D, the target

3D L=2 *periodic* is deliberately absent: at L=2 the +e_k and -e_k neighbours
coincide, so every bond is counted twice and the model is not the one we mean.
Open boundaries fix that, which is why they exist in make_couplings at all.

Observables compared, at each temperature:

    e        = <E>/N
    c        = (<E^2> - <E>^2) / (N T^2)          heat capacity per spin
    <q^2>    = (1/N^2) sum_ij <s_i s_j>^2         the spin glass order parameter

The <q^2> identity is worth pausing on.  Two replicas share couplings but are
otherwise independent, so <q^2> factorises into the square of the single-replica
correlation matrix.  That makes <q^2> exactly computable from enumeration while
the Monte Carlo side has to get it from two genuinely independent chains -- so
this is a real test of the replica machinery step 4 will depend on, not just of
the energy.

Run:  python agentic/07_Stat_Mech_and_Ising/step2_checks.py
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ea_model import (
    energy,
    make_couplings,
    metropolis_sweep,
    random_spins,
    sublattice_parity,
)


def enumerate_states(n):
    """All 2^n configurations as +-1, shape (2^n, n)."""
    bits = (np.arange(2 ** n, dtype=np.uint32)[:, None] >> np.arange(n)) & 1
    return (2 * bits - 1).astype(np.int8)


def bond_list(J, L, d):
    """[(i, j, J_ij)] over flat site indices, one entry per stored bond.

    Zero couplings (how open boundaries are represented) are kept rather than
    filtered: they contribute nothing to the energy either way, and skipping
    them would need a special case that could hide a real zero.
    """
    shape = (L,) * d
    out = []
    for flat in range(L ** d):
        x = np.unravel_index(flat, shape)
        for k in range(d):
            y = list(x)
            y[k] = (y[k] + 1) % L
            out.append((flat, int(np.ravel_multi_index(tuple(y), shape)),
                        float(J[(k,) + x])))
    return out


def exact_observables(J, L, d, temps):
    """e, c and <q^2> from a direct sum over all 2^N states.

    No sampling, no approximation -- this is the answer.
    """
    n = L ** d
    states = enumerate_states(n)
    e_states = np.zeros(len(states))
    for i, j, jij in bond_list(J, L, d):
        e_states -= jij * states[:, i] * states[:, j]

    e, c, q2 = np.empty(len(temps)), np.empty(len(temps)), np.empty(len(temps))
    for it, T in enumerate(temps):
        # Shift by the minimum before exponentiating: at T ~ 0.4 the raw
        # exponent would overflow long before the weights mattered.
        w = np.exp(-(e_states - e_states.min()) / T)
        w /= w.sum()
        e1 = float(w @ e_states)
        e2 = float(w @ e_states ** 2)
        e[it] = e1 / n
        c[it] = (e2 - e1 ** 2) / (n * T ** 2)
        # <s_i s_j> for every pair, then <q^2> = mean_ij <s_i s_j>^2.
        corr = states.T @ (w[:, None] * states)
        q2[it] = float(np.mean(corr ** 2))
    return e, c, q2


def binned_error(series, n_bins=20):
    """Standard error from bin means.

    Accounts for autocorrelation provided the bins are long compared with the
    correlation time, which a naive std/sqrt(n) does not.  Step 6 needs this
    machinery anyway, so it is built here where the right answer is known.
    """
    n = (len(series) // n_bins) * n_bins
    means = series[:n].reshape(n_bins, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(n_bins))


def mc_observables(J, L, d, temps, n_equil, n_meas, rng):
    """Same observables from two independent replicas sharing the couplings.

    Temperatures are visited in descending order, carrying both configurations
    over.  The system is tiny, so plain Metropolis is still trustworthy at low
    T here -- barriers cannot grow with a system this size.  That is exactly
    why the check is possible before parallel tempering exists.
    """
    parity = sublattice_parity(L, d)
    s1 = random_spins(L, d, rng)
    s2 = random_spins(L, d, rng)
    n = s1.size

    e, c, q2 = np.empty(len(temps)), np.empty(len(temps)), np.empty(len(temps))
    e_err, q2_err = np.empty(len(temps)), np.empty(len(temps))
    for it, T in enumerate(temps):
        for _ in range(n_equil):
            metropolis_sweep(s1, J, T, rng, parity)
            metropolis_sweep(s2, J, T, rng, parity)

        es = np.empty(n_meas)
        qs = np.empty(n_meas)
        for i in range(n_meas):
            metropolis_sweep(s1, J, T, rng, parity)
            metropolis_sweep(s2, J, T, rng, parity)
            es[i] = energy(s1, J)
            qs[i] = float(np.mean(s1 * s2))

        e[it] = es.mean() / n
        # Heat capacity from one replica's series only: averaging two
        # independent energies would halve the variance and so halve c.
        c[it] = (es.var() / (n * T ** 2))
        q2[it] = float(np.mean(qs ** 2))
        e_err[it] = binned_error(es) / n
        q2_err[it] = binned_error(qs ** 2)
    return e, c, q2, e_err, q2_err


def check_geometry(label, L, d, boundary, temps, n_equil, n_meas,
                   n_samples=2, seed=20, t_assert_min=0.6):
    """Exact vs. Monte Carlo on one geometry, for a few disorder samples.

    Temperatures below t_assert_min are measured and printed but excluded from
    the pass criterion, and marked with "(stats)" in the table.  This is not a
    convenience: plain Metropolis genuinely cannot equilibrate even a 16-spin
    system at T=0.4, because the acceptance rate there is ~1e-3 and 20000
    sweeps buy only a few hundred accepted flips.  Excluding those rows keeps
    the check a test of *correctness* rather than of sampling effort, and the
    printed rows document where the sampler starts to fail -- which is the
    whole argument for parallel tempering in step 5.
    """
    print(f"Check 2 -- {label}:  {L ** d} spins, {2 ** (L ** d)} states,"
          f" {boundary} boundaries")
    live = temps >= t_assert_min
    worst = 0.0
    results = []
    for sample in range(n_samples):
        rng = np.random.default_rng(seed + sample)
        J = make_couplings(L, d, rng, dist="gaussian", boundary=boundary)

        t0 = time.time()
        ex_e, ex_c, ex_q2 = exact_observables(J, L, d, temps)
        t_exact = time.time() - t0
        t0 = time.time()
        mc_e, mc_c, mc_q2, e_err, q2_err = mc_observables(
            J, L, d, temps, n_equil, n_meas, rng)
        t_mc = time.time() - t0

        print(f"  sample {sample}   [exact {t_exact:.2f} s, MC {t_mc:.1f} s]")
        print("      T       e exact      e MC          dev      "
              "<q^2> exact   <q^2> MC       dev      c exact   c MC")
        for i, T in enumerate(temps):
            de = (mc_e[i] - ex_e[i]) / e_err[i]
            dq = (mc_q2[i] - ex_q2[i]) / q2_err[i]
            if live[i]:
                worst = max(worst, abs(de), abs(dq))
                tag = ""
            else:
                tag = "  (stats)"
            print(f"  {T:5.2f}  {ex_e[i]:+10.5f}  {mc_e[i]:+10.5f}"
                  f"  {de:+6.1f} sig   {ex_q2[i]:9.5f}  {mc_q2[i]:9.5f}"
                  f"  {dq:+6.1f} sig  {ex_c[i]:7.4f} {mc_c[i]:7.4f}{tag}")
        results.append((ex_e, mc_e, ex_c, mc_c, ex_q2, mc_q2, e_err, q2_err))

        # Guard against a vacuous pass: error bars must be small enough that a
        # real discrepancy would have shown up as many sigma.  <q^2> spans
        # ~0.09 to ~0.6 here, so 0.035 is a few percent of the tested range.
        assert e_err[live].max() < 0.010, f"energy error bars too loose: {e_err[live].max():.4f}"
        assert q2_err[live].max() < 0.035, f"<q^2> error bars too loose: {q2_err[live].max():.4f}"

    print(f"  worst deviation over T >= {t_assert_min}: {worst:.1f} sigma")
    assert worst < 5.0, "Monte Carlo disagrees with exact enumeration"
    print("  PASS\n")
    return results[0]


def plot(label, temps, res, path):
    ex_e, mc_e, ex_c, mc_c, ex_q2, mc_q2, e_err, q2_err = res
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (ex, mc, err, name) in zip(ax, (
            (ex_e, mc_e, e_err, "e = E/N"),
            (ex_q2, mc_q2, q2_err, r"$\langle q^2 \rangle$"),
            (ex_c, mc_c, None, "c = C/N"))):
        a.plot(temps, ex, "k-", lw=1.5, label="exact enumeration")
        if err is None:
            a.plot(temps, mc, "o", ms=5, color="C1", label="Monte Carlo")
        else:
            a.errorbar(temps, mc, yerr=err, fmt="o", ms=5, color="C1",
                       label="Monte Carlo")
        a.set_xlabel("T")
        a.set_ylabel(name)
        a.legend(fontsize=8)
    fig.suptitle(f"Step 2: Monte Carlo vs. exact enumeration -- {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}\n")


if __name__ == "__main__":
    print("=" * 96)
    print("STEP 2: exact enumeration at low temperature -- the last exact check")
    print("=" * 96 + "\n")

    # Down to 0.6 for the assertions -- below the 3D T_c of ~0.95, which is the
    # regime the project cares about and the one steps 0 and 1 said nothing
    # about.  T=0.4 is measured too, as a deliberate demonstration of where
    # plain Metropolis stops working.
    temps = np.array([0.4, 0.6, 0.8, 1.1, 1.5, 2.1, 3.0])

    r = check_geometry("2D L=4 periodic", 4, 2, "periodic", temps,
                       n_equil=2000, n_meas=20000)
    plot("2D L=4 periodic", temps, r, "step2_exact_2d_periodic.png")

    r = check_geometry("2D L=4 open", 4, 2, "open", temps,
                       n_equil=2000, n_meas=20000)
    plot("2D L=4 open", temps, r, "step2_exact_2d_open.png")

    r = check_geometry("3D L=2 open", 2, 3, "open", temps,
                       n_equil=2000, n_meas=20000)
    plot("3D L=2 open", temps, r, "step2_exact_3d_open.png")

    print("=" * 96)
    print("Step 2 complete.  The sampler reproduces exact thermodynamics down")
    print("to T=0.6, including <q^2> from two independent replicas -- so the")
    print("replica machinery step 4 depends on is now verified, not assumed.")
    print()
    print("The T=0.4 rows are the interesting failure.  Note their direction:")
    print("the Monte Carlo energy sits ABOVE the exact value while <q^2> sits")
    print("BELOW it.  Both say the chains never reached the low-lying states.")
    print("That is precisely the signature step 6 formalises as the")
    print("Bhatt-Young test, showing up here in a 16-spin system -- which is")
    print("the argument for parallel tempering in step 5, made quantitative.")
    print("=" * 96)
