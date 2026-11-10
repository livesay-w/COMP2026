"""
Step 0 -- validate the lattice, the energy, and the sampler before any disorder.

Three of the four checks need no Monte Carlo at all and demand *exact*
agreement.  They are run first on purpose: once sampling and disorder enter,
every disagreement becomes arguable ("bug, or just statistics?") and the
ability to demand equality is gone.

    Check 1  ferromagnetic ground state      e = -d exactly
    Check 2  np.roll energy vs. brute force  machine precision
    Check 3  field/energy identity           sum_i s_i h_i = -2E
    Check 4  ferromagnet thermodynamics      T_c = 2.2692 (2D), 4.5115 (3D)

Run:  python agentic/07_Stat_Mech_and_Ising/step0_checks.py
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ea_model import (
    energy,
    local_fields,
    make_couplings,
    metropolis_sweep,
    random_spins,
    sublattice_parity,
)

# Exact critical temperatures of the *ferromagnetic* Ising model, the
# unfrustrated reference this step validates against.
TC_2D = 2.0 / np.log(1.0 + np.sqrt(2.0))  # Onsager, 2.269185...
TC_3D = 4.5115                            # high-precision MC literature value


def energy_bruteforce(s, J):
    """Reference energy from explicit Python loops over every bond.

    Deliberately slow and dumb.  Its only job is to validate the np.roll
    indexing in energy() -- that rolling by -1 along axis k really walks the
    neighbour we think it does.
    """
    d, L, shape = J.shape[0], s.shape[0], s.shape
    total = 0.0
    for flat in range(s.size):
        x = np.unravel_index(flat, shape)
        for k in range(d):
            y = list(x)
            y[k] = (y[k] + 1) % L
            total -= J[(k,) + x] * s[x] * s[tuple(y)]
    return float(total)


def check1_ground_state():
    """All J = +1, all spins up.  With periodic boundaries an L^d lattice has
    exactly d*N bonds, so e = -d on the nose.  Prints -6 if bonds are being
    double-counted, -1.5 if something was halved that should not have been."""
    print("Check 1 -- ferromagnetic ground state energy")
    for d in (2, 3):
        L = 4
        J = make_couplings(L, d, np.random.default_rng(0), dist="ferro")
        s = np.ones((L,) * d, dtype=np.int8)
        e = energy(s, J) / s.size
        print(f"  d={d}  L={L}   e = {e:+.15f}   exact = {-float(d):+.1f}")
        assert e == -float(d), f"expected {-float(d)}, got {e}"

        # All-up ferromagnet: every site sees 2d aligned neighbours.
        h = local_fields(s, J)
        assert np.all(h == 2 * d), "local field should be 2d everywhere"
    print("  PASS: energies exact, local field = 2d everywhere\n")


def check2_bruteforce():
    """Vectorised energy vs. explicit per-bond loops, on random Gaussian
    couplings and a random configuration."""
    print("Check 2 -- np.roll energy vs. brute-force bond loop")
    for d in (2, 3):
        L = 4
        rng = np.random.default_rng(1)
        J = make_couplings(L, d, rng, dist="gaussian")
        s = random_spins(L, d, rng)
        fast, slow = energy(s, J), energy_bruteforce(s, J)
        print(f"  d={d}  L={L}   vectorised = {fast:+.12f}   brute = {slow:+.12f}"
              f"   diff = {abs(fast - slow):.2e}")
        assert np.isclose(fast, slow, rtol=0, atol=1e-10), "indexing mismatch"
    print("  PASS: indexing agrees to machine precision\n")


def check3_field_identity():
    """sum_i s_i h_i = -2E, because every bond is counted once from each end.

    This ties local_fields() to energy().  A sampler with a correct energy but
    a subtly wrong local field runs happily, accepts at a plausible rate, and
    returns wrong physics with no visible symptom -- so test the link directly.
    """
    print("Check 3 -- field/energy identity  sum_i s_i h_i = -2E")
    for d in (2, 3):
        for dist in ("gaussian", "pm"):
            L = 6
            rng = np.random.default_rng(2)
            J = make_couplings(L, d, rng, dist=dist)
            s = random_spins(L, d, rng)
            lhs = float(np.sum(s * local_fields(s, J)))
            rhs = -2.0 * energy(s, J)
            print(f"  d={d}  {dist:>8}   sum s.h = {lhs:+.10f}   -2E = {rhs:+.10f}"
                  f"   diff = {abs(lhs - rhs):.2e}")
            assert np.isclose(lhs, rhs, rtol=0, atol=1e-9), "local_fields inconsistent with energy"
    print("  PASS: local_fields is consistent with energy\n")


def ferro_scan(L, d, temps, n_equil, n_meas, seed):
    """Metropolis on the unfrustrated ferromagnet, cooled through T_c.

    Temperatures are visited in descending order carrying the configuration
    over, which is the cheapest sensible equilibration schedule.  There is no
    frustration here, so Metropolis suffers only critical slowing down
    (tau ~ L^z, z ~ 2) -- a polynomial nuisance, not the wall that appears
    once the couplings have mixed signs.
    """
    rng = np.random.default_rng(seed)
    J = make_couplings(L, d, rng, dist="ferro")
    parity = sublattice_parity(L, d)
    s = random_spins(L, d, rng)
    N = s.size

    e, m_abs, chi, acc = [], [], [], []
    for T in temps:
        for _ in range(n_equil):
            metropolis_sweep(s, J, T, rng, parity)

        es = np.empty(n_meas)
        ms = np.empty(n_meas)
        a = 0.0
        for i in range(n_meas):
            a += metropolis_sweep(s, J, T, rng, parity)[0]
            es[i] = energy(s, J)
            ms[i] = s.mean()

        e.append(es.mean() / N)
        m_abs.append(np.abs(ms).mean())
        # chi from the fluctuations of |m|, the finite-size-safe definition:
        # below T_c a finite system tunnels between +m and -m over long runs.
        chi.append(N * (np.mean(ms ** 2) - np.mean(np.abs(ms)) ** 2) / T)
        acc.append(a / n_meas)

    return np.array(e), np.array(m_abs), np.array(chi), np.array(acc)


def check4_ferro(d, L, temps, tc_ref, cold, hot, n_equil, n_meas, seed=0):
    """Does the sampler reproduce the known ferromagnetic transition?

    Be honest about what a single L can show: the finite-size susceptibility
    peak is shifted from T_c by O(L^{-1/nu}) with nu ~ 0.63 in 3D, so a few
    percent off is correct behaviour, not a bug.  The hard assertions are on
    the robust limits; the peak position is reported against the reference.
    """
    print(f"Check 4 -- ferromagnet thermodynamics, d={d}, L={L}"
          f"  (T_c = {tc_ref:.4f})")
    t0 = time.time()
    e, m, chi, acc = ferro_scan(L, d, temps, n_equil, n_meas, seed)
    print(f"  {len(temps)} temperatures, {n_equil}+{n_meas} sweeps each"
          f"  [{time.time() - t0:.1f} s]")
    print("      T       e/N      <|m|>       chi     accept")
    for i, T in enumerate(temps):
        mark = "  <-- chi peak" if i == int(np.argmax(chi)) else ""
        print(f"  {T:6.3f}  {e[i]:+8.4f}  {m[i]:8.4f}  {chi[i]:9.3f}"
              f"  {acc[i]:7.3f}{mark}")

    i_cold = int(np.argmin(np.abs(temps - cold)))
    i_hot = int(np.argmin(np.abs(temps - hot)))
    t_peak = temps[int(np.argmax(chi))]
    print(f"  ordered   T={temps[i_cold]:.2f}: <|m|> = {m[i_cold]:.3f}  (expect > 0.9)")
    print(f"  disordered T={temps[i_hot]:.2f}: <|m|> = {m[i_hot]:.3f}  (expect < 0.2)")
    print(f"  chi peak at T = {t_peak:.3f}   reference T_c = {tc_ref:.4f}"
          f"   ({100 * abs(t_peak - tc_ref) / tc_ref:.1f}% off)")

    assert m[i_cold] > 0.9, "ferromagnet failed to order well below T_c"
    assert m[i_hot] < 0.2, "ferromagnet still magnetised well above T_c"
    assert e[-1] < -0.8 * d, "low-T energy per spin nowhere near the ground state -d"
    assert abs(t_peak - tc_ref) / tc_ref < 0.10, "chi peak far from the known T_c"
    print("  PASS\n")
    return e, m, chi


def plot(d, L, temps, e, m, chi, tc_ref, path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for a, y, lab in zip(ax, (e, m, chi), ("e = E/N", r"$\langle |m| \rangle$", r"$\chi$")):
        a.plot(temps, y, "o-", ms=4)
        a.axvline(tc_ref, color="k", ls="--", lw=1, label=f"$T_c$ = {tc_ref:.4f}")
        a.set_xlabel("T")
        a.set_ylabel(lab)
        a.legend(fontsize=8)
    fig.suptitle(f"Step 0: ferromagnetic Ising check, d={d}, L={L}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}\n")


if __name__ == "__main__":
    print("=" * 68)
    print("STEP 0: lattice, energy, and sampler -- no disorder yet")
    print("=" * 68 + "\n")

    check1_ground_state()
    check2_bruteforce()
    check3_field_identity()

    # 2D first: 256 spins, and Onsager gives an exact target.
    t2 = np.array([3.5, 3.2, 3.0, 2.8, 2.6, 2.5, 2.4, 2.35, 2.3,
                   2.25, 2.2, 2.1, 2.0, 1.8, 1.5])
    e2, m2, c2 = check4_ferro(2, 16, t2, TC_2D, cold=1.5, hot=3.5,
                              n_equil=800, n_meas=3000)
    plot(2, 16, t2, e2, m2, c2, TC_2D, "step0_ferro_2d.png")

    # Then 3D, the actual target geometry: 512 spins.
    t3 = np.array([6.0, 5.5, 5.2, 5.0, 4.8, 4.7, 4.6, 4.5, 4.4,
                   4.3, 4.2, 4.0, 3.5, 3.0, 2.5])
    e3, m3, c3 = check4_ferro(3, 8, t3, TC_3D, cold=3.0, hot=6.0,
                              n_equil=800, n_meas=3000)
    plot(3, 8, t3, e3, m3, c3, TC_3D, "step0_ferro_3d.png")

    print("=" * 68)
    print("Step 0 complete.  The lattice, energy, local field, and Metropolis")
    print("sweep are validated against exact values.  Step 1 adds quenched")
    print("disorder to the same code and checks it against the high-T limit.")
    print("=" * 68)
