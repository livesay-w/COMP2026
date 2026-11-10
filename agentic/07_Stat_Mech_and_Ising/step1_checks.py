"""
Step 1 -- add quenched disorder to the validated step 0 machinery.

Step 0 proved the lattice, the energy, and the Metropolis sweep are right on
the *unfrustrated* ferromagnet.  Step 1 draws random J_ij, keeps everything
else identical, and checks the three things that disorder newly puts at risk.

    Check 1.1  frustration is real      exactly 1/2 of plaquettes frustrated
               (ferromagnet: exactly 0)
    Check 1.2  incremental vs full E    machine precision over 2000 sweeps
    Check 1.3  high-temperature limit   e*T -> -d <J^2>  as T -> infinity

Check 1.1 is static -- no Monte Carlo at all.  Check 1.3 is the first
quantitative prediction for the disordered model, and it has no free
parameters.

Run:  python agentic/07_Stat_Mech_and_Ising/step1_checks.py
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ea_model import (
    energy,
    frustrated_fraction,
    make_couplings,
    metropolis_sweep,
    random_spins,
    sublattice_parity,
)


def check11_frustration():
    """Count frustrated plaquettes: those whose four bonds have a negative
    product, so that no assignment of the four spins satisfies all of them.

    For a symmetric coupling distribution the four signs are independent and
    symmetric, so the exact answer is 1/2 -- independent of L and d.  The
    ferromagnet must give exactly 0.  This is what separates the spin glass
    from step 0, and it needs no sampling to verify.
    """
    print("Check 1.1 -- frustrated plaquettes (static, no Monte Carlo)")
    for d in (2, 3):
        L = 10
        n_plaq = L ** d * (d * (d - 1) // 2)
        sigma = np.sqrt(0.25 / n_plaq)  # binomial error on a fraction of 1/2
        for dist, expect in (("ferro", 0.0), ("pm", 0.5), ("gaussian", 0.5)):
            J = make_couplings(L, d, np.random.default_rng(3), dist=dist)
            f = frustrated_fraction(J)
            print(f"  d={d}  L={L}  {dist:>8}  frustrated = {f:.4f}"
                  f"   expect {expect:.1f}   ({n_plaq} plaquettes, sigma={sigma:.4f})")
            if expect == 0.0:
                assert f == 0.0, "ferromagnet must have no frustration"
            else:
                assert abs(f - 0.5) < 4 * sigma, "frustrated fraction is not 1/2"
    print("  PASS: disorder frustrates exactly half the plaquettes;"
          " the ferromagnet none\n")


def check12_incremental_energy(T=1.0, n_sweeps=2000, check_every=200):
    """Track E by accumulating dE from accepted flips, and periodically
    recompute it from scratch.  They must agree to machine precision.

    This catches sign errors and double-counted bonds, but it does more than
    that here: the accumulated dE is only equal to the true energy change if no
    two simultaneously-flipped spins are neighbours.  So a broken checkerboard
    -- an odd L, or both sublattices updated at once -- shows up as drift that
    grows with sweep count.  With +-1 couplings every energy is an integer, so
    the drift there should be exactly zero.
    """
    print(f"Check 1.2 -- incremental vs. full energy ({n_sweeps} sweeps, T={T})")
    for d in (2, 3):
        for dist in ("gaussian", "pm"):
            L = 6
            rng = np.random.default_rng(4)
            J = make_couplings(L, d, rng, dist=dist)
            parity = sublattice_parity(L, d)
            s = random_spins(L, d, rng)

            e_tracked = energy(s, J)
            max_drift, acc = 0.0, 0.0
            for i in range(n_sweeps):
                a, dE = metropolis_sweep(s, J, T, rng, parity)
                e_tracked += dE
                acc += a
                if (i + 1) % check_every == 0:
                    max_drift = max(max_drift, abs(e_tracked - energy(s, J)))

            print(f"  d={d}  {dist:>8}   E = {e_tracked:+12.6f}"
                  f"   max drift = {max_drift:.2e}   accept = {acc / n_sweeps:.3f}")
            assert max_drift < 1e-8, "incremental energy drifted from the truth"
    print("  PASS: dE bookkeeping is exact, so the checkerboard is bipartite\n")


def high_T_scan(L, d, temps, n_equil, n_meas, n_samples, seed):
    """Disorder-averaged energy per spin at high temperature.

    Each disorder sample gets its own couplings and its own chain.  At high T
    the system decorrelates in roughly one sweep and the energy is strongly
    self-averaging, so a handful of samples and short runs suffice -- unlike
    P(q) later, which is not self-averaging at all.

    Returns e[n_samples, n_temps] and the empirical <J^2>, <J^4> per sample.
    """
    e = np.empty((n_samples, len(temps)))
    j2 = np.empty(n_samples)
    j4 = np.empty(n_samples)
    for sample in range(n_samples):
        rng = np.random.default_rng(seed + sample)
        J = make_couplings(L, d, rng, dist="gaussian")
        j2[sample] = float(np.mean(J ** 2))
        j4[sample] = float(np.mean(J ** 4))
        parity = sublattice_parity(L, d)
        s = random_spins(L, d, rng)
        for it, T in enumerate(temps):
            for _ in range(n_equil):
                metropolis_sweep(s, J, T, rng, parity)
            es = np.empty(n_meas)
            for i in range(n_meas):
                metropolis_sweep(s, J, T, rng, parity)
                es[i] = energy(s, J)
            e[sample, it] = es.mean() / s.size
    return e, j2, j4


def check13_high_temperature(d, L, temps, n_equil, n_meas, n_samples, seed=10):
    """The first quantitative prediction for the disordered model.

    As beta -> 0 a single bond contributes <s_i s_j> = tanh(beta J) ~ beta J,
    so with d*N bonds on an L^d lattice,

        e = E/N  ->  -d <J^2> / T,       i.e.   e*T -> -d <J^2>

    tanh is odd, so the first correction is O(beta^3): expanding
    tanh(x) = x - x^3/3 gives

        e*T = -d <J^2> + d <J^4> / (3 T^2) + O(1/T^4)

    so e*T plotted against 1/T^2 is a straight line whose intercept is the
    prediction.  (Lattice loop corrections enter at the same order but vanish
    on disorder average, since <J_ij J_jk J_kl J_li> = 0 for independent
    zero-mean bonds.)  No free parameters anywhere.

    Two things this check has to get right to be meaningful:

    * The comparison is against each sample's *empirical* <J^2>, not against 1.
      A finite lattice has only d*N couplings, so its sample variance differs
      from 1 by several percent -- larger than the statistical error, and
      enough to muddy the test if ignored.

    * The temperature window is kept moderate (T ~ 6-16) rather than extreme.
      sigma(e) is independent of T at high temperature, so sigma(e*T) grows
      *linearly* with T: a T=50 point is six times noisier than a T=8 one while
      adding almost no new information.  Going to very high T makes the check
      worse, not better.
    """
    print(f"Check 1.3 -- high-temperature limit, d={d}, L={L}, "
          f"{n_samples} disorder samples")
    t0 = time.time()
    e, j2, j4 = high_T_scan(L, d, temps, n_equil, n_meas, n_samples, seed)
    print(f"  {n_samples} x {len(temps)} runs of {n_equil}+{n_meas} sweeps"
          f"  [{time.time() - t0:.1f} s]")

    # Work with the per-sample *residual*, e*T + d<J^2>, which the prediction
    # says is (d<J^4>/3)/T^2 -- a line through the origin.  Subtracting each
    # sample's own -d<J^2> removes the sample-to-sample variation in coupling
    # variance, which is a genuine physical difference between realisations
    # and not an uncertainty in whether the law holds.  Left in, it swamps the
    # statistical error by about an order of magnitude and makes the check
    # look far weaker than it is.
    x = 1.0 / temps ** 2
    resid = e * temps + d * j2[:, None]
    fits = np.array([np.polyfit(x, resid[k], 1) for k in range(n_samples)])
    slope, intercept = fits[:, 0], fits[:, 1]
    slope_pred = d * j4 / 3.0

    print("   sample   <J^2>    <J^4>    intercept     slope   predicted")
    for k in range(n_samples):
        print(f"   {k:4d}    {j2[k]:.4f}   {j4[k]:.4f}  {intercept[k]:+9.4f}"
              f"  {slope[k]:+8.3f}  {slope_pred[k]:+8.3f}")

    r_mean = resid.mean(axis=0)
    r_err = resid.std(axis=0, ddof=1) / np.sqrt(n_samples)
    print("      T     e*T + d<J^2>        predicted")
    for i, T in enumerate(temps):
        print(f"  {T:6.1f}   {r_mean[i]:+8.4f} +- {r_err[i]:.4f}"
              f"     {slope_pred.mean() * x[i]:+8.4f}")

    # Two parameter-free targets now: intercept 0 and slope d<J^4>/3.  Each is
    # tested as a *mean* deviation against its standard error -- the mean of
    # |deviations| would measure per-sample scatter instead, and would not
    # shrink with more samples however correct the code was.
    for name, diff, floor, teeth in (("intercept", intercept, 0.02, 0.05),
                                     ("slope", slope - slope_pred, 0.30, 1.00)):
        mean_d = float(diff.mean())
        se = float(diff.std(ddof=1) / np.sqrt(n_samples))
        tol = max(3.0 * se, floor)  # floor absorbs the residual O(1/T^4) term
        print(f"  {name:>9}: mean deviation = {mean_d:+.4f} +- {se:.4f}"
              f"   tolerance {tol:.4f}")
        # Guard against a vacuous pass: a huge error bar agrees with anything.
        assert se < teeth, f"{name} error {se:.4f} too large for the test to have teeth"
        assert abs(mean_d) < tol, f"high-temperature {name} does not match prediction"
    print("  PASS\n")
    return e, j2, j4


def plot(d, L, temps, e, j2, j4, path):
    """Left panel is orientation; the right panel is the actual test.

    Left  -- e/<J^2> against T, with the predicted -d/T.  Dividing by each
             sample's own coupling variance collapses the disorder spread, so
             the error bars shown are statistical rather than a mixture of
             statistics and real variation between realisations.

    Right -- the residual e*T + d<J^2> against 1/T^2.  The prediction is a
             straight line through the *origin* with slope d<J^4>/3, so the
             dashed line is not fitted to anything: both its intercept and its
             slope come from the coupling moments alone.  Agreement here is
             the whole of check 1.3.
    """
    n = e.shape[0]
    x = 1.0 / temps ** 2

    en = e / j2[:, None]
    en_mean = en.mean(axis=0)
    en_err = en.std(axis=0, ddof=1) / np.sqrt(n)

    resid = e * temps + d * j2[:, None]
    r_mean = resid.mean(axis=0)
    r_err = resid.std(axis=0, ddof=1) / np.sqrt(n)
    slope_pred = d * j4.mean() / 3.0
    fit = np.polyfit(x, r_mean, 1)
    xs = np.linspace(0.0, x.max() * 1.05, 50)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))

    tt = np.linspace(temps.min() * 0.92, temps.max() * 1.04, 200)
    ax[0].errorbar(temps, en_mean, yerr=en_err, fmt="o", ms=4, label="Monte Carlo")
    ax[0].plot(tt, -d / tt, "k--", lw=1, label=r"$-d/T$")
    ax[0].set_xlabel("T")
    ax[0].set_ylabel(r"$e\,/\,\langle J^2\rangle$")
    ax[0].set_title("leading behaviour (orientation)", fontsize=9)
    ax[0].legend(fontsize=8)

    ax[1].axhline(0.0, color="0.6", lw=0.8)
    ax[1].errorbar(x, r_mean, yerr=r_err, fmt="o", ms=4, zorder=3, label="Monte Carlo")
    ax[1].plot(xs, slope_pred * xs, "k--", lw=1.2,
               label=rf"predicted: slope $d\langle J^4\rangle/3$ = {slope_pred:.2f}")
    ax[1].plot(xs, np.polyval(fit, xs), "-", lw=1, color="C1",
               label=f"fit: slope {fit[0]:.2f}, intercept {fit[1]:+.3f}")
    ax[1].set_xlabel(r"$x = 1/T^2$")
    ax[1].set_ylabel(r"$e\,T + d\langle J^2\rangle$")
    ax[1].set_title("residual: must pass through the origin", fontsize=9)
    ax[1].legend(fontsize=7.5)

    fig.suptitle(f"Step 1: high-temperature limit with quenched disorder, "
                 f"d={d}, L={L}, {n} disorder samples")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}\n")


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 1: quenched disorder on the step 0 machinery")
    print("=" * 72 + "\n")

    check11_frustration()
    check12_incremental_energy()

    # Moderate temperatures, not extreme ones: sigma(e*T) grows linearly in T.
    # T=5 is included to lengthen the lever arm in 1/T^2, which is what pins
    # down the slope; the O(1/T^4) term there is still ~0.01 out of ~3.
    temps = np.array([16.0, 13.0, 10.0, 8.0, 6.0, 5.0])

    e2, j2_2, j4_2 = check13_high_temperature(2, 10, temps, n_equil=300,
                                              n_meas=6000, n_samples=6)
    plot(2, 10, temps, e2, j2_2, j4_2, "step1_highT_2d.png")

    e3, j2_3, j4_3 = check13_high_temperature(3, 6, temps, n_equil=300,
                                              n_meas=6000, n_samples=6)
    plot(3, 6, temps, e3, j2_3, j4_3, "step1_highT_3d.png")

    print("=" * 72)
    print("Step 1 complete.  Disorder is in, frustration is verified, the dE")
    print("bookkeeping is exact, and e(T) matches the parameter-free high-T")
    print("limit.  Step 2 pins the code down at *low* T against exact")
    print("enumeration of a small lattice, where Metropolis alone can still")
    print("be trusted because the system is tiny.")
    print("=" * 72)
