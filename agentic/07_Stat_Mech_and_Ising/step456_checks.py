"""
Steps 4-6 -- replicas, parallel tempering, and thermalisation.

Step 2 established that the sampler is correct where it can be checked exactly,
and that plain Metropolis stops working somewhere below T=0.6 even on 16 spins.
These three steps build the machinery that fixes that and verify it:

    Check 4.1  P(q) symmetry under q -> -q          global Ising symmetry
    Check 4.2  PT reproduces exact enumeration      INCLUDING T=0.4, where
                                                    plain Metropolis failed
    Check 5.1  swap acceptance uniform across rungs 20-40%, no dead pairs
    Check 5.2  replica round trips through ladder   escape mechanism working
    Check 6.1  link-overlap identity                exact for Gaussian bonds
    Check 6.2  log-binning                          last half vs last quarter

Check 4.2 is the one that matters most: it is the same comparison step 2 made,
with parallel tempering in place of plain Metropolis, on the same 8-spin
lattice with the same exact answer.  If PT works, the T=0.4 row that failed in
step 2 now passes.  That is a direct demonstration rather than an assertion.

Check 6.1 deserves a note.  For Gaussian couplings, integrating by parts over
the disorder gives an *exact* relation between the link overlap and the energy:

    q_link = 1 + u T / (d <J^2>),      u = <E>/N

The left side is measured directly from two replicas; the right side comes
from the energy alone.  Out of equilibrium they approach the truth from
opposite sides, so the gap between them bounds how unthermalised the run is.
This is the Bhatt-Young/Katzgraber test, and it is the only thermalisation
diagnostic here that is quantitative rather than visual.

Run:  python agentic/07_Stat_Mech_and_Ising/step456_checks.py
"""

import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ea_model import make_couplings
from ea_pt import make_ladder, run_sample
from step2_checks import exact_observables


def check42_vs_exact(temps, n_equil, n_meas, n_samples=6, seed=40):
    """Parallel tempering against exact enumeration, 3D L=2 open (8 spins).

    The same geometry and the same exact answer as step 2, so the comparison
    is like for like.  The T=0.4 column is the point of the exercise.
    """
    L, d = 2, 3
    print(f"Check 4.2 -- PT vs exact enumeration, 3D L={L} open ({L ** d} spins)")
    worst_e = worst_q = 0.0
    acc_e, acc_q = [], []
    for sample in range(n_samples):
        rng = np.random.default_rng(seed + sample)
        J = make_couplings(L, d, rng, dist="gaussian", boundary="open")
        ex_e, _, ex_q2 = exact_observables(J, L, d, temps)
        out = run_sample(J, L, d, temps, n_equil, n_meas, rng)
        de = (out["e"] - ex_e) / out["e_err"]
        dq = (out["q2"] - ex_q2) / out["q2_err"]
        worst_e = max(worst_e, np.abs(de).max())
        worst_q = max(worst_q, np.abs(dq).max())
        acc_e.append(de)
        acc_q.append(dq)
        if sample == 0:
            print("      T      e exact     e PT         dev    "
                  "<q^2> exact  <q^2> PT       dev")
            for i, T in enumerate(temps):
                print(f"  {T:5.2f}  {ex_e[i]:+10.5f} {out['e'][i]:+10.5f}"
                      f"  {de[i]:+6.1f} sig  {ex_q2[i]:9.5f} {out['q2'][i]:9.5f}"
                      f"  {dq[i]:+6.1f} sig")
    print(f"  worst over {n_samples} samples and all T:"
          f"  e {worst_e:.1f} sigma,  <q^2> {worst_q:.1f} sigma")
    assert worst_e < 5.0, "PT energy disagrees with exact enumeration"
    assert worst_q < 5.0, "PT <q^2> disagrees with exact enumeration"
    print("  PASS: PT is exact at every temperature, T=0.4 included\n")
    return np.array(acc_e), np.array(acc_q)


def production_like(L, d, temps, n_equil, n_meas, n_samples, seed=50):
    """A short production-shaped run, used by checks 4.1, 5.x and 6.x."""
    t0 = time.time()
    outs = []
    for sample in range(n_samples):
        rng = np.random.default_rng(seed + sample)
        J = make_couplings(L, d, rng, dist="gaussian")
        outs.append(run_sample(J, L, d, temps, n_equil, n_meas, rng))
    print(f"  {n_samples} samples, L={L}, d={d}, {len(temps)} temperatures,"
          f" {n_equil}+{n_meas} sweeps  [{time.time() - t0:.1f} s]")
    return outs


def check41_symmetry(outs, temps):
    """P(q) must be symmetric under q -> -q.

    Nothing in the Hamiltonian distinguishes s from -s, so any asymmetry is
    either a failure to thermalise or correlated replicas -- both fatal, and
    both invisible in the energy.  This is the cheapest check that the overlap
    machinery is sound.
    """
    print("Check 4.1 -- P(q) symmetry under q -> -q")
    hist = sum(o["hist"] for o in outs)
    worst, worst_t = 0.0, None
    for i, T in enumerate(temps):
        h = hist[i]
        tot = h.sum()
        asym = np.abs(h - h[::-1]).sum() / (2 * tot)
        if asym > worst:
            worst, worst_t = asym, T
    print(f"  worst asymmetry {worst:.4f} of total weight, at T={worst_t:.3f}")
    # With finite statistics a perfectly symmetric distribution still shows
    # residual asymmetry of order 1/sqrt(counts per bin).
    assert worst < 0.08, "P(q) is not symmetric -- replicas correlated or unthermalised"
    print("  PASS\n")
    return hist


def check5_swaps(outs, temps):
    """Uniform swap acceptance and enough round trips.

    A pair stuck near zero acceptance is a gap in the ladder, and every rung
    below it is still trapped -- PT has bought nothing there.  Round trips are
    the honest diagnostic: a replica that never reaches the hot end never
    escapes anything.
    """
    print("Check 5.1/5.2 -- swap acceptance and round trips")
    acc = np.mean([o["swap_acc"] for o in outs], axis=0)
    trips = np.mean([o["trips"].mean() for o in outs])
    print("   pair   T_lo   T_hi   acceptance")
    for i in range(len(temps) - 1):
        flag = "   <-- low" if acc[i] < 0.15 else ""
        print(f"   {i:3d}  {temps[i]:6.3f} {temps[i + 1]:6.3f}   {acc[i]:.3f}{flag}")
    print(f"  mean round trips per replica: {trips:.1f}")
    assert acc.min() > 0.10, f"dead rung in the ladder: min acceptance {acc.min():.3f}"
    assert trips > 5.0, f"too few round trips ({trips:.1f}) -- replicas still trapped"
    print("  PASS\n")
    return acc


def check61_link_identity(outs, temps, d):
    """q_link measured directly vs. derived from the energy.

    Exact for Gaussian couplings after disorder averaging.  Out of equilibrium
    the two approach from opposite sides, so their difference bounds the
    non-equilibration -- which is what makes this a test rather than a
    tautology.
    """
    print("Check 6.1 -- link-overlap identity  q_link = 1 + u T / d")
    n = len(outs)
    q_direct = np.mean([o["q_link"] for o in outs], axis=0)
    q_err = np.std([o["q_link"] for o in outs], axis=0, ddof=1) / np.sqrt(n)
    u = np.mean([o["e"] for o in outs], axis=0)
    u_err = np.std([o["e"] for o in outs], axis=0, ddof=1) / np.sqrt(n)
    q_energy = 1.0 + u * temps / d
    q_energy_err = u_err * temps / d

    print("      T     q_link direct     q_link from E        difference")
    worst = 0.0
    for i, T in enumerate(temps):
        diff = q_direct[i] - q_energy[i]
        sig = np.hypot(q_err[i], q_energy_err[i])
        worst = max(worst, abs(diff) / sig)
        print(f"  {T:5.3f}   {q_direct[i]:.5f} +- {q_err[i]:.5f}"
              f"   {q_energy[i]:.5f} +- {q_energy_err[i]:.5f}"
              f"   {diff:+.5f} ({diff / sig:+.1f} sig)")
    print(f"  worst deviation: {worst:.1f} sigma")
    assert worst < 5.0, "link-overlap identity violated -- not thermalised"
    print("  PASS\n")
    return q_direct, q_energy


def check62_logbinning(outs, temps):
    """Last half vs. last quarter of the measurement phase.

    If the results are still moving, the run is not converged.  Unlike the
    identity above this is only a necessary condition -- a chain stuck in one
    basin looks perfectly converged to it -- which is why both are here.
    """
    print("Check 6.2 -- log-binning: last half vs. last quarter of the run")
    n = len(outs)
    half = np.mean([o["q2_half"] for o in outs], axis=0)
    quarter = np.mean([o["q2_quarter"] for o in outs], axis=0)
    err = np.std([o["q2_half"] - o["q2_quarter"] for o in outs], axis=0,
                 ddof=1) / np.sqrt(n)
    dev = np.abs(half - quarter) / np.maximum(err, 1e-12)
    print("      T     <q^2> half   <q^2> quarter     drift")
    for i, T in enumerate(temps):
        print(f"  {T:5.3f}   {half[i]:9.5f}   {quarter[i]:9.5f}"
              f"   {half[i] - quarter[i]:+.5f} ({dev[i]:+.1f} sig)")
    print(f"  worst drift: {dev.max():.1f} sigma")
    assert dev.max() < 5.0, "<q^2> still drifting -- not thermalised"
    print("  PASS\n")


def plot_diagnostics(temps, hist, acc, q_direct, q_energy, dev_e, dev_q, path):
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))

    centres = np.linspace(-1.0, 1.0, hist.shape[1] + 1)
    centres = 0.5 * (centres[1:] + centres[:-1])
    for i in (0, len(temps) // 3, 2 * len(temps) // 3, len(temps) - 1):
        p = hist[i] / hist[i].sum() / (centres[1] - centres[0])
        ax[0].plot(centres, p, lw=1.3, label=f"T = {temps[i]:.2f}")
    ax[0].set_xlabel("q")
    ax[0].set_ylabel("P(q)")
    ax[0].set_title("4.1  P(q) is symmetric", fontsize=9)
    ax[0].legend(fontsize=7)

    ax[1].plot(0.5 * (temps[1:] + temps[:-1]), acc, "o-", ms=4)
    ax[1].axhspan(0.2, 0.4, color="C2", alpha=0.15, label="healthy band")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlabel("T (pair midpoint)")
    ax[1].set_ylabel("swap acceptance")
    ax[1].set_title("5.1  no dead rungs", fontsize=9)
    ax[1].legend(fontsize=7)

    ax[2].plot(temps, q_direct, "o-", ms=4, label="measured from replicas")
    ax[2].plot(temps, q_energy, "s--", ms=4, label=r"from energy: $1 + uT/d$")
    ax[2].set_xlabel("T")
    ax[2].set_ylabel(r"$q_\mathrm{link}$")
    ax[2].set_title("6.1  exact identity holds", fontsize=9)
    ax[2].legend(fontsize=7)

    ax[3].hist(dev_e.ravel(), bins=15, alpha=0.6, label="e")
    ax[3].hist(dev_q.ravel(), bins=15, alpha=0.6, label=r"$\langle q^2 \rangle$")
    ax[3].set_xlabel("deviation from exact / sigma")
    ax[3].set_ylabel("count")
    ax[3].set_title("4.2  PT matches enumeration", fontsize=9)
    ax[3].legend(fontsize=7)

    fig.suptitle("Steps 4-6: parallel tempering validated")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}\n")


if __name__ == "__main__":
    print("=" * 84)
    print("STEPS 4-6: replicas, parallel tempering, thermalisation")
    print("=" * 84 + "\n")

    # Same geometry and temperatures step 2 used, so the comparison is direct.
    exact_temps = make_ladder(0.4, 3.0, 12)
    dev_e, dev_q = check42_vs_exact(exact_temps, n_equil=3000, n_meas=12000)

    # Production-shaped: 3D L=4, a ladder spanning T_c ~ 0.95.
    temps = make_ladder(0.5, 2.0, 16)
    outs = production_like(4, 3, temps, n_equil=4000, n_meas=8000, n_samples=12)
    hist = check41_symmetry(outs, temps)
    acc = check5_swaps(outs, temps)
    q_direct, q_energy = check61_link_identity(outs, temps, 3)
    check62_logbinning(outs, temps)

    plot_diagnostics(temps, hist, acc, q_direct, q_energy, dev_e, dev_q,
                     "step456_diagnostics.png")

    print("=" * 84)
    print("Steps 4-6 complete.  Parallel tempering reproduces exact")
    print("enumeration at every temperature including T=0.4, where plain")
    print("Metropolis failed in step 2.  Swaps are healthy, replicas make")
    print("round trips, and both thermalisation tests pass.  Step 7 runs this")
    print("at production scale for L = 4, 6, 8 and locates T_c.")
    print("=" * 84)
