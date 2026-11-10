"""
P(q): the sharp ferromagnetic spikes, and how the spin glass differs.

A ferromagnet below T_c has ONE state and its global flip.  Two independent
replicas each pick up or down, so the overlap is q = +- m^2 and P(q) is two
spikes whose width is only the 1/sqrt(N) fluctuation of the magnetisation --
as sharp as a finite system allows.

A spin glass below T_c has peaks at +- q_EA too, but weight survives *between*
them, because the two replicas can land in genuinely different states.  How
much weight survives as L grows is the Parisi-vs-droplet question.

So the ferromagnet is the control that makes the spin glass P(q) readable: run
the identical code with every J = +1 and compare at matched reduced temperature
T/T_c, since the two models have completely different absolute scales
(T_c = 4.5115 vs ~0.95).

Consistency checks, both exact:

    S.1  ferromagnet peaks sit at q = +- m^2, with m^2 taken from the
         independently measured magnetisation -- not fitted
    S.2  the spin glass carries strictly more weight in |q| < 0.5 than the
         ferromagnet at the same T/T_c

Requires step7_d3_L8.npz from step7_production.py.
Run:  python agentic/07_Stat_Mech_and_Ising/pq_spikes.py
"""

import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ea_model import make_couplings
from ea_pt import make_ladder, run_sample
from step7_plots import load, pq_curves

TC_FERRO_3D = 4.5115
TC_GLASS_3D = 0.95
CACHE = "ferro_L8.npz"


def ferro_reference(L=8, d=3, n_seeds=4, n_equil=8000, n_meas=16000):
    """The same PT machinery with every J = +1.

    Cached: there is no disorder here, so this never needs rerunning once the
    statistics are adequate.  The ladder brackets the ferromagnetic T_c.
    """
    if os.path.exists(CACHE):
        print(f"  loading cached ferromagnet reference from {CACHE}")
        return dict(np.load(CACHE))

    temps = make_ladder(2.5, 6.5, 18)
    print(f"  running ferromagnet reference: L={L}, {n_seeds} seeds,"
          f" {n_equil}+{n_meas} sweeps")
    t0 = time.time()
    hist = None
    q2 = np.empty((n_seeds, len(temps)))
    for k in range(n_seeds):
        rng = np.random.default_rng(900 + k)
        J = make_couplings(L, d, rng, dist="ferro")
        out = run_sample(J, L, d, temps, n_equil, n_meas, rng)
        hist = out["hist"] if hist is None else hist + out["hist"]
        q2[k] = out["q2"]
        print(f"    seed {k + 1}/{n_seeds}  [{time.time() - t0:.0f} s]")
    data = dict(temps=temps, hist=hist, q2=q2, L=L, d=d)
    np.savez(CACHE, **data)
    print(f"    wrote {CACHE}  [{time.time() - t0:.0f} s]")
    return data


def weight_in_centre(data, t_target, cut=0.5):
    """Fraction of P(q) with |q| < cut, at the temperature nearest t_target."""
    temps = data["temps"]
    i = int(np.argmin(np.abs(temps - t_target)))
    h = data["hist"][i]
    nb = len(h)
    centres = 0.5 * (np.linspace(-1, 1, nb + 1)[1:] + np.linspace(-1, 1, nb + 1)[:-1])
    return float(h[np.abs(centres) < cut].sum() / h.sum()), float(temps[i])


def main():
    print("=" * 76)
    print("P(q): ferromagnetic spikes vs. spin glass")
    print("=" * 76 + "\n")

    ferro = ferro_reference()
    glass = load(3, 8)

    reduced = [0.6, 0.8, 1.0]
    t_f = [r * TC_FERRO_3D for r in reduced]
    t_g = [r * TC_GLASS_3D for r in reduced]

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))

    for T, c, p in pq_curves(ferro, t_f):
        ax[0].plot(c, p, lw=1.8, label=f"$T/T_c$ = {T / TC_FERRO_3D:.2f}")
    ax[0].set_title(f"ferromagnet, 3D L=8\ntwo sharp spikes: one state + its flip",
                    fontsize=10)

    for T, c, p in pq_curves(glass, t_g):
        ax[1].plot(c, p, lw=1.8, label=f"$T/T_c$ = {T / TC_GLASS_3D:.2f}")
    ax[1].set_title("spin glass, 3D L=8\npeaks at $\\pm q_{EA}$, weight in between",
                    fontsize=10)

    Tf, cf, pf = pq_curves(ferro, [0.6 * TC_FERRO_3D])[0]
    Tg, cg, pg = pq_curves(glass, [0.6 * TC_GLASS_3D])[0]
    ax[2].semilogy(cf, np.maximum(pf, 1e-4), lw=1.8, color="C2",
                   label=f"ferromagnet, $T/T_c$={Tf / TC_FERRO_3D:.2f}")
    ax[2].semilogy(cg, np.maximum(pg, 1e-4), lw=1.8, color="C3",
                   label=f"spin glass, $T/T_c$={Tg / TC_GLASS_3D:.2f}")
    ax[2].set_ylim(1e-4, 50)
    ax[2].set_title("same plot, log scale\nthe gap between the peaks is the physics",
                    fontsize=10)

    for a in ax:
        a.set_xlabel("q")
        a.set_ylabel("P(q)")
        a.legend(fontsize=8)
    fig.suptitle("P(q): what a single state looks like, and what many look like")
    fig.tight_layout()
    fig.savefig("pq_spikes.png", dpi=130)
    plt.close(fig)
    print("  wrote pq_spikes.png\n")

    # S.1 -- the ferromagnetic peaks must sit at +- m^2, with m^2 obtained from
    # <q^2> rather than from the peak positions themselves.  Below T_c,
    # q = +- m^2 with 1/sqrt(N) scatter, so <q^2> = m^4 and the peak is at
    # sqrt(<q^2>).  Nothing here is fitted to the histogram.
    i = int(np.argmin(np.abs(ferro["temps"] - 0.6 * TC_FERRO_3D)))
    m2_expected = np.sqrt(ferro["q2"].mean(axis=0)[i])
    peak = cf[int(np.argmax(pf[cf > 0]) + np.sum(cf <= 0))]
    print(f"S.1  ferromagnet at T/T_c = 0.60:")
    print(f"     peak of P(q) at q = {peak:+.4f}")
    print(f"     sqrt(<q^2>)       = {m2_expected:+.4f}   (independent estimate)")
    assert abs(peak - m2_expected) < 0.08, "ferromagnetic peak is not at m^2"
    print("     PASS\n")

    # S.2 -- the whole point: the glass keeps weight where the ferromagnet has
    # essentially none.
    wf, tf_used = weight_in_centre(ferro, 0.6 * TC_FERRO_3D)
    wg, tg_used = weight_in_centre(glass, 0.6 * TC_GLASS_3D)
    print("S.2  weight of P(q) in |q| < 0.5, at T/T_c = 0.60:")
    print(f"     ferromagnet (T={tf_used:.2f}): {wf:.4f}")
    print(f"     spin glass  (T={tg_used:.2f}): {wg:.4f}")
    print(f"     ratio: {wg / max(wf, 1e-9):.1f}x")
    assert wg > wf, "spin glass should carry more central weight than a ferromagnet"
    print("     PASS\n")
    print("=" * 76)


if __name__ == "__main__":
    main()
