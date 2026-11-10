"""
Step 7 analysis -- turn the production .npz files into the figures.

Separate from the simulation on purpose: the runs take tens of minutes, the
plots take a second, and you will want to redraw them many times.

Three figures:

  step7_main_results.png   the four panels that are the EA result
  step7_pq_by_size.png     P(q) against system size, 3D vs 2D
  step7_equilibration.png  the evidence that the runs are converged

Error bars are jackknife over disorder samples.  That matters for xi_L and the
Binder ratio because both are *non-linear* functions of averaged quantities --
chi(0)/chi(k) and <q^4>/<q^2>^2 -- so propagating errors naively would be
wrong, and averaging the ratio sample by sample would be biased.

Run:  python agentic/07_Stat_Mech_and_Ising/step7_plots.py
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TC_3D_REF = 0.95  # literature value for 3D EA with Gaussian bonds


def load(d, L):
    return dict(np.load(f"step7_d{d}_L{L}.npz"))


def load_available(d, sizes):
    """Load the geometries that exist, and say plainly which do not.

    The production run is long and has already been interrupted once, so the
    figures are built from whatever is finished rather than refusing to draw
    anything.  Missing sizes are named in the output and in the panel titles --
    a half-finished crossing must not be mistaken for a complete one.
    """
    out, missing = [], []
    for L in sizes:
        path = f"step7_d{d}_L{L}.npz"
        (out.append(load(d, L)) if os.path.exists(path) else missing.append(L))
    if missing:
        print(f"  d={d}: missing L={missing} (still running); using L="
              f"{[int(o['L']) for o in out]}")
    return out


def jackknife(fn, *arrays):
    """Estimate and error for fn(<a>, <b>, ...) by leaving out one sample.

    The averages go inside fn, not outside: xi_L and the Binder ratio are
    non-linear in the averaged quantities, so the order matters.
    """
    n = arrays[0].shape[0]
    full = fn(*[a.mean(axis=0) for a in arrays])
    parts = np.array([fn(*[np.delete(a, k, axis=0).mean(axis=0) for a in arrays])
                      for k in range(n)])
    err = np.sqrt((n - 1) / n * ((parts - parts.mean(axis=0)) ** 2).sum(axis=0))
    return full, err


def xi_over_L(data):
    L = int(data["L"])
    scale = 1.0 / (2.0 * np.sin(np.pi / L) * L)

    def fn(chi0, chik):
        return np.sqrt(np.maximum(chi0 / chik - 1.0, 0.0)) * scale

    return jackknife(fn, data["chi0"], data["chik"])


def binder(data):
    def fn(q2, q4):
        return 0.5 * (3.0 - q4 / q2 ** 2)

    return jackknife(fn, data["q2"], data["q4"])


def crossing(temps, y1, y2):
    """Temperature where two curves intersect, by linear interpolation.

    Returns None when they never cross in the measured window -- which is the
    expected and desired answer in 2D.
    """
    diff = y1 - y2
    for i in range(len(temps) - 1):
        if diff[i] == 0.0:
            return float(temps[i])
        if diff[i] * diff[i + 1] < 0.0:
            f = diff[i] / (diff[i] - diff[i + 1])
            return float(temps[i] + f * (temps[i + 1] - temps[i]))
    return None


def pq_curves(data, which_temps):
    """Normalised P(q) at selected temperatures, rebinned to match quantisation.

    q = (1/N) sum_i s_i^a s_i^b takes only the discrete values (N - 2m)/N, so
    it is quantised in steps of 2/N.  When the histogram bins are narrower than
    that spacing, allowed values land in some bins and skip others and the
    curve combs -- full bins alternating with empty ones.  That is a binning
    artefact and nothing to do with the physics.

    At L=8 (N=512, spacing 0.004) the stored 101 bins already hold about five
    allowed values each and need no help.  At L=4 (N=64, spacing 0.031) they do.
    Adjacent bins are merged until each holds at least FOUR allowed values,
    which for a histogram is simply coarser binning.  Four, not two: at two the
    bins alternate between holding two and three allowed values and the curve
    still visibly zigzags.  The cost is a coarse histogram at small L, which is
    honest -- L=4 has only 65 possible values of q in total.

    reduceat is used because 101 is prime and will not divide evenly.
    """
    temps = data["temps"]
    hist = data["hist"]
    n_spins = int(data["L"]) ** int(data["d"])
    nb = hist.shape[1]
    raw_width = 2.0 / nb
    group = max(1, int(np.ceil(4.0 * (2.0 / n_spins) / raw_width)))

    edges = np.linspace(-1.0, 1.0, nb + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.arange(0, nb, group)
    per_group = np.diff(np.append(idx, nb))
    c = np.add.reduceat(centres, idx) / per_group
    widths = per_group * raw_width

    out = []
    for t in which_temps:
        i = int(np.argmin(np.abs(temps - t)))
        h = np.add.reduceat(hist[i], idx)
        out.append((temps[i], c, h / h.sum() / widths))
    return out


def main_results(sets_3d, sets_2d, path):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # (a) P(q) across the transition, largest 3D lattice.
    big = sets_3d[-1]
    for T, c, p in pq_curves(big, [0.50, 0.70, 0.95, 1.30, 2.00]):
        ax[0, 0].plot(c, p, lw=1.6, label=f"T = {T:.2f}")
    ax[0, 0].set_xlabel("q")
    ax[0, 0].set_ylabel("P(q)")
    ax[0, 0].set_title(f"(a)  3D L={int(big['L'])}: P(q) across the transition",
                       fontsize=10)
    ax[0, 0].legend(fontsize=8)

    # (b) xi_L/L in 3D -- the curves must cross, and the crossing is T_c.
    cross = []
    for data in sets_3d:
        temps = data["temps"]
        y, err = xi_over_L(data)
        ax[0, 1].errorbar(temps, y, yerr=err, fmt="o-", ms=3.5, lw=1.2,
                          label=f"L = {int(data['L'])}")
    for i in range(len(sets_3d) - 1):
        y1 = xi_over_L(sets_3d[i])[0]
        y2 = xi_over_L(sets_3d[i + 1])[0]
        tc = crossing(sets_3d[i]["temps"], y1, y2)
        if tc is not None:
            cross.append(tc)
    ax[0, 1].axvline(TC_3D_REF, color="k", ls="--", lw=1,
                     label=f"literature $T_c$ = {TC_3D_REF}")
    if cross:
        ax[0, 1].axvline(np.mean(cross), color="C3", ls=":", lw=1.5,
                         label=f"crossing = {np.mean(cross):.3f}")
    ax[0, 1].set_xlabel("T")
    ax[0, 1].set_ylabel(r"$\xi_L / L$")
    ax[0, 1].set_title("(b)  3D: curves cross at $T_c$", fontsize=10)
    ax[0, 1].legend(fontsize=8)

    # (c) The negative control: same pipeline, 2D, no crossing.
    for data in sets_2d:
        temps = data["temps"]
        y, err = xi_over_L(data)
        ax[1, 0].errorbar(temps, y, yerr=err, fmt="s-", ms=3.5, lw=1.2,
                          label=f"L = {int(data['L'])}")
    ax[1, 0].set_xlabel("T")
    ax[1, 0].set_ylabel(r"$\xi_L / L$")
    ax[1, 0].set_title("(c)  2D control: $T_c = 0$, curves must NOT cross",
                       fontsize=10)
    if ax[1, 0].lines:
        ax[1, 0].legend(fontsize=8)
    else:
        ax[1, 0].text(0.5, 0.5, "2D control still running",
                      ha="center", va="center", transform=ax[1, 0].transAxes,
                      color="0.5")

    # (d) Binder ratio -- an independent crossing from different moments.
    cross_b = []
    for data in sets_3d:
        y, err = binder(data)
        ax[1, 1].errorbar(data["temps"], y, yerr=err, fmt="o-", ms=3.5, lw=1.2,
                          label=f"L = {int(data['L'])}")
    for i in range(len(sets_3d) - 1):
        tc = crossing(sets_3d[i]["temps"], binder(sets_3d[i])[0],
                      binder(sets_3d[i + 1])[0])
        if tc is not None:
            cross_b.append(tc)
    ax[1, 1].axvline(TC_3D_REF, color="k", ls="--", lw=1,
                     label=f"literature $T_c$ = {TC_3D_REF}")
    if cross_b:
        ax[1, 1].axvline(np.mean(cross_b), color="C3", ls=":", lw=1.5,
                         label=f"crossing = {np.mean(cross_b):.3f}")
    ax[1, 1].set_xlabel("T")
    ax[1, 1].set_ylabel(r"Binder $g$")
    ax[1, 1].set_title("(d)  3D: independent estimator, same $T_c$", fontsize=10)
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("Step 7: the 3D Edwards-Anderson spin glass transition", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")
    return cross, cross_b


def pq_by_size(sets_3d, sets_2d, path):
    """P(q) against system size.

    The question these panels pose: as L grows, does the weight between the
    peaks survive (many states, RSB-like) or drain away (one state plus its
    flip, droplet-like)?  At these sizes the answer is not decidable -- which
    is the honest result, and the reason the debate is still open.
    """
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))

    specs = [(sets_3d, 0.55, "-", "3D, well below $T_c$"),
             (sets_3d, 0.95, "-", "3D, at $T_c$"),
             (sets_2d, 0.55, "--", "2D control, no glass phase")]
    for a, (sets, t_target, style, title) in zip(ax, specs):
        for data in sets:
            _, c, p = pq_curves(data, [t_target])[0]
            a.plot(c, p, style, lw=1.6, label=f"L = {int(data['L'])}")
        a.set_title(f"{title}\nT = {t_target:.2f}", fontsize=10)

    for a in ax:
        a.set_xlabel("q")
        a.set_ylabel("P(q)")
        if a.lines:
            a.legend(fontsize=8)
        else:
            a.text(0.5, 0.5, "not yet available", ha="center", va="center",
                   transform=a.transAxes, color="0.5")
    fig.suptitle("Step 7: does the weight between the peaks survive as L grows?")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def equilibration(sets_3d, path):
    """The evidence that the low-temperature points can be believed."""
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.7))

    for data in sets_3d:
        temps = data["temps"]
        mid = 0.5 * (temps[1:] + temps[:-1])
        ax[0].plot(mid, data["swap"].mean(axis=0), "o-", ms=3.5,
                   label=f"L = {int(data['L'])}")
    ax[0].axhspan(0.2, 0.4, color="C2", alpha=0.15)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("T (pair midpoint)")
    ax[0].set_ylabel("swap acceptance")
    ax[0].set_title("5.1  no dead rungs", fontsize=9)
    ax[0].legend(fontsize=7)

    ax[1].bar([str(int(d["L"])) for d in sets_3d],
              [d["trips"].mean() for d in sets_3d], color="C0", width=0.5)
    ax[1].set_xlabel("L")
    ax[1].set_ylabel("round trips per replica")
    ax[1].set_title("5.2  replicas traverse the ladder", fontsize=9)

    for data in sets_3d:
        temps = data["temps"]
        direct = data["q_link"].mean(axis=0)
        from_e = 1.0 + data["e"].mean(axis=0) * temps / int(data["d"])
        ax[2].plot(temps, direct, "o-", ms=3.5, label=f"L={int(data['L'])} measured")
        ax[2].plot(temps, from_e, "--", lw=1, color=ax[2].lines[-1].get_color())
    ax[2].set_xlabel("T")
    ax[2].set_ylabel(r"$q_\mathrm{link}$")
    ax[2].set_title("6.1  solid = replicas, dashed = $1 + uT/d$", fontsize=9)
    ax[2].legend(fontsize=7)

    for data in sets_3d:
        temps = data["temps"]
        drift = data["q2_half"].mean(axis=0) - data["q2_quarter"].mean(axis=0)
        ax[3].plot(temps, drift, "o-", ms=3.5, label=f"L = {int(data['L'])}")
    ax[3].axhline(0.0, color="k", lw=0.8)
    ax[3].set_xlabel("T")
    ax[3].set_ylabel(r"$\langle q^2\rangle$ half $-$ quarter")
    ax[3].set_title("6.2  log-binning drift (want zero)", fontsize=9)
    ax[3].legend(fontsize=7)

    fig.suptitle("Step 7: equilibration evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


if __name__ == "__main__":
    print("=" * 76)
    print("STEP 7 ANALYSIS")
    print("=" * 76 + "\n")

    sets_3d = load_available(3, (4, 6, 8))
    sets_2d = load_available(2, (8, 12, 16))
    assert sets_3d, "no 3D data yet -- run step7_production.py first"
    if len(sets_3d) < 2:
        print("  NOTE: only one 3D size present, so no crossing can be located."
              "\n        The panels below show the pipeline, not the result.\n")

    cross, cross_b = main_results(sets_3d, sets_2d, "step7_main_results.png")
    pq_by_size(sets_3d, sets_2d, "step7_pq_by_size.png")
    equilibration(sets_3d, "step7_equilibration.png")

    print()
    print("3D crossings from xi_L/L :", [f"{c:.3f}" for c in cross])
    print("3D crossings from Binder :", [f"{c:.3f}" for c in cross_b])
    if cross:
        print(f"  T_c estimate (xi)     = {np.mean(cross):.3f}"
              f"   literature {TC_3D_REF}")
    if cross_b:
        print(f"  T_c estimate (Binder) = {np.mean(cross_b):.3f}")

    for data in sets_2d:
        y = xi_over_L(data)[0]
        print(f"2D L={int(data['L'])}: xi_L/L ranges "
              f"{y.min():.3f} to {y.max():.3f}")
    pairs = [crossing(sets_2d[i]["temps"], xi_over_L(sets_2d[i])[0],
                      xi_over_L(sets_2d[i + 1])[0])
             for i in range(len(sets_2d) - 1)]
    print("2D crossings (expect all None):", pairs)
    print("=" * 76)
