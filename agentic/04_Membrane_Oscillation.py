"""Membrane oscillation -- Steps 1 and 2.

A circular aluminum-foil membrane, clamped at its rim.

  Step 1  geometry, material parameters, discrete polar Laplacian.
  Step 2  the Bessel solution: normal modes, and the steady-state response to a
          plane sound wave.  Produces two figures.

Requires scipy (installed 2026-10-02) for J_m and its zeros.

UNITS: SI throughout (metres, kilograms, seconds, pascals).  This departs from the
natural units (hbar = m = omega = 1) used elsewhere in this course, deliberately:
the problem names a real material, so real numbers are the whole point -- a foil
disk should come out ringing at an audible pitch, and that is checkable.

The governing equation, for reference:

    sigma * w_tt = T * laplacian(w)      =>      c = sqrt(T / sigma)

  w(r, theta, t)  transverse displacement        [m]
  sigma = rho * h areal density                 [kg/m^2]
  T              tension per unit length        [N/m]
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jn_zeros, jv

# ---------------------------------------------------------------------------
# Material and setup.
#
# rho and h come from the problem statement ("aluminum", "standard foil").
# RADIUS and PRESTRESS do NOT -- the prompt never gives them, and together they
# set every frequency in this project, so they are the two numbers to argue with.
#
# PRESTRESS is the in-plane stress we stretch the foil to in its frame.  Tension
# per unit length is then T = PRESTRESS * h.  Tension is not a material property:
# it is how hard you pulled.  10 MPa is a modest fraction of foil's yield stress
# (tens of MPa), so the foil stays elastic and does not tear.
# ---------------------------------------------------------------------------
RHO_ALUMINUM = 2700.0     # density of aluminum                    [kg/m^3]
H_FOIL = 16e-6            # standard household foil thickness      [m]
PRESTRESS = 10e6          # applied in-plane stress                [Pa]
RADIUS = 0.10             # membrane radius                        [m]

# First zero of the Bessel function J_0.  Hardcoded because scipy is not installed;
# step 2 computes it from scratch and must reproduce this value.
J01 = 2.404825557695773


class PolarDisk:
    """A clamped circular membrane on a polar grid.

    Fields live on an array W of shape (n_r + 1, n_theta):

        row 0        r = 0, the centre.  All n_theta entries hold the same value --
                     they are one physical point.
        rows 1..n_r-1  interior
        row n_r      r = a, the clamped rim.  Held at 0, never updated.

    theta wraps around, which is why the angular derivative below is a np.roll
    and needs no boundary treatment at all.

    Steps 2-4 only ever use `laplacian`, `quad_weights` and `apply_bc`.  Other
    shapes (square, L-shape) will supply the same three on a Cartesian mask, so
    nothing downstream has to know these are polar coordinates.
    """

    def __init__(self, radius=RADIUS, n_r=64, n_theta=128,
                 prestress=PRESTRESS, thickness=H_FOIL, density=RHO_ALUMINUM):
        self.a = radius
        self.n_r, self.n_theta = n_r, n_theta

        self.r = np.linspace(0.0, radius, n_r + 1)          # r = a IS a node
        self.theta = np.arange(n_theta) * (2 * np.pi / n_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = 2 * np.pi / n_theta
        self.R, self.THETA = np.meshgrid(self.r, self.theta, indexing="ij")
        self.X, self.Y = self.R * np.cos(self.THETA), self.R * np.sin(self.THETA)

        # Material -> wave speed.
        self.h = thickness
        self.sigma = density * thickness        # areal density       [kg/m^2]
        self.tension = prestress * thickness    # per unit length     [N/m]
        self.c = np.sqrt(self.tension / self.sigma)

        # Area element r*dr*dtheta, trapezoidal in r (halved at both ends),
        # periodic in theta (no halving).  Summing f * quad_weights integrates
        # f over the disk.
        w_r = np.full(n_r + 1, self.dr)
        w_r[0] = w_r[-1] = 0.5 * self.dr
        self.quad_weights = (self.r * w_r)[:, None] * self.dtheta * np.ones(n_theta)

    def zeros(self):
        return np.zeros((self.n_r + 1, self.n_theta))

    def apply_bc(self, W):
        """Clamp the rim and keep the centre single-valued.  Modifies W in place."""
        W[-1, :] = 0.0
        W[0, :] = W[0, :].mean()
        return W

    def laplacian(self, W):
        """Discrete polar Laplacian, second-order accurate.

            lap(w) = w_rr + w_r / r + w_tt / r^2

        The 1/r and 1/r^2 blow up at the centre, so row 0 uses the regular-origin
        formula instead: for a smooth field, lap(w)|_0 = 4*(<w on ring 1> - w_0)/dr^2.
        Only the angular average of ring 1 survives, which is correct -- every
        m > 0 angular component contributes nothing to the Laplacian at r = 0.

        Returns 0 on the clamped rim, where the operator is not defined.
        """
        L = self.zeros()
        r = self.r[1:-1, None]
        Wm, W0, Wp = W[:-2], W[1:-1], W[2:]

        d2_dr2 = (Wp - 2 * W0 + Wm) / self.dr ** 2
        d1_dr = (Wp - Wm) / (2 * self.dr) / r
        d2_dtheta2 = (np.roll(W0, -1, axis=1) - 2 * W0
                      + np.roll(W0, 1, axis=1)) / (r * self.dtheta) ** 2

        L[1:-1] = d2_dr2 + d1_dr + d2_dtheta2
        L[0, :] = 4.0 * (W[1].mean() - W[0, 0]) / self.dr ** 2
        return L

    def integrate(self, F):
        return np.sum(F * self.quad_weights)

    def rayleigh(self, W):
        """Rayleigh quotient -int w*lap(w) dA / int w^2 dA.

        For any trial shape vanishing on the rim this is an upper bound on the
        lowest eigenvalue of -lap, by the variational principle.  One-sided, so
        it is a check that cannot be satisfied by accident.
        """
        return -self.integrate(W * self.laplacian(W)) / self.integrate(W ** 2)


# ---------------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------------

def check_measure(d):
    """1. The r*dr*dtheta Jacobian.  Drop the r and this is off by ~2x."""
    area = d.integrate(np.ones_like(d.R))
    mass = d.sigma * area
    print("1. area      = %.12e m^2   (exact pi*a^2 = %.12e)" % (area, np.pi * d.a ** 2))
    print("   mass      = %.4f g   foil disk, r = %.0f cm" % (1e3 * mass, 100 * d.a))
    assert abs(area / (np.pi * d.a ** 2) - 1.0) < 1e-13, "quadrature weights are wrong"


def check_wave_speed(d):
    """2. c = sqrt(T/sigma) = sqrt(stress/rho): the thickness cancels exactly.

    Two routes to the same number through different parameters.  Confusing stress
    (Pa) with tension per unit length (N/m) -- the easiest mistake in membrane
    problems, they differ by a factor of h = 16 microns -- breaks this.
    """
    c_alt = np.sqrt(PRESTRESS / RHO_ALUMINUM)
    f1 = d.c * J01 / (2 * np.pi * d.a)      # previewing step 2, not solving it
    print("2. c         = %.6f m/s  (sqrt(stress/rho) = %.6f, h cancels)" % (d.c, c_alt))
    print("   f_00      = %.1f Hz   audible, so a sound wave can drive it" % f1)
    assert abs(d.c - c_alt) < 1e-12, "stress and tension-per-length are being confused"
    assert 20.0 < f1 < 20000.0, "fundamental is outside the audible band"


def check_laplacian_exact(d):
    """3. w = 1 - (r/a)^2 has lap(w) = -4/a^2 everywhere, including at r = 0.

    A quadratic is differentiated exactly by a second-order stencil, so the error
    here is machine noise, not truncation.  Sharpest possible test of the stencil.
    """
    W = d.apply_bc(1.0 - (d.R / d.a) ** 2)
    L = d.laplacian(W)[:-1]                  # rim excluded: operator undefined there
    err = np.max(np.abs(L + 4.0 / d.a ** 2)) / (4.0 / d.a ** 2)
    print("3. max rel. error on lap(1-(r/a)^2) = %.2e   (origin node included)" % err)
    assert err < 1e-10, "radial stencil or origin formula is wrong"


def check_laplacian_harmonic(d_coarse, d_fine):
    """4. w = r^2*cos(2*theta) is harmonic: lap(w) = 0 exactly.

    This one needs the angular term and the 1/r^2 to be right, and they must
    cancel the two radial terms to zero.  cos(2*theta) is not a polynomial in
    theta, so the residual is truncation error, O(dtheta^2) -- doubling n_theta
    must quarter it.  That convergence rate is the real assertion.
    """
    def residual(d):
        W = (d.R ** 2) * np.cos(2 * d.THETA)
        W[0, :] = 0.0
        return np.max(np.abs(d.laplacian(W)[:-1]))

    r_coarse, r_fine = residual(d_coarse), residual(d_fine)
    print("4. |lap(r^2 cos2t)|: n_theta=%d -> %.3e,  n_theta=%d -> %.3e,  ratio %.2f"
          % (d_coarse.n_theta, r_coarse, d_fine.n_theta, r_fine, r_coarse / r_fine))
    assert 3.6 < r_coarse / r_fine < 4.4, "angular term is not second-order accurate"


def check_variational_bound(d):
    """5. The variational bound -- the one check that ties everything together.

    The trial shape w = 1 - (r/a)^2 vanishes on the rim, so its Rayleigh quotient
    must lie ABOVE the true lowest eigenvalue (j_01/a)^2, and never below.  By hand,
    int|grad w|^2 / int w^2 = 6/a^2, versus the exact 5.783/a^2 -- high by 3.8%,
    as a bound should be.  It exercises the Laplacian, the Jacobian and the
    boundary condition at once, and being one-sided it cannot be passed by luck.
    """
    W = d.apply_bc(1.0 - (d.R / d.a) ** 2)
    lam = d.rayleigh(W)
    lam_exact = (J01 / d.a) ** 2
    print("5. Rayleigh   = %.6f /m^2   (hand value 6/a^2 = %.6f)" % (lam, 6.0 / d.a ** 2))
    print("   exact      = %.6f /m^2   bound is high by %.2f%%"
          % (lam_exact, 100 * (lam / lam_exact - 1)))
    assert lam > lam_exact, "a variational bound below the true eigenvalue is impossible"
    assert abs(lam - 6.0 / d.a ** 2) / (6.0 / d.a ** 2) < 1e-3, "does not match 6/a^2"


# ===========================================================================
# STEP 2 -- the Bessel solution
#
# Separating sigma*w_tt = T*lap(w) with w = 0 on the rim gives
#
#     w_mn(r, theta) = J_m(j_mn * r/a) * cos(m*theta),    f_mn = c*j_mn/(2*pi*a)
#
# where j_mn is the n-th zero of J_m.  m counts nodal diameters, n-1 counts
# interior nodal circles.  The frequencies are NOT harmonics of each other --
# the ratios are ratios of Bessel zeros, which is why a drum is not a guitar.
# ===========================================================================

N_DRIVE_MODES = 40        # J_0 modes kept in the driven sum
Q_FACTOR = 50.0           # quality factor at the fundamental (air-loaded foil)


def mode_table(n_m=4, n_rad=3):
    """The lowest modes as (m, n, j_mn), sorted by frequency."""
    rows = [(m, n, j) for m in range(n_m)
            for n, j in enumerate(jn_zeros(m, n_rad), start=1)]
    return sorted(rows, key=lambda row: row[2])


def mode_frequency(d, j):
    """Frequency in Hz of the mode with Bessel zero j."""
    return d.c * j / (2 * np.pi * d.a)


def mode_shape(d, m, n):
    """Normal mode J_m(j_mn r/a)*cos(m*theta), scaled to max |w| = 1."""
    j = jn_zeros(m, n)[-1]
    W = jv(m, j * d.R / d.a) * np.cos(m * d.THETA)
    return d.apply_bc(W / np.max(np.abs(W)))


def modal_amplitudes(d, freq_hz, pressure_pa, q_factor=Q_FACTOR,
                     n_modes=N_DRIVE_MODES):
    """Complex steady-state amplitude of each J_0 mode under a uniform drive.

    Project sigma*w_tt + gamma*w_t = T*lap(w) + p0*cos(wt) onto mode n:

        q_n'' + (gamma/sigma) q_n' + omega_n^2 q_n = (p0/sigma) * P_n * cos(wt)

    with P_n = <phi_n, 1>/<phi_n, phi_n> = 2/(j_n J_1(j_n)) the overlap of a
    uniform pressure with mode n.  The steady state is then one complex divide
    per mode -- no time stepping needed.

    gamma is set from the requested Q at the fundamental: Q = omega_1*sigma/gamma.
    Without damping, driving exactly on a resonance diverges and the heat map
    would be meaningless.
    """
    j = jn_zeros(0, n_modes)
    omega_n = d.c * j / d.a
    omega = 2 * np.pi * freq_hz
    gamma = omega_n[0] * d.sigma / q_factor
    proj = 2.0 / (j * jv(1, j))
    drive = pressure_pa * proj / d.sigma
    return j, drive / (omega_n ** 2 - omega ** 2 + 1j * gamma * omega / d.sigma)


def complex_field(d, freq_hz, pressure_pa=1.0, **kw):
    """Complex amplitude C(r); the physical field is Re(C * exp(i*omega*t)).

    |C| is the envelope the membrane sweeps out over a cycle, and arg(C) is the
    phase lag behind the sound wave.

    At 233 Hz a sound wave in air is 1.5 m long, far bigger than the 20 cm
    membrane, so the pressure is uniform across the face.  A uniform load has
    exactly zero overlap with every m > 0 mode, so only the axisymmetric J_0
    family responds: the driven field is circular at every frequency, never
    lopsided.  Breaking that needs oblique incidence (step 4).
    """
    j, amp = modal_amplitudes(d, freq_hz, pressure_pa, **kw)
    C = np.zeros_like(d.R, dtype=complex)
    for jn, an in zip(j, amp):
        C += an * jv(0, jn * d.R / d.a)
    return C


def driven_displacement(d, t, freq_hz, pressure_pa=1.0, **kw):
    """Displacement field [m] at time t under a plane sound wave.

    freq_hz is the pitch, pressure_pa the volume (1 Pa ~ 94 dB SPL).
    """
    C = complex_field(d, freq_hz, pressure_pa, **kw)
    return d.apply_bc(np.real(C * np.exp(1j * 2 * np.pi * freq_hz * t)))


def snapshot_time(d, freq_hz, pressure_pa=1.0, **kw):
    """The instant in a cycle at which the membrane is most deflected.

    Worth its own function because of a trap.  A driven oscillator on resonance
    responds in QUADRATURE -- 90 degrees behind the drive -- so at t = 0 the
    resonant mode is passing through zero displacement at maximum speed.  A heat
    map snapshotted at t = 0 therefore shows resonance as an almost flat
    membrane, which looks exactly like a broken solver.  The fix is not to
    plot the envelope (that throws away the sign structure) but to snapshot at
    the time the dominant point reaches its extreme, which this returns.
    """
    if freq_hz == 0.0:
        return 0.0
    C = complex_field(d, freq_hz, pressure_pa, **kw)
    peak = np.unravel_index(np.argmax(np.abs(C)), C.shape)
    return -np.angle(C[peak]) / (2 * np.pi * freq_hz)


def center_amplitude(d, freq_hz, pressure_pa=1.0, **kw):
    """Peak displacement [m] at the centre, where J_0 = 1 for every mode."""
    _, amp = modal_amplitudes(d, freq_hz, pressure_pa, **kw)
    return np.abs(np.sum(amp))


def static_deflection(d, pressure_pa=1.0):
    """Exact zero-frequency shape: T*lap(w) = -p0 gives w = p0(a^2-r^2)/(4T)."""
    return pressure_pa * (d.a ** 2 - d.R ** 2) / (4 * d.tension)


def radial_eigenvalues(d, n_want=4):
    """Lowest axisymmetric eigenvalues of -lap, from step 1's stencil alone.

    Assembles the m = 0 radial operator as a matrix and diagonalizes it.  There
    is no Bessel function anywhere in this routine, so when sqrt(lambda)*a comes
    back as 2.4048, 5.5201, ... that is the grid independently rediscovering the
    zeros of J_0.  This matrix is also exactly the machinery step 3 needs, where
    nonuniform material removes the analytic route entirely.
    """
    n = d.n_r                         # unknowns are rows 0..n_r-1; the rim is clamped
    A = np.zeros((n, n))
    A[0, 0], A[0, 1] = -4.0 / d.dr ** 2, 4.0 / d.dr ** 2   # regular-origin formula
    for i in range(1, n):
        A[i, i - 1] = 1.0 / d.dr ** 2 - 1.0 / (2 * d.dr * d.r[i])
        A[i, i] = -2.0 / d.dr ** 2
        if i + 1 < n:                 # the dropped column is the clamped rim, u = 0
            A[i, i + 1] = 1.0 / d.dr ** 2 + 1.0 / (2 * d.dr * d.r[i])
    return np.sort(np.linalg.eigvals(-A).real)[:n_want]


# ---------------------------------------------------------------------------
# CHECKS -- step 2
# ---------------------------------------------------------------------------

def _mode_residual(d, m, n, j):
    """lap(mode) + (j/a)^2 * mode, which is zero for the exact solution."""
    W = mode_shape(d, m, n)
    res = np.zeros_like(W)
    res[:-1] = d.laplacian(W)[:-1] + (j / d.a) ** 2 * W[:-1]
    return res, (j / d.a) ** 2 * W


def check_modes_are_eigenfunctions(d_coarse, d_fine):
    """6. Step 1's stencil applied to a Bessel mode must give -(j_mn/a)^2 times it.

    This is the join between the two steps: the analytic solution and the discrete
    operator have to describe the same membrane.  J_m is not a polynomial, so the
    error is genuine truncation and the assertion is on the convergence rate -- both
    grid directions are refined together, so second order means a factor of 4.

    Two norms, because they say different things:

      L2, area-weighted -- second order for every mode.  This is the norm that
      matters: it is what controls the eigenvalue error, and errors are weighted by
      the area they occupy.

      max, pointwise -- second order for the m = 0 modes, but only FIRST order for
      m > 0.  That is not a bug and not an excuse.  The whole discrepancy lives on
      the single ring at r = dr next to the origin, where the 1/r^2 in the angular
      term amplifies angular truncation error.  For a mode going like r*cos(theta)
      the radial and angular terms must cancel exactly, and what survives is
      ~dtheta^2/(12*dr), which only halves when both spacings halve.  So the third
      assertion below demands that ratio be 2, not 4: it is a test of the
      explanation, and would fail if the mechanism were something else.  The ring
      carries area ~dr^2, which is why L2 never notices it.
    """
    def norms(d, rows):
        worst_l2, worst_max = 0.0, 0.0
        for m, n, j in rows:
            res, ref = _mode_residual(d, m, n, j)
            worst_l2 = max(worst_l2, np.sqrt(d.integrate(res ** 2)
                                             / d.integrate(ref ** 2)))
            worst_max = max(worst_max, np.max(np.abs(res)) / (j / d.a) ** 2)
        return worst_l2, worst_max

    rows = mode_table(3, 2)
    axisym = [r for r in rows if r[0] == 0]
    diam = [r for r in rows if r[0] > 0]
    label = "n_r %d->%d, n_theta %d->%d" % (d_coarse.n_r, d_fine.n_r,
                                            d_coarse.n_theta, d_fine.n_theta)

    l2_c, _ = norms(d_coarse, rows)
    l2_f, _ = norms(d_fine, rows)
    _, mx0_c = norms(d_coarse, axisym)
    _, mx0_f = norms(d_fine, axisym)
    _, mxd_c = norms(d_coarse, diam)
    _, mxd_f = norms(d_fine, diam)

    print("6. residual of lap(mode) + (j/a)^2*mode,  %s" % label)
    print("   L2  all modes : %.3e -> %.3e   ratio %.2f  (second order)"
          % (l2_c, l2_f, l2_c / l2_f))
    print("   max m = 0     : %.3e -> %.3e   ratio %.2f  (second order)"
          % (mx0_c, mx0_f, mx0_c / mx0_f))
    print("   max m > 0     : %.3e -> %.3e   ratio %.2f  (first order, at r = dr)"
          % (mxd_c, mxd_f, mxd_c / mxd_f))
    assert 3.5 < l2_c / l2_f < 4.5, "stencil is not second order in the energy norm"
    assert 3.5 < mx0_c / mx0_f < 4.5, "stencil is not second order on the J_0 modes"
    assert 1.7 < mxd_c / mxd_f < 2.3, \
        "m>0 pointwise error is not the expected dtheta^2/dr behaviour at the origin"


def check_orthogonality(d_coarse, d_fine):
    """7. Distinct modes are orthogonal -- which is what licenses the modal sum.

    Orthogonality is what makes the driven solver legitimate: it is the reason a
    uniform pressure decomposes into independent one-line solves per mode instead
    of a coupled system.  It fails in two quite different ways, so they are
    measured separately:

      different m -- orthogonal to MACHINE precision.  sum_j cos(m*theta_j)
      cos(m'*theta_j) over a uniform theta grid is exactly zero for m != m' (the
      discrete trig identity), not merely small.  No quadrature error can enter.

      same m, different n -- orthogonal only to QUADRATURE accuracy.  The radial
      integral of J_m(j_mn r/a) J_m(j_mn' r/a) r dr vanishes analytically, but
      trapezoid gets it to O(dr^2).  2e-4 on a 64-point grid is the quadrature,
      not a broken mode, so the claim to assert is the convergence rate.

    The same O(dr^2) sets the error in <phi_0n,phi_0n> vs pi*a^2*J_1(j_0n)^2.
    """
    def measure(d):
        rows = mode_table(3, 2)
        shapes = {(m, n): mode_shape(d, m, n) for m, n, _ in rows}
        keys = list(shapes)
        same_m, diff_m = 0.0, 0.0
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                u, v = shapes[k1], shapes[k2]
                o = abs(d.integrate(u * v)) / np.sqrt(d.integrate(u ** 2)
                                                      * d.integrate(v ** 2))
                if k1[0] == k2[0]:
                    same_m = max(same_m, o)
                else:
                    diff_m = max(diff_m, o)
        j = jn_zeros(0, 3)
        norms = np.array([d.integrate(jv(0, jn * d.R / d.a) ** 2) for jn in j])
        norm_err = np.max(np.abs(norms / (np.pi * d.a ** 2 * jv(1, j) ** 2) - 1.0))
        return same_m, diff_m, norm_err

    same_c, diff_c, nerr_c = measure(d_coarse)
    same_f, diff_f, nerr_f = measure(d_fine)
    print("7. overlap of distinct modes,  n_r %d->%d" % (d_coarse.n_r, d_fine.n_r))
    print("   different m   : %.2e -> %.2e   (exact discrete trig identity)"
          % (diff_c, diff_f))
    print("   same m        : %.2e -> %.2e   ratio %.2f  (trapezoid, O(dr^2))"
          % (same_c, same_f, same_c / same_f))
    print("   <phi,phi> err : %.2e -> %.2e   ratio %.2f  vs pi*a^2*J_1(j)^2"
          % (nerr_c, nerr_f, nerr_c / nerr_f))
    assert diff_f < 1e-12, "modes of different m must be orthogonal exactly"
    assert 3.5 < same_c / same_f < 4.5, "radial orthogonality is not converging"
    assert 3.5 < nerr_c / nerr_f < 4.5, "mode norm is not converging"


def check_uniform_drive_symmetry(d):
    """8. A uniform pressure has zero overlap with every m > 0 mode.

    cos(m*theta) integrates to zero over a full turn for m > 0, so a plane wave
    at normal incidence cannot excite a mode with a nodal diameter no matter how
    loud it is.  If this failed, the driven heat maps would be lopsided.
    """
    overlaps = {(m, n): d.integrate(mode_shape(d, m, n)) / (np.pi * d.a ** 2)
                for m, n, _ in mode_table(4, 2) if m > 0}
    worst = max(abs(v) for v in overlaps.values())
    print("8. max <1, mode_mn>/area over all m>0 modes = %.2e" % worst)
    assert worst < 1e-12, "a uniform drive is leaking into an m>0 mode"


def check_static_limit(d, pressure_pa=1.0):
    """9. At zero frequency the Bessel sum must collapse to a parabola.

    T*lap(w) = -p0 has the closed form w = p0(a^2-r^2)/(4T), so the modal sum
    sum_n 2*J_0(j_n r/a)/(j_n^3 J_1(j_n)) * p0 a^2/T has to reproduce it -- which
    at r = 0 is the identity sum 1/(j_n^3 J_1(j_n)) = 1/8.  This validates the
    driving projection P_n, the mode frequencies, and the normalization at once,
    against an answer that involves no Bessel functions at all.
    """
    W = driven_displacement(d, 0.0, 0.0, pressure_pa)
    exact = static_deflection(d, pressure_pa)
    err = np.max(np.abs(W - exact)) / np.max(exact)
    print("9. centre deflection at DC = %.4f um   (exact p0*a^2/4T = %.4f um)"
          % (1e6 * W[0, 0], 1e6 * exact[0, 0]))
    print("   max rel. error vs the parabola, %d modes = %.2e" % (N_DRIVE_MODES, err))
    assert err < 1e-3, "the modal sum does not reproduce the static parabola"


def check_resonance(d, pressure_pa=1.0):
    """10. On resonance, mode 1 is amplified by exactly Q, and lags by 90 degrees.

    Two separate claims.  The gain pins down the damping convention and is exact
    rather than asymptotic: |q_1(omega_1)|/|q_1(0)| = omega_1*sigma/gamma = Q.

    The phase is the one that bites.  A resonant oscillator responds in
    quadrature, so at t = 0 the resonant mode is at zero displacement and the
    membrane looks nearly flat -- the t = 0 snapshot understates the true
    excursion by a factor of ~Q.  Asserting that ratio is small, and that the
    snapshot at the peak time recovers the full envelope, is what stops a
    correct solver from being read as a broken one.

    The slope is reported because linear membrane theory needs |grad w| << 1.
    That -- not w << thickness -- is the assumption actually in play.
    """
    f1 = mode_frequency(d, jn_zeros(0, 1)[0])
    _, amp_dc = modal_amplitudes(d, 0.0, pressure_pa)
    _, amp_res = modal_amplitudes(d, f1, pressure_pa)
    gain = abs(amp_res[0]) / abs(amp_dc[0])

    C = complex_field(d, f1, pressure_pa)
    envelope = np.max(np.abs(C))
    at_zero = np.max(np.abs(driven_displacement(d, 0.0, f1, pressure_pa)))
    t_star = snapshot_time(d, f1, pressure_pa)
    at_peak = np.max(np.abs(driven_displacement(d, t_star, f1, pressure_pa)))
    slope = envelope * jn_zeros(0, 1)[0] / d.a

    print("10. f_01 = %.2f Hz,  mode-1 gain on resonance = %.6f  (Q = %.1f)"
          % (f1, gain, Q_FACTOR))
    print("    envelope peak at %.3g Pa = %.3f mm,  max slope %.4f (<<1, linear OK)"
          % (pressure_pa, 1e3 * envelope, slope))
    print("    snapshot at t=0 reaches only %.3f of that (90-deg lag);"
          " at t=T/4 it reaches %.6f" % (at_zero / envelope, at_peak / envelope))
    assert abs(gain - Q_FACTOR) / Q_FACTOR < 1e-12, "damping is not giving the stated Q"
    assert at_zero / envelope < 0.05, "resonant response is not in quadrature"
    assert abs(at_peak / envelope - 1.0) < 1e-12, "peak-time snapshot misses the envelope"
    # The lag is exactly 90 deg for mode 1 alone; the off-resonant modes in the
    # sum shift it by O(1/Q), so the tolerance is 1/Q and not machine precision.
    assert abs(2 * np.pi * f1 * t_star - np.pi / 2) < 3.0 / Q_FACTOR, \
        "the resonant phase lag is not 90 degrees"
    assert slope < 0.1, "deflection is too large for linear membrane theory"


def check_numerical_eigenvalues(d):
    """11. The grid finds the zeros of J_0 without being told about Bessel.

    Diagonalizing the m = 0 radial operator and reading off sqrt(lambda)*a must
    reproduce 2.4048, 5.5201, 8.6537, 11.7915.  Independent confirmation that
    the analytic solution and the numerics describe the same membrane.
    """
    lam = radial_eigenvalues(d, 4)
    found, exact = np.sqrt(lam) * d.a, jn_zeros(0, 4)
    print("11. zeros of J_0 from the grid:  " +
          "  ".join("%.4f" % v for v in found))
    print("    exact:                       " +
          "  ".join("%.4f" % v for v in exact))
    err = np.max(np.abs(found / exact - 1.0))
    print("    max relative error = %.2e  (O(dr^2) truncation)" % err)
    assert err < 1e-2, "the discrete operator has the wrong spectrum"


# ---------------------------------------------------------------------------
# FIGURES
#
# Displacement is signed, so both heat maps use a diverging map with limits
# symmetric about zero -- that puts undisplaced membrane at the neutral midpoint
# and makes up and down read as opposite colors.  A sequential map would imply
# zero is an extreme, and a rainbow map would invent structure that isn't there.
# Nodal lines are drawn as contours so the mode structure survives in greyscale.
# ---------------------------------------------------------------------------

def _closed(d, W):
    """Repeat the theta = 0 column at theta = 2*pi so the disk has no seam."""
    return np.c_[d.X, d.X[:, :1]], np.c_[d.Y, d.Y[:, :1]], np.c_[W, W[:, :1]]


def _bare_disk(ax, d):
    """Strip the axes and draw the rim explicitly.

    The rim has to be drawn by hand rather than left to the zero-contour: the
    displacement is exactly 0 all along it, so a contour at level 0 wanders in
    and out of the boundary row and renders the rim as broken arcs.  Nodal lines
    are therefore contoured on the interior only.
    """
    ang = np.linspace(0.0, 2 * np.pi, 400)
    ax.plot(d.a * np.cos(ang), d.a * np.sin(ang), color="0.3", lw=1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def _heatmap(ax, d, W, lim):
    """Diverging heat map of a signed field, plus its interior nodal lines."""
    X, Y, Wc = _closed(d, W)
    mesh = ax.pcolormesh(X, Y, Wc, cmap="RdBu_r", vmin=-lim, vmax=lim,
                         shading="gouraud")
    ax.contour(X[:-1], Y[:-1], Wc[:-1], levels=[0.0], colors="0.3", linewidths=1.0)
    _bare_disk(ax, d)
    return mesh


def plot_modes(d, path):
    """Heat maps of the six lowest normal modes."""
    rows = mode_table(4, 3)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), constrained_layout=True)
    for ax, (m, n, j) in zip(axes.ravel(), rows):
        mesh = _heatmap(ax, d, mode_shape(d, m, n), 1.0)
        ax.set_title("(m,n) = (%d,%d)   %.1f Hz\n%d nodal diameter(s), "
                     "%d interior circle(s)"
                     % (m, n, mode_frequency(d, j), m, n - 1), fontsize=9)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.75, ticks=[-1, 0, 1])
    cbar.set_label("displacement / peak")
    fig.suptitle("Normal modes of a clamped aluminum-foil disk "
                 "(a = %.0f cm, c = %.1f m/s)" % (100 * d.a, d.c))
    fig.savefig(path, dpi=140)
    print("   wrote %s" % path)


def plot_driven(d, path, pressure_pa=1.0):
    """Frequency response, and driven heat maps at three frequencies."""
    f1, f2 = (mode_frequency(d, j) for j in jn_zeros(0, 2))
    freqs = np.linspace(20.0, 1200.0, 2000)
    resp = np.array([center_amplitude(d, f, pressure_pa) for f in freqs])
    dc = static_deflection(d, pressure_pa)[0, 0]

    fig = plt.figure(figsize=(10.5, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15])

    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(freqs, 1e6 * resp, color="#20618f", lw=2)
    ax.axhline(1e6 * dc, ls="--", lw=1, color="0.45")
    ax.annotate("static limit, $p_0a^2/4T$ = %.1f $\\mu$m" % (1e6 * dc),
                (freqs[0], 1e6 * dc), xytext=(8, 7), textcoords="offset points",
                ha="left", fontsize=9, color="0.35")
    for k, fk in enumerate(mode_frequency(d, jn_zeros(0, 5))):
        if fk >= freqs[-1]:
            continue
        ax.axvline(fk, lw=1, ls=":", color="0.65")
        near_edge = fk > 0.88 * freqs[-1]
        ax.annotate("$f_{0%d}$ = %.0f Hz" % (k + 1, fk), (fk, ax.get_ylim()[1]),
                    xytext=(-4 if near_edge else 4, -12), textcoords="offset points",
                    ha="right" if near_edge else "left", fontsize=9, color="0.35")
    ax.set_xlabel("drive frequency (Hz)")
    ax.set_ylabel("centre amplitude ($\\mu$m)")
    ax.set_title("Response to a %.3g Pa plane wave (Q = %.0f at $f_{01}$) -- "
                 "only the m = 0 modes answer a uniform drive" % (pressure_pa, Q_FACTOR))
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlim(freqs[0], freqs[-1])

    # Each panel is snapshotted at its own peak time, not at t = 0.  On resonance
    # the response is 90 degrees behind the drive, so a t = 0 snapshot would show
    # a nearly flat membrane and badly misrepresent the physics.
    for col, (f, label) in enumerate([(100.0, "100 Hz, below resonance"),
                                      (f1, "%.0f Hz, $f_{01}$" % f1),
                                      (f2, "%.0f Hz, $f_{02}$" % f2)]):
        axk = fig.add_subplot(gs[1, col])
        t_star = snapshot_time(d, f, pressure_pa)
        W = driven_displacement(d, t_star, f, pressure_pa)
        lim = np.max(np.abs(W))
        _heatmap(axk, d, W, lim)
        axk.set_title("%s\npeak %.1f $\\mu$m" % (label, 1e6 * lim), fontsize=10)

    fig.text(0.5, 0.005, "Each disk is at its own peak instant and on its own "
             "colour scale -- the peaks differ by 40x, so a shared scale would "
             "flatten two of the three.", ha="center", fontsize=8.5, color="0.4")
    fig.savefig(path, dpi=140)
    print("   wrote %s" % path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    disk = PolarDisk()
    print("grid: n_r = %d, n_theta = %d, dr = %.3e m, dtheta = %.3e rad\n"
          % (disk.n_r, disk.n_theta, disk.dr, disk.dtheta))

    print("--- step 1: the system ---")
    check_measure(disk)
    check_wave_speed(disk)
    check_laplacian_exact(disk)
    check_laplacian_harmonic(PolarDisk(n_theta=64), PolarDisk(n_theta=128))
    check_variational_bound(disk)

    print("\n--- step 2: the Bessel solution ---")
    check_modes_are_eigenfunctions(PolarDisk(n_r=32, n_theta=64),
                                   PolarDisk(n_r=64, n_theta=128))
    check_orthogonality(PolarDisk(n_r=32), PolarDisk(n_r=64))
    check_uniform_drive_symmetry(disk)
    check_static_limit(disk)
    check_resonance(disk)
    check_numerical_eigenvalues(disk)

    print("\nmode frequencies (Hz):")
    for m, n, j in mode_table(4, 3)[:8]:
        print("   (m,n) = (%d,%d)   j = %7.4f   f = %8.2f Hz   f/f_01 = %.4f"
              % (m, n, j, mode_frequency(disk, j), j / jn_zeros(0, 1)[0]))

    print("\nfigures:")
    plot_modes(disk, os.path.join(here, "04_Membrane_modes.png"))
    plot_driven(disk, os.path.join(here, "04_Membrane_driven.png"))

    print("\nPASSED: steps 1 and 2")
    plt.show()
