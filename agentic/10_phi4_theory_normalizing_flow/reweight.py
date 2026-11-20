"""Step 7e: multi-histogram reweighting in m2 (Ferrenberg & Swendsen 1989; the MBAR form of Shirts & Chodera 2008).

The action is S = S_0 + m2 E with E = 1/2 sum_x phi_x^2, so the chain at m2_k samples p_k ∝ exp(-S_0 - m2_k E). All K chains of one
L are combined: every sample s gets, for a target m2, the weight
    w_s(m2) = exp(-m2 E_s) / D_s,      D_s = sum_k n_k exp(f_k - m2_k E_s),
where f_k = -log Z(m2_k) (up to one constant, f_0 = 0) solves the self-consistency exp(-f_j) = sum_s w_s(m2_j). Then
<O>(m2) = sum_s w_s O_s / sum_s w_s is a smooth function of m2 across the simulated range. This replaces linear interpolation
between grid points, and uses every chain, not just the two bracketing ones.

The f_k minimise the convex function F(f) = sum_s log D_s - sum_k n_k f_k; Newton's method on it converges in a few steps.
Everything is done with logs: m2 E is ~ -10^4 at L = 128.

Lattice units, lam = 1. Used by quotient.py and checked by check_reweight.py.
"""
import numpy as np
from scipy.special import logsumexp


class Reweighter:
    """Multi-histogram estimator for the chains of one L. E: list of K sample arrays of E = 1/2 sum phi^2; m2: (K,).
    f_start: starting f for Newton (e.g. the full-sample solution when refitting a jackknife replica); None starts from
    thermodynamic integration. Either way Newton converges to the same f."""

    def __init__(self, E, m2, f_start=None, tol=1e-10):
        self.m2 = np.asarray(m2, dtype=float)
        self.E = np.concatenate(E)
        self.logn = np.log([len(e) for e in E])
        self.f = self._solve(self._start(E) if f_start is None else np.array(f_start, dtype=float), tol)
        self.logD = logsumexp(self.logn + self.f - np.outer(self.E, self.m2), axis=1)

    def _start(self, E):
        """Starting f from thermodynamic integration, df/dm2 = <E> (trapezoid between neighbouring chains). Newton needs it:
        from f = 0 each sample is assigned entirely to one chain (m2_k E_s differs by ~10 between neighbours), so the
        Hessian is singular."""
        order = np.argsort(self.m2)
        means = np.array([e.mean() for e in E])[order]
        f_sorted = np.r_[0, np.cumsum(np.diff(self.m2[order]) * 0.5 * (means[1:] + means[:-1]))]
        f = np.empty_like(f_sorted)
        f[order] = f_sorted
        return f - f[0]

    def _solve(self, f, tol):
        """Newton iteration for f, with f_0 = 0 fixed. Converged when every gradient component (an expected sample count
        minus the actual count n_k) is below tol * n_k."""
        n = np.exp(self.logn)
        for _ in range(100):
            logP = self.logn + f - np.outer(self.E, self.m2)
            P = np.exp(logP - logsumexp(logP, axis=1, keepdims=True))  # (N, K); each row sums to 1
            grad = P.sum(axis=0) - n
            if np.max(np.abs(grad) / n) < tol:
                return f
            H = np.diag(P.sum(axis=0)) - P.T @ P
            f[1:] -= np.linalg.solve(H[1:, 1:], grad[1:])
        raise RuntimeError("multi-histogram free energies did not converge in 100 Newton steps")

    def mean(self, obs, m2):
        """<obs>(m2) for each column of obs (N, n_obs), at one target m2 inside the simulated range."""
        assert self.m2.min() - 1e-12 <= m2 <= self.m2.max() + 1e-12, "reweighting outside the simulated m2 range"
        logw = -m2 * self.E - self.logD
        w = np.exp(logw - logw.max())
        return w @ obs / w.sum()


def u4_chi(rw, M, L, m2):
    """Binder U4 and chi = V<M^2> at m2 from a Reweighter rw and the matching M samples (concatenated in the same order)."""
    m2_, m4 = rw.mean(np.stack([M**2, M**4], axis=1), m2)
    return 1 - m4 / (3 * m2_**2), L * L * m2_
