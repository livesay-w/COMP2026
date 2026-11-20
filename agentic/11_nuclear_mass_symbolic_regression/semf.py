"""The semi-empirical (Bethe-Weizsaecker) mass formula, fitted to AME2020 by linear least squares. Energies in MeV.

    B(Z, N) = a_V A - a_S A^(2/3) - a_C Z^2 / A^(1/3) - a_A (N - Z)^2 / A + a_P parity / sqrt(A)

parity = +1 for even-even, -1 for odd-odd, 0 for odd A. B is linear in the five coefficients, so the best fit
is a single lstsq solve. This is the hand-built reference that the symbolic-regression results are measured against.
"""
import numpy as np

TERMS = ["volume", "surface", "Coulomb", "asymmetry", "pairing"]
NAMES = ["a_V", "a_S", "a_C", "a_A", "a_P"]
# The formula describes the bulk of the nuclear chart. The lightest nuclei (4He, 12C, ...) are dominated by
# individual-nucleon structure and pull the fit, so they are left out. check_semf.py reports the effect of this cut.
A_MIN = 16


def parity(Z, N):
    """+1 even-even, -1 odd-odd, 0 odd A."""
    return np.where((Z % 2 == 0) & (N % 2 == 0), 1, np.where((Z % 2 == 1) & (N % 2 == 1), -1, 0))


def design(Z, N):
    """Columns = each term per unit coefficient, with its sign, so B = design(Z, N) @ coefficients."""
    A = (Z + N).astype(float)
    return np.stack([A, -A ** (2 / 3), -Z**2 / A ** (1 / 3), -(N - Z) ** 2 / A, parity(Z, N) / np.sqrt(A)], axis=1)


def fit(Z, N, B, n_terms=5):
    """Least-squares coefficients using the first n_terms terms (in TERMS order), and the residuals B - fit."""
    X = design(Z, N)[:, :n_terms]
    coef, *_ = np.linalg.lstsq(X, B, rcond=None)
    return coef, B - X @ coef


def fit_sample(d, a_min=A_MIN):
    """The measured nuclei with A >= a_min, as (Z, N, B)."""
    s = d["measured"] & (d["A"] >= a_min)
    return d["Z"][s], d["N"][s], d["B"][s]
