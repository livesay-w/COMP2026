"""Step 2 checks: the least-squares fit of the semi-empirical mass formula.  python check_semf.py"""
import numpy as np

from ame import load_ame
from semf import A_MIN, NAMES, TERMS, design, fit, fit_sample, parity

d = load_ame()
Z, N, B = fit_sample(d)
rms = lambda r: np.sqrt(np.mean(r**2))

# 1. The fit itself: B built from known coefficients at the real (Z, N) must give those coefficients back exactly.
planted = np.array([15.0, 17.0, 0.7, 23.0, 12.0])
coef_syn, res_syn = fit(Z, N, design(Z, N) @ planted)
print(f"[1] planted {planted} -> fitted {np.round(coef_syn, 12)};  max |residual| = {np.abs(res_syn).max():.1e} MeV")
assert np.allclose(coef_syn, planted, atol=1e-10)

# 2. Real data. Coefficients near a standard textbook set (Rohlf 1994: 15.75, 17.8, 0.711, 23.7, 11.18 MeV, same
#    functional form). Published fits differ by data set and cuts, so the check allows 10%.
coef, res = fit(Z, N, B)
textbook = np.array([15.75, 17.8, 0.711, 23.7, 11.18])
print(f"[2] {len(B)} measured nuclei with A >= {A_MIN}")
for name, term, c, t in zip(NAMES, TERMS, coef, textbook):
    print(f"    {name} ({term:9s}) = {c:7.3f} MeV   textbook {t:6.3f}   ({100 * (c / t - 1):+.1f}%)")
print(f"    RMS residual = {rms(res):.2f} MeV, max |residual| = {np.abs(res).max():.1f} MeV "
      f"(B up to {B.max():.0f} MeV, so {100 * rms(res) / np.mean(B):.2f}% of the mean B)")
assert np.all(np.abs(coef / textbook - 1) < 0.10)
assert 2 < rms(res) < 4

# 3. Physics in the Coulomb coefficient: a uniformly charged sphere of radius R = r0 A^(1/3) has
#    a_C = (3/5) e^2 / (4 pi eps0 r0), with e^2 / (4 pi eps0) = 1.43996 MeV fm. Electron scattering gives r0 ~ 1.2 fm.
r0 = 0.6 * 1.43996 / coef[2]
print(f"[3] nuclear radius from a_C: r0 = {r0:.3f} fm  (electron scattering: ~1.2 fm)")
assert 1.1 < r0 < 1.35

# 4. Pairing is real: without the pairing term, even-even nuclei sit above the fit and odd-odd below, by ~2 delta.
#    With it, every parity class averages to ~0.
_, res4 = fit(Z, N, B, n_terms=4)
p = parity(Z, N)
mean4, mean5 = ([r[p == k].mean() for k in (1, 0, -1)] for r in (res4, res))
print(f"[4] mean residual (even-even, odd A, odd-odd): without pairing {np.round(mean4, 2)},  with {np.round(mean5, 2)} MeV")
assert mean4[0] > mean4[1] > mean4[2] and np.all(np.abs(mean5) < 0.5)

# 5. Not asserted: the error as each term is added in turn (the hand-made version of PySR's complexity ladder),
#    and the sensitivity to the light-nucleus cut A_MIN. This order is also the greedy one (each step adds the term
#    that lowers the RMS most). Coulomb and asymmetry only pay off together: heavy nuclei have N > Z because Coulomb
#    repulsion pushes them off N = Z, so neither term fits the valley of stability without the other.
print("[5] RMS as terms are added: " + ",  ".join(
    f"+{t} {rms(fit(Z, N, B, k + 1)[1]):.1f}" for k, t in enumerate(TERMS)) + " MeV")
for a_min in (2, 16, 40):
    c, r = fit(*fit_sample(d, a_min))
    print(f"    A >= {a_min:2d}: coefficients {np.round(c, 3)},  RMS {rms(r):.2f} MeV,  max |residual| {np.abs(r).max():.1f} MeV")

print("All step-2 checks passed.")
