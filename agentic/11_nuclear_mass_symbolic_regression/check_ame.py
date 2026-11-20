"""Step 1 checks: the AME2020 table is read correctly.  python check_ame.py"""
import numpy as np

from ame import AME_FILE, HEADER_LINES, load_ame

d = load_ame()
meas = d["measured"]
idx = {(z, a): i for i, (z, a) in enumerate(zip(d["Z"], d["A"]))}


def nucleus(Z, A):
    return idx[(Z, A)]


# 1. Every data line was read, once each (A = N + Z and the N - Z column are asserted inside load_ame).
n_lines = len(AME_FILE.read_text().splitlines()) - HEADER_LINES
print(f"[1] {len(d['A'])} nuclides read of {n_lines} data lines;  measured {meas.sum()}, estimated {(~meas).sum()};  "
      f"Z = {d['Z'].min()}-{d['Z'].max()}, A = {d['A'].min()}-{d['A'].max()}")
assert len(d["A"]) == n_lines == len(idx)

# 2. Two different columns agree. With atomic mass excesses Delta, B = Z Delta(1H) + N Delta(n) - Delta(Z, N).
#    A wrong column slice or a units slip would break this for every nucleus.
d_H, d_n = d["mass_excess"][nucleus(1, 1)], d["mass_excess"][nucleus(0, 1)]
B_from_mass = d["Z"] * d_H + d["N"] * d_n - d["mass_excess"]
diff = np.abs(B_from_mass - d["B"])[meas]
print(f"[2] B from the mass-excess column vs the binding-energy column (measured): max |diff| = {diff.max() * 1e3:.3f} keV")
assert diff.max() < 1e-3  # 1 keV, against binding energies up to ~1800 MeV

# 3. Textbook binding energies (MeV).
for name, Z, A, known in [("2H", 1, 2, 2.22457), ("4He", 2, 4, 28.2957), ("12C", 6, 12, 92.1618), ("208Pb", 82, 208, 1636.43)]:
    B = d["B"][nucleus(Z, A)]
    print(f"[3] {name:6s} B = {B:10.4f} MeV   (textbook {known})")
    assert abs(B - known) < 0.01

# 4. Binding energy per nucleon peaks at 62Ni, just above 58Fe and 56Fe (measured nuclei).
BA = np.where(meas, d["B"] / d["A"], -np.inf)
top = np.argsort(BA)[::-1][:3]
print("[4] largest B/A: " + ",  ".join(f"{d['A'][i]}{d['element'][i]} {BA[i]:.4f} MeV" for i in top))
assert [(d["Z"][i], d["A"][i]) for i in top] == [(28, 62), (26, 58), (26, 56)]

# 5. Beta stability: of all isobars, the stable one has the lowest atomic mass (mass excess).
for A, Z_stable in [(12, 6), (56, 26), (100, 44), (208, 82)]:
    iso = np.flatnonzero((d["A"] == A) & meas)
    Z_min = d["Z"][iso[np.argmin(d["mass_excess"][iso])]]
    print(f"[5] A = {A:3d}: lightest isobar has Z = {Z_min}  (stable: Z = {Z_stable})")
    assert Z_min == Z_stable

print("All step-1 checks passed.")
