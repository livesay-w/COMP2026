"""AME2020 atomic mass table: binding energies of nuclei. Energies in MeV.

Data: data/mass_1.mas20.txt from https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt
(W.J. Huang et al. and M. Wang et al., Chinese Physics C 45, 030002 and 030003 (2021)).
Fixed-width Fortran format, documented in the file header:
    a1,i3,i5,i5,i5,1x,a3,a4,1x,f14.6,f12.6,f13.5,1x,f10.5,...
    cc N-Z  N  Z  A    el  o  mass_excess unc  B/A   unc       (keV)
A '#' in place of the decimal point marks an estimated (non-experimental) value.
"""
from pathlib import Path

import numpy as np

AME_FILE = Path(__file__).parent / "data" / "mass_1.mas20.txt"
HEADER_LINES = 36  # the file's own documentation; data start on line 37


def keV(field):
    """A numeric field of the table in keV, and whether it was measured (no '#')."""
    return float(field.replace("#", ".")), "#" not in field


def load_ame(path=AME_FILE):
    """Every nuclide in the table, as a dict of arrays:
    Z, N, A; element; B and dB, total binding energy and its uncertainty (MeV); mass_excess (MeV, atomic);
    measured, False where AME gives only an estimate of the binding energy."""
    rows = []
    for line in path.read_text().splitlines()[HEADER_LINES:]:
        N, Z, A = int(line[4:9]), int(line[9:14]), int(line[14:19])
        assert N + Z == A and int(line[1:4]) == N - Z, f"misread columns: {line[:30]!r}"
        excess, _ = keV(line[28:42])
        b_per_a, measured = keV(line[54:67])
        db_per_a, _ = keV(line[68:78])
        rows.append((Z, N, A, line[20:23].strip(), A * b_per_a / 1e3, A * db_per_a / 1e3, excess / 1e3, measured))
    cols = list(zip(*rows))
    names = ["Z", "N", "A", "element", "B", "dB", "mass_excess", "measured"]
    return {name: np.array(col) for name, col in zip(names, cols)}
