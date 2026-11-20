"""Steps 3 and 3c: can PySR rediscover a known law? Synthetic B = the step-2 fitted formula.

    python synthetic.py clean   at the 2,484 real nuclei (the narrow valley of stability)
    python synthetic.py noisy   the same, plus Gaussian noise of 3.29 MeV (the real scatter about the formula)
    python synthetic.py wide    no noise, at 2,484 nuclei drawn from a wide (Z, N) grid far beyond the valley

PySR is told nothing about the formula (sr.py: generic operators). Saves runs/synthetic_<tag>_front.csv with each
formula's RMS against the target and the planted law, on the 1,003 AME-estimated nuclei that were never fitted, and
on the 2,484 real (valley) nuclei. Only a formula equal to the planted law is right everywhere.
"""
import sys

import numpy as np
import pandas as pd

from ame import load_ame
from semf import design, fit, fit_sample
from sr import RUNS, VARIABLES, features, front_table, make_model

N_ITER = 3000  # ~11 min on 10 threads


def wide_grid(n, seed=0):
    """n nuclei drawn uniformly from Z = 8..110, N = 0.6 Z..2.0 Z, A >= 16: a band about 3 times wider in N/Z than
    the valley of stability (N/Z = 1.0-1.55). Fixed bounds, chosen without reference to the formula."""
    Z, N = np.meshgrid(np.arange(8, 111), np.arange(5, 221), indexing="ij")
    keep = (N >= 0.6 * Z) & (N <= 2.0 * Z) & (Z + N >= 16)
    pick = np.random.default_rng(seed).choice(keep.sum(), size=n, replace=False)
    return Z[keep][pick], N[keep][pick]


if __name__ == "__main__":
    tag = sys.argv[1]
    assert tag in ("clean", "noisy", "wide")
    d = load_ame()
    Z_val, N_val, B_val = fit_sample(d)
    coef, res = fit(Z_val, N_val, B_val)
    Z, N = wide_grid(len(B_val)) if tag == "wide" else (Z_val, N_val)
    truth = design(Z, N) @ coef
    noise = np.sqrt(np.mean(res**2)) if tag == "noisy" else 0.0
    y = truth + noise * np.random.default_rng(0).normal(size=len(truth))

    # Never fitted: the AME estimates (A >= 16), and the real valley nuclei, each with the planted law as "truth".
    held = ~d["measured"] & (d["A"] >= 16)
    tests = {"held_out": (d["Z"][held], d["N"][held]), "valley": (Z_val, N_val)}

    model = make_model(f"synthetic_{tag}", N_ITER)
    model.fit(features(Z, N), y, variable_names=VARIABLES)
    front = front_table(model, features(Z, N), y).rename(columns={"rms": "rms_target"})
    rms_of = lambda Zt, Nt: [np.sqrt(np.mean((model.predict(features(Zt, Nt), index=i) - design(Zt, Nt) @ coef) ** 2))
                             for i in range(len(front))]
    front["rms_truth"] = rms_of(Z, N)
    for name, (Zt, Nt) in tests.items():
        front[f"rms_{name}"] = rms_of(Zt, Nt)
    front.to_csv(RUNS / f"synthetic_{tag}_front.csv", index=False)

    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 150)
    print(f"synthetic '{tag}': noise {noise:.2f} MeV, {len(y)} nuclei fitted (N/Z {np.min(N / Z):.2f}-{np.max(N / Z):.2f}), "
          f"{held.sum()} held out, {N_ITER} iterations")
    print(front[["complexity", "rms_target", "rms_truth", "rms_held_out", "rms_valley", "equation"]].to_string())
