"""Step 3 trial: PySR on clean synthetic data, B = the step-2 formula exactly (no noise), at the real (Z, N).

Settings tried: inputs A, Z, Nn, P (P = pairing parity; 'N' clashes with a sympy function, hence Nn),
operators + - * / cbrt sqrt square, maxsize 40, seeded and deterministic.
Target "B" (binding energy) or "BA" (B/A; the RMS is reported in MeV of B either way).

python search_trial.py 1000 B   ->  runs/pysr_trial_B/.../hall_of_fame.csv  (~5.5 min)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pysr import PySRRegressor

from ame import load_ame
from semf import design, fit, fit_sample, parity

RUNS = Path(__file__).parent / "runs"
n_iter, target = int(sys.argv[1]), sys.argv[2]

Z, N, B = fit_sample(load_ame())
coef, _ = fit(Z, N, B)
A = (Z + N).astype(float)
y = design(Z, N) @ coef  # the planted law, exactly
if target == "BA":
    y = y / A
X = np.stack([A, Z, N, parity(Z, N)], axis=1).astype(float)

model = PySRRegressor(
    niterations=n_iter, maxsize=40, binary_operators=["+", "-", "*", "/"], unary_operators=["cbrt", "sqrt", "square"],
    random_state=0, deterministic=True, parallelism="serial", verbosity=0, progress=False,
    output_directory=str(RUNS / f"pysr_trial_{target}"),
)
model.fit(X, y, variable_names=["A", "Z", "Nn", "P"])
eq = model.equations_
scale = 1 if target == "B" else A
eq["rms_MeV"] = [np.sqrt(np.mean((scale * (model.predict(X, index=i) - y)) ** 2)) for i in range(len(eq))]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 140)
print(eq[["complexity", "rms_MeV", "equation"]].to_string())
