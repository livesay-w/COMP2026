"""Step 0 check: PySR runs in this environment and recovers a planted law of the kind this project needs.

Planted law: y = 15 A - 17 A^(2/3) (a volume term minus a surface term) on A = 1..250, with no noise.
PySR must find it exactly. It needs the cube root to build A^(2/3), which the real fits will need too.
Run with miniconda base (it has PySR and Julia).  python check_setup.py
"""
from pathlib import Path

import numpy as np
from pysr import PySRRegressor

RUNS = Path(__file__).parent / "runs"

A = np.arange(1, 251, dtype=float)
y = 15 * A - 17 * A ** (2 / 3)

model = PySRRegressor(
    niterations=40, binary_operators=["+", "-", "*"], unary_operators=["cbrt"],
    random_state=0, deterministic=True, parallelism="serial",  # the same search every run
    verbosity=0, progress=False, output_directory=str(RUNS / "pysr_setup"),
)
model.fit(A.reshape(-1, 1), y, variable_names=["A"])
print(model.equations_[["complexity", "loss", "equation"]].to_string())

best = model.sympy()
err = np.max(np.abs(model.predict(A.reshape(-1, 1)) - y) / y.max())
print(f"\n[0] best formula: {best}\n    max |error| / max(y) on A = 1..250: {err:.1e}")
assert err < 1e-6, "PySR did not recover the planted law"
print("Step-0 check passed.")
