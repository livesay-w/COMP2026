"""PySR search settings shared by steps 3-6. Energies in MeV.

Inputs: A, Z, Nn (= N; 'N' clashes with a sympy function) and P = pairing parity (+1 even-even, 0 odd A,
-1 odd-odd). P is a fact about each nucleus, not part of any formula: no operator below can compute even/odd
from Z and N, so without P no even-odd effect could ever be found.

Two generic operator sets, both chosen without reference to the mass formula:
  "calculator" (default from step 3d on): + - * /  and the standard scientific-calculator functions
        sqrt, cbrt, square, cube, inv (1/x), exp, log. Powers are fixed shapes, nothing to tune.
  "power" (steps 3, 3b, 3c and the first ladder): + - * /,  x^c with the exponent limited to one constant or
        variable (so A^0.667 must be found as a tuned number),  exp, log.
        Step 3d showed the search cannot reliably tune fractional exponents: it misses even 15.4 A - 16.5 A^(2/3).
Nesting constraints forbid towers no physical law needs: exp or log inside exp or log (and, for "power",
anything exponential or ^ inside ^).

Search: 10 threads (2 of the 12 cores stay free). Seeded, but multithreaded PySR is not bit-for-bit
reproducible, so every result is saved and the figures are made from the saved fronts.
"""
import os

os.environ.setdefault("PYTHON_JULIACALL_THREADS", "10")  # must be set before Julia starts

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from pysr import PySRRegressor  # noqa: E402

from semf import parity  # noqa: E402

RUNS = Path(__file__).parent / "runs"
VARIABLES = ["A", "Z", "Nn", "P"]
MAXSIZE = 45
OPERATORS = {
    "calculator": dict(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sqrt", "cbrt", "square", "cube", "inv(x) = 1/x", "exp", "log"],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}},
    ),
    "power": dict(
        binary_operators=["+", "-", "*", "/", "^"], unary_operators=["exp", "log"],
        constraints={"^": (-1, 1)},
        nested_constraints={k: {"exp": 0, "log": 0, "^": 0} for k in ("exp", "log", "^")},
    ),
}


def features(Z, N):
    """The PySR inputs, columns in VARIABLES order."""
    return np.stack([Z + N, Z, N, parity(Z, N)], axis=1).astype(float)


def make_model(name, niterations, seed=0, maxsize=MAXSIZE, operators="calculator"):
    return PySRRegressor(
        niterations=niterations, maxsize=maxsize, **OPERATORS[operators],
        random_state=seed, parallelism="multithreading", deterministic=False,
        verbosity=0, progress=False, output_directory=str(RUNS / f"pysr_{name}"),
    )


def front_table(model, X, y):
    """The Pareto front with the RMS error (MeV) of every formula on (X, y)."""
    eq = model.equations_.copy()
    eq["rms"] = [np.sqrt(np.mean((model.predict(X, index=i) - y) ** 2)) for i in range(len(eq))]
    return eq[["complexity", "loss", "rms", "equation"]]
