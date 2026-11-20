"""Step 3b: stagewise (residual) symbolic regression on synthetic data made from the step-2 formula.

Build the law one piece at a time, the way the mass formula was built by hand:
  1. PySR searches what is still unexplained (stage 1: B itself), with a small size limit (STAGE_MAXSIZE).
  2. From that stage's front, keep the formula with the steepest drop in log(loss) per added node (PySR's "score"):
     the simplest formula that buys the most. Its shape becomes a new term.
  3. Re-fit the coefficients of ALL kept terms together (linear least squares, plus a constant), so earlier
     terms can adjust when a new one arrives. The new leftover = target - joint fit.
  4. Stop when a new term improves the error on held-out nuclei (AME estimates, never fitted) by less than STOP_GAIN.
Assumption, stated: B is a sum of contributions. Operators and inputs are the generic ones of sr.py.

    python stagewise.py clean     -> runs/stagewise_clean.csv, runs/stagewise_clean_stage<k>_front.csv
"""
import sys

import numpy as np
import pandas as pd

from ame import load_ame
from semf import design, fit, fit_sample
from sr import RUNS, VARIABLES, features, make_model

STAGE_MAXSIZE = 20   # room for two linked terms in one stage
STAGE_ITER = 1000
MAX_STAGES = 6
STOP_GAIN = 0.05     # stop when held-out RMS falls by less than 5%
tag = sys.argv[1]
assert tag in ("clean", "noisy")

d = load_ame()
Z, N, B = fit_sample(d)
coef, res = fit(Z, N, B)
truth = design(Z, N) @ coef
y = truth + (np.sqrt(np.mean(res**2)) if tag == "noisy" else 0.0) * np.random.default_rng(0).normal(size=len(truth))
held = ~d["measured"] & (d["A"] >= 16)
X, X_held = features(Z, N), features(d["Z"][held], d["N"][held])
truth_held = design(d["Z"][held], d["N"][held]) @ coef
rms = lambda r: np.sqrt(np.mean(r**2))


def joint_fit(cols, cols_held):
    """Least-squares coefficients for the kept term shapes plus a constant; predictions on fitted and held-out nuclei."""
    F, F_held = np.column_stack(cols + [np.ones(len(y))]), np.column_stack(cols_held + [np.ones(len(truth_held))])
    c, *_ = np.linalg.lstsq(F, y, rcond=None)
    return c, F @ c, F_held @ c


cols, cols_held, rows = [], [], []
pred, pred_held = np.zeros(len(y)), np.zeros(len(truth_held))
prev_held = rms(truth_held)  # the "no model" error
for stage in range(1, MAX_STAGES + 1):
    leftover = y - pred
    model = make_model(f"stagewise_{tag}_stage{stage}", STAGE_ITER, seed=stage, maxsize=STAGE_MAXSIZE)
    model.fit(X, leftover, variable_names=VARIABLES)
    eq = model.equations_
    eq[["complexity", "loss", "equation"]].to_csv(RUNS / f"stagewise_{tag}_stage{stage}_front.csv", index=False)
    score = -np.diff(np.log(eq["loss"].to_numpy())) / np.diff(eq["complexity"].to_numpy())
    k = np.argmax(score) + 1  # the formula at the bottom of the steepest drop
    cols.append(model.predict(X, index=k))
    cols_held.append(model.predict(X_held, index=k))
    c, new_pred, new_pred_held = joint_fit(cols, cols_held)
    gain = 1 - rms(new_pred_held - truth_held) / prev_held
    rows.append(dict(stage=stage, complexity=int(eq["complexity"][k]), term=eq["equation"][k],
                     rms_fit=rms(y - new_pred), rms_truth=rms(new_pred - truth),
                     rms_held_out=rms(new_pred_held - truth_held), held_gain=gain,
                     coefficients=" ".join(f"{v:.4g}" for v in c)))
    print(f"stage {stage}: kept (complexity {rows[-1]['complexity']}) {rows[-1]['term']}\n"
          f"    joint fit: RMS {rows[-1]['rms_fit']:.3f} MeV (vs truth {rows[-1]['rms_truth']:.3f}), "
          f"held-out {rows[-1]['rms_held_out']:.3f} MeV ({100 * gain:+.1f}%);  coefficients {rows[-1]['coefficients']}",
          flush=True)
    if gain < STOP_GAIN:
        print(f"    held-out gain below {100 * STOP_GAIN:.0f}%: stop (this last term is not kept)")
        rows[-1]["kept"] = False
        break
    rows[-1]["kept"] = True
    pred, pred_held, prev_held = new_pred, new_pred_held, rms(new_pred_held - truth_held)

pd.DataFrame(rows).to_csv(RUNS / f"stagewise_{tag}.csv", index=False)
