"""Step 3d: how large a law can the blind search rediscover exactly? Noise-free synthetic data at the real nuclei.

Planted laws: the step-2 formula truncated to its first k terms (step-2 coefficients). Their sizes in PySR's node
count depend on the operator set (sr.py):
    "power":       volume 3,  + surface 9 (a A^c),  + Coulomb 19,  + asymmetry 29                 (k = 1..4)
    "calculator":  volume 3,  + surface 9 (a square(cbrt A)),  + Coulomb 17,  + asymmetry 26,  + pairing 33  (k = 1..5)
Each gets its own blind search with maxsize = planted size + 6, 1,000 iterations. "Exact" = some formula on the
front matches the planted law to RMS < 1e-3 MeV (a relative error ~1e-6 of a typical B).

    python synthetic_ladder.py calculator  ->  runs/synthetic_ladder_calculator*.csv   (~15 min)
"""
import sys

import pandas as pd

from ame import load_ame
from semf import TERMS, design, fit, fit_sample
from sr import RUNS, VARIABLES, features, front_table, make_model

PLANTED_SIZE = {"power": {1: 3, 2: 9, 3: 19, 4: 29}, "calculator": {1: 3, 2: 9, 3: 17, 4: 26, 5: 33}}
N_ITER = 1000
EXACT = 1e-3  # MeV
ops = sys.argv[1]
name = f"synthetic_ladder_{ops}"

Z, N, B = fit_sample(load_ame())
coef, _ = fit(Z, N, B)
X = features(Z, N)
rows = []
for k, size in PLANTED_SIZE[ops].items():
    truth = design(Z, N)[:, :k] @ coef[:k]
    model = make_model(f"ladder_{ops}_k{k}", N_ITER, maxsize=size + 6, operators=ops)
    model.fit(X, truth, variable_names=VARIABLES)
    front = front_table(model, X, truth)
    front.to_csv(RUNS / f"{name}_k{k}_front.csv", index=False)
    hit = front[front["rms"] < EXACT]
    found = len(hit) > 0
    rows.append(dict(terms=k, law=" + ".join(TERMS[:k]), planted_size=size, exact=found,
                     found_at=int(hit["complexity"].iloc[0]) if found else -1,
                     best_rms=front["rms"].min(), formula=hit["equation"].iloc[0] if found else front["equation"].iloc[-1]))
    r = rows[-1]
    print(f"{k} terms ({r['law']}, {size} nodes): {'EXACT at complexity ' + str(r['found_at']) if found else 'not found'};"
          f"  best RMS {r['best_rms']:.2e} MeV\n    {r['formula']}", flush=True)
pd.DataFrame(rows).to_csv(RUNS / f"{name}.csv", index=False)
