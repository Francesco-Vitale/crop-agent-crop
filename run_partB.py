import json, os, warnings; warnings.filterwarnings("ignore")
import numpy as np
from cropagent.design import *
REPS = int(os.environ.get("REPS", 100)); NT = int(os.environ.get("TRIALS", 24))
Y = response_table(); true_net = net(Y.mean(0), N_GRID); sig = effective_sigma(Y)
res = {"n_trials": NT, "reps": REPS, "best_N": float(N_GRID[true_net.argmax()]), "best_net": float(true_net.max()), "model_sigma": sig}
print("model noise sd", round(sig, 1), "(old, overconfident setting: 30)")
for name, (design, M) in make_methods(sig).items():
    R, E = [], []
    for r in range(REPS):
        reg, rmse, *_ = run_method(Y, design, M(), NT, 1000 + r); R.append(reg); E.append(rmse)
    res[name] = {"regret": np.array(R).tolist(), "rmse": np.array(E).tolist()}
    print(name, "regret @6", round(np.mean(np.array(R)[:, 5]), 3), "@12", round(np.mean(np.array(R)[:, 11]), 3), "@24", round(np.mean(np.array(R)[:, -1]), 3), flush=True)
json.dump(res, open("results/partB.json", "w"))
