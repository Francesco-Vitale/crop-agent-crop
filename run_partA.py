import json, os, sys, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, d3rlpy
from cropagent.data import load_or_make, to_mdp
from cropagent.agents import *; from cropagent.finetune import *; from cropagent.env import *
E = int(os.environ.get("EPISODES", 40)); U = int(os.environ.get("UPDATES", 100)); NPY = int(os.environ.get("NPY", 30))
OFFSTEPS = int(os.environ.get("OFFSTEPS", 6000)); BCSTEPS = 3000
os.makedirs("results", exist_ok=True); os.makedirs("/tmp/ckpt", exist_ok=True)

def run_seed(seed):
    t0 = time.time(); out = {"seed": seed}
    arr, mr = load_or_make(path=f"data/historical_s{seed}.npz", n_per_year=NPY, seed=seed)
    ds = to_mdp(arr); off = (arr[0], arr[1], arr[2], arr[3]); out["behavior_train_return"] = mr
    # offline pretraining
    seed_all(seed); cql = make_cql(seed, alpha=0.3); fit_offline(cql, ds, OFFSTEPS)
    seed_all(seed); bc = make_bc(seed); fit_offline(bc, ds, BCSTEPS)
    out["offline_cql_train_scen"] = evaluate(cql, TRAIN); out["bc_train_scen"] = evaluate(bc, TRAIN)
    out["bc_shift"] = evaluate(bc, SHIFT_EVAL)
    p = f"/tmp/ckpt/cql_{seed}.d3"; cql.save(p)
    load = lambda: d3rlpy.load_learnable(p, device=DEV)
    # variants
    seed_all(seed + 100); c = load(); out["offline_ft"] = online_finetune(c, off, E, seed, updates_per_ep=U)
    seed_all(seed + 100); c = load(); c.impl._alpha = 0.0
    out["offline_ft_alpha0"] = online_finetune(c, off, E, seed, updates_per_ep=U)
    seed_all(seed + 100); q = make_dqn(seed); q.build_with_dataset(ds); bc_to_q(bc, q)
    out["bc_ft"] = online_finetune(q, off, E, seed, updates_per_ep=U)
    seed_all(seed + 100); q = make_dqn(seed); q.build_with_dataset(ds)
    out["scratch"] = online_finetune(q, off, E, seed, updates_per_ep=U, use_offline=False)
    out["offline_only"] = [(e,) + tuple(out["offline_ft"][0][1:]) for e, *_ in out["offline_ft"]]
    out["secs"] = time.time() - t0
    json.dump(out, open(f"results/partA_seed{seed}.json", "w")); print("seed", seed, "done", round(out["secs"]), flush=True)

if __name__ == "__main__":
    for s in [int(x) for x in sys.argv[1:]]: run_seed(s)
