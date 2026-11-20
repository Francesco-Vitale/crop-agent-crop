"""Historical-trials dataset from a noisy calendar rule (state-blind agronomic recommendation)."""
import os, numpy as np
from d3rlpy.dataset import MDPDataset
from .env import CropEnv, TRAIN, DOSES, H, rollout

def rule_policy_factory(rng, base_total=16.0, rand_p=0.15):
    """Recommendation: ~base_total g N/m2 split in two doses (weeks ~1 and ~4).
    Noise: lognormal dose-size error, +/-1 week timing jitter, 15% random actions."""
    total = base_total * rng.lognormal(0.0, 0.35)
    t1 = int(np.clip(1 + rng.integers(-1, 2), 0, H - 1)); t2 = int(np.clip(t1 + 3 + rng.integers(-1, 2), 0, H - 1))
    plan = {t1: total / 2, t2: total / 2}
    state = {"t": 0}
    def pol(o):
        t = state["t"]; state["t"] += 1
        if rng.random() < rand_p: return int(rng.integers(len(DOSES)))
        want = plan.get(t, 0.0)
        return int(np.abs(DOSES - want).argmin())
    return pol

def generate(n_per_year=40, seed=0, scenario=TRAIN):
    rng = np.random.default_rng(seed); env = CropEnv(scenario, seed)
    O, A, R, T, TO = [], [], [], [], []
    rets = []
    for y in scenario.years:
        for _ in range(n_per_year):
            ret, traj, info = rollout(env, rule_policy_factory(rng), year=y); rets.append(ret)
            for i, (o, a, r, o2, d) in enumerate(traj):
                O.append(o); A.append(a); R.append(r); T.append(float(d)); TO.append(0.0)
    return (np.array(O, np.float32), np.array(A, np.int64), np.array(R, np.float32),
            np.array(T, np.float32), np.array(TO, np.float32)), float(np.mean(rets))

def load_or_make(path="data/historical.npz", **kw):
    if os.path.exists(path):
        z = np.load(path); arr = tuple(z[k] for k in ("O", "A", "R", "T", "TO")); mr = float(z["mean_return"])
    else:
        arr, mr = generate(**kw); os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, O=arr[0], A=arr[1], R=arr[2], T=arr[3], TO=arr[4], mean_return=mr)
    return arr, mr

def to_mdp(arr):
    O, A, R, T, TO = arr
    return MDPDataset(O, A, R, T, TO, action_space=None) if False else MDPDataset(O, A, R, T, TO)
