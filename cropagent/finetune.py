"""Online fine-tuning loop with balanced offline/online mini-batches (d3rlpy algos, custom loop)."""
import numpy as np, torch
from d3rlpy.dataset import TransitionMiniBatch
from .env import CropEnv, SHIFT_FT, SHIFT_EVAL, H
from .agents import (seed_all, make_bc, make_cql, make_dqn, fit_offline, evaluate, greedy)

def next_obs(O, T):
    NO = np.zeros_like(O); NO[:-1] = O[1:]; NO[T > 0] = 0.0
    return NO

def batch_from(O, A, R, NO, T, idx):
    n = len(idx); col = lambda x, dt: x[idx].reshape(n, 1).astype(dt)
    return TransitionMiniBatch(observations=O[idx].astype(np.float32), actions=col(A, np.float32),
        rewards=col(R, np.float32), next_observations=NO[idx].astype(np.float32),
        next_actions=np.zeros((n, 1), np.float32), terminals=col(T, np.float32),
        intervals=np.ones((n, 1), np.float32), transitions=[])

def bc_to_q(bc, q):
    """Copy BC network (encoder + linear head) into every Q-function head and target of `q`."""
    sd = bc.impl.modules.imitator.state_dict()
    for group in (q.impl.modules.q_funcs, q.impl.modules.targ_q_funcs):
        for m in group: m.load_state_dict(sd)

def online_finetune(algo, offline, n_episodes, seed, eval_every=4, updates_per_ep=60, batch=128,
                    eps0=0.3, eps1=0.05, eps_eps=20, use_offline=True, scenario=SHIFT_FT, eval_scenario=SHIFT_EVAL):
    rng = np.random.default_rng(seed); env = CropEnv(scenario, seed)
    Oo, Ao, Ro, To = offline; NOo = next_obs(Oo, To)
    on = {k: [] for k in "OARNT"}
    curve = [(0,) + evaluate(algo, eval_scenario)]
    for ep in range(1, n_episodes + 1):
        eps = max(eps1, eps0 + (eps1 - eps0) * ep / eps_eps)
        o, _ = env.reset(); done = False
        while not done:
            a = int(rng.integers(6)) if rng.random() < eps else int(algo.predict(o[None])[0])
            o2, r, done, _, _ = env.step(a)
            on["O"].append(o); on["A"].append(a); on["R"].append(r); on["N"].append(o2); on["T"].append(float(done)); o = o2
        On, An, Rn, NOn, Tn = (np.array(on[k], np.float32 if k != "A" else np.int64) for k in "OARNT")
        for _ in range(updates_per_ep):
            if use_offline:
                half = batch // 2
                i1 = rng.integers(len(Oo), size=half); i2 = rng.integers(len(On), size=batch - half)
                b1 = (Oo[i1], Ao[i1], Ro[i1], NOo[i1], To[i1]); b2 = (On[i2], An[i2], Rn[i2], NOn[i2], Tn[i2])
                cat = [np.concatenate([x, y]) for x, y in zip(b1, b2)]
            else:
                i2 = rng.integers(len(On), size=batch); cat = [On[i2], An[i2], Rn[i2], NOn[i2], Tn[i2]]
            algo.update(batch_from(*cat, np.arange(len(cat[0]))))
        if ep % eval_every == 0: curve.append((ep,) + evaluate(algo, eval_scenario))
    return curve
