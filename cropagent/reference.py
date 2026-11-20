"""Reference policies: best open-loop 3-dose plan (weeks 1-3) per scenario, from brute-force simulation."""
import itertools, json, numpy as np
from .env import CropEnv, DOSES, rollout

def best_plan(scenario):
    env = CropEnv(scenario); best = (-1e9, None)
    for plan in itertools.product(range(6), repeat=3):
        acts = list(plan) + [0] * 9; it = iter(acts)
        R = np.mean([rollout(env, lambda o: next(it_), year=y)[0] for y in scenario.years
                     for it_ in [iter(acts)]])
        if R > best[0]: best = (float(R), [float(DOSES[a]) for a in plan])
    return best
