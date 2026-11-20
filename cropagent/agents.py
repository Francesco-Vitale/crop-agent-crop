import logging, structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
import numpy as np, torch, d3rlpy
from d3rlpy.logging import NoopAdapterFactory
from .env import CropEnv, rollout, H

DEV = "cpu:0"
def seed_all(s):
    d3rlpy.seed(s); np.random.seed(s); torch.manual_seed(s)

def make_bc(seed, lr=1e-3):
    return d3rlpy.algos.DiscreteBCConfig(batch_size=128, learning_rate=lr).create(device=DEV)

def make_cql(seed, alpha=1.0, lr=3e-4):
    return d3rlpy.algos.DiscreteCQLConfig(batch_size=128, learning_rate=lr, gamma=1.0, alpha=alpha,
                                          target_update_interval=500).create(device=DEV)

def make_dqn(seed, lr=3e-4):
    return d3rlpy.algos.DQNConfig(batch_size=128, learning_rate=lr, gamma=1.0,
                                  target_update_interval=500).create(device=DEV)

def fit_offline(algo, dataset, n_steps):
    algo.fit(dataset, n_steps=n_steps, n_steps_per_epoch=max(n_steps // 5, 1), show_progress=False,
             logger_adapter=NoopAdapterFactory(), with_timestamp=False, experiment_name="x")
    return algo

def greedy(algo):
    return lambda o: int(algo.predict(np.asarray(o, np.float32)[None])[0])

def evaluate(algo, scenario):
    env = CropEnv(scenario); R, N = [], []
    pol = greedy(algo)
    for y in scenario.years:
        r, _, inf = rollout(env, pol, year=y); R.append(r); N.append(inf["n_tot"])
    return float(np.mean(R)), float(np.mean(N))
