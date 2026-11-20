"""Gymnasium environment: weekly nitrogen decisions for LINTUL3 spring wheat.

Observation (7): week/H, DVS, LAI/6, mineral soil N/20, NNI, total N applied/30, TRANRF.
Action (discrete 0..5): N dose in DOSES g N/m2 (x10 = kg N/ha).
Reward: per step -COST_N*dose; at the end +PRICE*yield. Units: 100 EUR/ha.
Scenario shift: fertilizer recovery fraction (share of applied N that becomes plant-available).
"""
from dataclasses import dataclass
import numpy as np, gymnasium as gym
from gymnasium import spaces
from . import sim

DOSES = np.array([0., 2., 4., 6., 8., 10.])
H = 12                  # decision weeks after emergence
PRICE = 2.0 / 100       # 100 EUR/ha per (g/m2 grain-equivalent WSO): 0.20 EUR/kg
COST_N = 10.0 / 100     # 100 EUR/ha per g N/m2: 1.0 EUR/kg N
OBS_VARS = ("DVS", "LAI", "TNSOIL", "NNI", "TRANRF")

@dataclass(frozen=True)
class Scenario:
    recovery: float = 0.7
    years: tuple = tuple(sim.VALID_YEARS[:14])      # 1976-1989

TRAIN = Scenario(0.7, tuple(range(1976, 1990)))
SHIFT_FT = Scenario(0.45, (1992, 1993, 1994, 1995))   # online fine-tuning years (shifted)
SHIFT_EVAL = Scenario(0.45, (1996, 1997, 1998, 1999)) # held-out evaluation years (shifted)

class CropEnv(gym.Env):
    def __init__(self, scenario=TRAIN, seed=0):
        self.sc = scenario
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.action_space = spaces.Discrete(len(DOSES))
        self.fixed_year = None

    def _obs(self):
        e = self.e
        v = [e.get_variable(k) for k in OBS_VARS]
        v = [(1.0 if k == "NNI" else 0.0) if x is None else float(x) for k, x in zip(OBS_VARS, v)]
        o = [self.t / H, v[0] / 2.0, v[1] / 6.0, v[2] / 20.0, v[3], self.n_tot / 30.0, v[4]]
        return np.array(o, np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.year = self.fixed_year or int(self.rng.choice(self.sc.years))
        self.e = sim.make_engine(self.year)
        self.e.run(days=sim.STEP_DAYS)          # reach emergence (Mar 31) before the first decision
        self.t, self.n_tot = 0, 0.0
        return self._obs(), {"year": self.year}

    def step(self, a):
        dose = float(DOSES[int(a)])
        sim.apply_n(self.e, dose, recovery=self.sc.recovery)
        self.n_tot += dose
        r = -COST_N * dose
        self.e.run(days=sim.STEP_DAYS); self.t += 1
        done = self.t >= H or self.e.flag_terminate
        if done:
            if not self.e.flag_terminate: self.e.run_till_terminate()
            wso = self.e.get_output()[-1]["WSO"]
            r += PRICE * float(wso)
            return self._obs() if False else np.zeros(7, np.float32), r, True, False, {"wso": wso, "n_tot": self.n_tot, "year": self.year}
        return self._obs(), r, False, False, {}

def rollout(env, policy, year=None):
    env.fixed_year = year
    o, info = env.reset(); R = 0.0; traj = []
    while True:
        a = policy(o); o2, r, d, _, inf = env.step(a); R += r; traj.append((o, a, r, o2, d))
        o = o2
        if d: break
    env.fixed_year = None
    return R, traj, inf
