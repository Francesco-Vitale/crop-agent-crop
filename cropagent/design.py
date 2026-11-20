"""Part B: sequential N-rate trial design. 2x2: design {fixed grid, Bayesian optimization} x model {flat GP, saturating prior}.

A 'trial' = one season at a randomly drawn weather year with total N applied at week 2 (g N/m2);
observation = simulated yield (g/m2) + Gaussian plot noise. Goal: recommend the N rate maximizing expected net return
(expectation over weather years).

Noise handling: the *data* contain plot noise (SIGMA) plus year-to-year weather variability (from the simulator).
The models are told a constant effective noise sd (`effective_sigma`) that includes both; it is pooled over N, although the
true year-to-year sd grows with N (about 15 g/m2 at N=0 to about 70 g/m2 at N>=24). Passing sigma=SIGMA reproduces the
earlier, overconfident (misspecified) analysis.
"""
import os, numpy as np
from . import sim
from .env import PRICE, COST_N

N_GRID = np.arange(0, 41, 1.0)
SIGMA = 30.0   # plot-to-plot noise sd, g/m2 (about 4% of plateau yield)

def response_table(path="data/response_table.npy"):
    """Y[year, N] simulated yield for all valid years and N_GRID (cached)."""
    if os.path.exists(path): return np.load(path)
    Y = np.array([[sim.run_plan(y, {2: float(n)}) for n in N_GRID] for y in sim.VALID_YEARS])
    os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, Y); return Y

def effective_sigma(Y):
    """Observation-noise sd the models should assume: plot noise + year-to-year variability, pooled over N.
    Calibrated from the simulator's own year-to-year variability; in a real trial series it would have to be estimated
    from historical data or pilot trials."""
    return float(np.sqrt(SIGMA ** 2 + (Y.std(0, ddof=1) ** 2).mean()))

def net(yield_gm2, n): return PRICE * yield_gm2 - COST_N * n

# ---------------- models: posterior mean/sd of the mean YIELD curve on N_GRID ----------------
def rbf(a, b, ls, var): return var * np.exp(-0.5 * ((a[:, None] - b[None, :]) / ls) ** 2)

class FlatGP:
    """Zero-knowledge GP on yield: constant mean = sample mean, RBF kernel with fixed hyperparameters."""
    def __init__(self, sigma=SIGMA, ls=8.0, var=300.0 ** 2): self.sigma, self.ls, self.var = sigma, ls, var
    def posterior(self, X, y):
        if len(X) == 0: return np.full(len(N_GRID), 400.0), np.full(len(N_GRID), np.sqrt(self.var))
        mu0 = np.mean(y); K = rbf(X, X, self.ls, self.var) + self.sigma ** 2 * np.eye(len(X))
        Ks = rbf(N_GRID, X, self.ls, self.var); Kinv = np.linalg.inv(K)
        mean = mu0 + Ks @ Kinv @ (y - mu0)
        var = self.var - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks)
        return mean, np.sqrt(np.maximum(var, 1e-9))

class SaturatingPrior:
    """Human knowledge: yield(N) = b0 + A * (1 - exp(-N / tau)): increasing and saturating when A > 0.
    Priors: tau log-free grid over 2-15 g N/m2; Gaussian b0 ~ N(150, 300^2), A ~ N(500, 600^2).
    NOTE: positivity of A is *favored* by this prior (about 80% of prior mass) but not enforced.
    Exact Bayes on the tau grid (conjugate in b0, A)."""
    taus = np.linspace(2.0, 15.0, 27)
    def __init__(self, sigma=SIGMA): self.sigma = sigma
    def posterior(self, X, y):
        mus, sds, logw = [], [], []
        Pb = np.diag([1 / 300.0 ** 2, 1 / 600.0 ** 2]); m0 = np.array([150.0, 500.0]); s2 = self.sigma ** 2
        for tau in self.taus:
            G = lambda n: np.stack([np.ones_like(n), 1 - np.exp(-n / tau)], 1)
            Gg = G(N_GRID)
            if len(X):
                Gx = G(X); P = Pb + Gx.T @ Gx / s2; S = np.linalg.inv(P)
                m = S @ (Pb @ m0 + Gx.T @ y / s2)
                C = s2 * np.eye(len(X)) + Gx @ np.diag([300.0 ** 2, 600.0 ** 2]) @ Gx.T
                r = y - Gx @ m0; _, ld = np.linalg.slogdet(C); lw = -0.5 * (r @ np.linalg.solve(C, r) + ld)
            else:
                S = np.linalg.inv(Pb); m = m0; lw = 0.0
            mus.append(Gg @ m); sds.append(np.einsum("ij,jk,ik->i", Gg, S, Gg)); logw.append(lw)
        w = np.exp(np.array(logw) - max(logw)); w /= w.sum()
        mus, sds = np.array(mus), np.array(sds)
        mean = w @ mus; var = w @ (sds + mus ** 2) - mean ** 2
        return mean, np.sqrt(np.maximum(var, 1e-9))

# ---------------- designs ----------------
GRID_LEVELS = np.array([0, 8, 16, 24, 32, 40.0])
def grid_choice(n_done): return GRID_LEVELS[n_done % len(GRID_LEVELS)]

def bo_choice(model, X, y, rng):
    """Thompson sampling on the net-return surface (independent marginal draws: simple, not tuned)."""
    mean, sd = model.posterior(X, y)
    draw = net(mean + sd * rng.standard_normal(len(N_GRID)), N_GRID)
    return N_GRID[int(np.argmax(draw))]

def trial_stream(seed, n_trials, n_years):
    """Weather years and plot-noise draws for one repetition. Drawn from their own generator, so every method
    sees the identical sequence regardless of how many random numbers its design consumes (common random numbers)."""
    r = np.random.default_rng([seed, 1])
    return r.integers(n_years, size=n_trials), SIGMA * r.standard_normal(n_trials)

def run_method(Y, design, model, n_trials, seed):
    """Per-trial regret of the recommended N and RMSE of the posterior-mean net curve vs truth.
    Returns (regret, rmse, years_used, plot_noise_used)."""
    years, noise = trial_stream(seed, n_trials, Y.shape[0])
    alg_rng = np.random.default_rng([seed, 2])            # design-only randomness (Thompson draws)
    true_net = net(Y.mean(0), N_GRID); best = true_net.max()
    X, y, reg, rmse = [], [], [], []
    for t in range(n_trials):
        n = grid_choice(t) if design == "grid" else bo_choice(model, np.array(X), np.array(y), alg_rng)
        X.append(n); y.append(Y[years[t], int(n)] + noise[t])
        mean, _ = model.posterior(np.array(X), np.array(y))
        rec = int(np.argmax(net(mean, N_GRID)))
        reg.append(best - true_net[rec]); rmse.append(float(np.sqrt(np.mean((net(mean, N_GRID) - true_net) ** 2))))
    return np.array(reg), np.array(rmse), years, noise

def make_methods(sigma):
    return {"fixed grid + flat GP": ("grid", lambda: FlatGP(sigma)), "BO + flat GP": ("bo", lambda: FlatGP(sigma)),
            "fixed grid + saturating prior": ("grid", lambda: SaturatingPrior(sigma)),
            "BO + saturating prior": ("bo", lambda: SaturatingPrior(sigma))}
