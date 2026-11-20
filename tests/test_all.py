import warnings; warnings.filterwarnings("ignore")
import numpy as np, pytest
from cropagent import sim
from cropagent.env import CropEnv, TRAIN, SHIFT_EVAL, H, DOSES, rollout
from cropagent.data import rule_policy_factory
from cropagent.design import FlatGP, SaturatingPrior, run_method, trial_stream, N_GRID, net

def test_simulator_n_response_saturates():
    y0, y16, y30 = (sim.run_plan(1985, {2: n}) for n in (0.0, 16.0, 30.0))
    assert y16 > 3 * y0                      # strong response to N
    assert abs(y30 - y16) / y16 < 0.1        # plateau: extra N adds little

def test_late_n_is_less_effective():
    assert sim.run_plan(1985, {2: 15.0}) > 1.5 * sim.run_plan(1985, {10: 15.0})

def test_env_episode_length_and_determinism():
    env = CropEnv(TRAIN); acts = [0, 4, 0, 0, 4] + [0] * 7
    R = []
    for _ in range(2):
        it = iter(acts); r, traj, info = rollout(env, lambda o: next(it), year=1985); R.append(r)
        assert len(traj) == H and traj[-1][4] is True
    assert R[0] == R[1]                       # same year + same actions -> same return

def test_shift_reduces_return_for_same_policy():
    acts = [0, 4, 0, 0, 4] + [0] * 7
    env_a, env_b = CropEnv(TRAIN), CropEnv(SHIFT_EVAL)
    it = iter(acts); ra = rollout(env_a, lambda o: next(it), year=1996)[0]
    it = iter(acts); rb = rollout(env_b, lambda o: next(it), year=1996)[0]
    assert rb < ra                            # lower fertilizer recovery -> less yield

def test_behavior_policy_actions_valid_and_noisy():
    rng = np.random.default_rng(0); env = CropEnv(TRAIN); totals = []
    for _ in range(6):
        _, traj, info = rollout(env, rule_policy_factory(rng), year=1980)
        assert all(0 <= a < len(DOSES) for _, a, *_ in traj); totals.append(info["n_tot"])
    assert len(set(totals)) > 1               # noise makes trials differ

def test_saturating_prior_learns_curve_and_shrinks_uncertainty():
    rng = np.random.default_rng(0); X = np.tile([0., 4, 8, 12, 20, 30], 3)
    truth = lambda n: 100 + 700 * (1 - np.exp(-n / 6))
    y = truth(X) + 10 * rng.standard_normal(len(X)); m = SaturatingPrior(sigma=10.0)
    mean0, sd0 = m.posterior(np.array([]), np.array([])); mean, sd = m.posterior(X, y)
    assert np.abs(mean - truth(N_GRID)).max() < 40 and sd.mean() < sd0.mean()

def _toy_Y():
    return np.array([[100 + 700 * (1 - np.exp(-n / 6)) + 20 * k for n in N_GRID] for k in range(5)])

def test_run_method_reproducible_and_regret_nonnegative():
    Y = _toy_Y()
    a = run_method(Y, "bo", FlatGP(), 6, 1); b = run_method(Y, "bo", FlatGP(), 6, 1)
    assert np.allclose(a[0], b[0]) and (a[0] >= -1e-9).all()

def test_common_random_numbers_across_designs():
    """Fixed grid and BO must see identical weather years and noise draws for the same repetition seed."""
    Y = _toy_Y()
    _, _, years_grid, noise_grid = run_method(Y, "grid", FlatGP(), 8, 7)
    _, _, years_bo, noise_bo = run_method(Y, "bo", FlatGP(), 8, 7)
    assert np.array_equal(years_grid, years_bo) and np.allclose(noise_grid, noise_bo)
    y1, n1 = trial_stream(7, 8, 5); y2, n2 = trial_stream(7, 8, 5)
    assert np.array_equal(y1, y2) and np.allclose(n1, n2)

def test_bc_to_q_copies_weights():
    import torch
    from cropagent.agents import make_bc, make_dqn, seed_all
    from cropagent.finetune import bc_to_q
    from d3rlpy.dataset import MDPDataset
    O = np.random.rand(60, 7).astype(np.float32); A = np.random.randint(0, 6, 60); R = np.random.rand(60).astype(np.float32)
    T = np.zeros(60, np.float32); T[11::12] = 1
    ds = MDPDataset(O, A, R, T, np.zeros(60, np.float32)); seed_all(0)
    bc, q = make_bc(0), make_dqn(0); bc.build_with_dataset(ds); q.build_with_dataset(ds); bc_to_q(bc, q)
    w_bc = bc.impl.modules.imitator._fc.weight; w_q = q.impl.modules.q_funcs[0]._fc.weight
    assert torch.allclose(w_bc, w_q)

def test_pipeline_integration_tiny():
    """Generate a tiny historical dataset, train CQL for a few steps, run 2 online episodes, evaluate."""
    from cropagent.data import generate, to_mdp
    from cropagent.agents import make_cql, fit_offline, seed_all
    from cropagent.finetune import online_finetune
    arr, mr = generate(n_per_year=1, seed=0)
    assert arr[0].shape == (14 * H, 7) and arr[3].sum() == 14        # one terminal per episode
    seed_all(0); cql = make_cql(0, alpha=0.3); fit_offline(cql, to_mdp(arr), 20)
    curve = online_finetune(cql, (arr[0], arr[1], arr[2], arr[3]), n_episodes=2, seed=0, eval_every=1, updates_per_ep=3)
    assert [c[0] for c in curve] == [0, 1, 2] and all(np.isfinite(c[1]) for c in curve)
