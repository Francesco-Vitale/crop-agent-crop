"""Sensitivity of Part B to (a) the noise level the models assume and (b) the fixed-grid levels."""
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np
from cropagent import design as D
Y = D.response_table(); out = {}; sig_eff = D.effective_sigma(Y)
def go(tag, sigma, methods=None):
    for name, (design, M) in D.make_methods(sigma).items():
        if methods and name not in methods: continue
        R = np.array([D.run_method(Y, design, M(), 24, 1000 + r)[0] for r in range(100)])
        out[f"{tag} | {name}"] = [float(R[:, 5].mean()), float(R[:, 11].mean()), float(R[:, -1].mean()), float(R[:, -1].std(ddof=1) / 10)]
        print(f"{tag:34s} {name:30s} regret@6 {R[:,5].mean():.3f} @12 {R[:,11].mean():.3f} @24 {R[:,-1].mean():.3f}", flush=True)
go(f"model sigma {sig_eff:.0f} (calibrated)", sig_eff)
go("model sigma 30 (earlier, overconfident)", D.SIGMA)
for levels in ([0, 6, 12, 18, 24, 30, 40], [3, 10, 19, 27, 35, 40]):
    D.GRID_LEVELS = np.array(levels, float)
    go(f"grid levels {levels}, calibrated", sig_eff, methods=("fixed grid + flat GP", "fixed grid + saturating prior"))
json.dump(out, open("results/partB_sensitivity.json", "w"), indent=1)
