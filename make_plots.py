import glob, json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ref = json.load(open("results/reference.json"))
files = sorted(glob.glob("results/partA_seed*.json")); runs = [json.load(open(f)) for f in files]
def stack(key, col=1): return np.array([[row[col] for row in r[key]] for r in runs])
eps = [row[0] for row in runs[0]["scratch"]]
se = lambda a: a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else np.zeros(a.shape[1])
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
A = ax[0]
styles = [("scratch", "From scratch (DQN)", "tab:gray", "-"), ("offline_only", "Offline only (CQL)", "tab:blue", "-"),
          ("offline_ft", "Offline + fine-tune (CQL, penalty kept)", "tab:orange", "-"),
          ("bc_ft", "BC + fine-tune (BC-initialized DQN)", "tab:green", "-"),
          ("offline_ft_alpha0", "Extra: offline + fine-tune, penalty off online", "tab:red", "--")]
summary = {}
for key, lab, c, ls in styles:
    a = stack(key); m = a.mean(0); s = se(a)
    A.plot(eps, m, ls, color=c, label=lab, lw=2); A.fill_between(eps, m - s, m + s, color=c, alpha=0.18)
    summary[key] = (float(m[0]), float(m[-1]), float(s[-1]), float(a[:, -1].min()), float(a[:, -1].max()))
A.axhline(ref["shift_eval"][0], color="k", ls=":", label=f"Best fixed 3-dose plan, tuned on eval years ({ref['shift_eval'][0]:.1f})")
A.set_xlabel("Online episodes (shifted scenario: fertilizer recovery 0.45)"); A.set_ylabel("Return on held-out shifted years (100 EUR/ha)")
A.set_title(f"A. Offline-to-online under distribution shift (mean ± SE, {len(runs)} seeds)"); A.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, framealpha=0.9); A.set_ylim(0, 15.5); A.grid(alpha=.3)
try:
    B = json.load(open("results/partB.json")); nt = np.arange(1, B["n_trials"] + 1)
    cols = {"fixed grid + flat GP": "tab:gray", "BO + flat GP": "tab:blue", "fixed grid + saturating prior": "tab:orange", "BO + saturating prior": "tab:red"}
    for name, c in cols.items():
        r = np.array(B[name]["regret"]); m = r.mean(0); s = r.std(0, ddof=1) / np.sqrt(len(r))
        ax[1].plot(nt, m, color=c, lw=2, label=name); ax[1].fill_between(nt, m - s, m + s, color=c, alpha=0.18)
    ax[1].set_yscale("log"); ax[1].set_xlabel("Number of N-rate trials"); ax[1].set_ylabel("Regret of recommended N rate (100 EUR/ha)")
    ax[1].set_title(f"B. Trial design: regret vs trials (mean ± SE, {B['reps']} reps)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
except FileNotFoundError:
    ax[1].text(0.5, 0.5, "Part B not run", ha="center")
plt.tight_layout(); plt.savefig("results/main.png", dpi=150, bbox_inches="tight")
json.dump({"partA_summary(start,final_mean,final_SE,min,max)": summary, "n_seeds": len(runs)}, open("results/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
